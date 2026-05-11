// ==========================================
// 動画分析 (Video Analysis) - Splyzaライク
// ローカル動画の再生・コマ送り・描画・タグ記録・2動画比較・ポーズ推定
// ==========================================
import { createStepIndicator } from '../utils.js';
import {
    createAnnotationState,
    renderAnnotations,
    attachDrawingHandlers,
    clearAnnotations,
    undoLastAnnotation,
    exportAnnotationsJSON,
    importAnnotationsJSON,
    ANNOTATION_TOOLS
} from './video_analysis/annotation.js';
import {
    ensurePoseDetector,
    detectPoseOnce,
    drawPoseOverlay,
    computeJointAngles,
    startContinuousDetection,
    exportPoseSeriesCSV,
    POSE_ANGLE_DEFS
} from './video_analysis/pose.js';

const STEPS = ['メディア準備', '解析・記録', '出力'];

const THEME = '#dc2626';
const THEME_LIGHT = 'rgba(220, 38, 38, 0.08)';
const THEME_BORDER = 'rgba(220, 38, 38, 0.35)';
const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const FRAME_STEP_SEC = 1 / 30;
const ACCEPTED_VIDEO_TYPES = ['video/mp4', 'video/webm', 'video/quicktime', 'video/ogg'];
const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'];
const VIDEO_EXT = /\.(mp4|webm|mov|ogg|m4v)$/i;
const IMAGE_EXT = /\.(jpe?g|png|webp|gif|bmp)$/i;

const TOOL_LABELS = {
    pen: 'フリーハンド',
    line: '直線',
    arrow: '矢印',
    circle: '円',
    rect: '四角'
};
const TOOL_ICONS = {
    pen: 'fa-pen',
    line: 'fa-slash',
    arrow: 'fa-long-arrow-alt-right',
    circle: 'fa-circle',
    rect: 'fa-square'
};

const DEFAULT_TAG_DEFS = [
    { id: 'tag-good', label: 'グッドプレー', color: '#16a34a', hotkey: '1' },
    { id: 'tag-error', label: 'ミス', color: '#dc2626', hotkey: '2' },
    { id: 'tag-shot', label: 'シュート', color: '#2563eb', hotkey: '3' },
    { id: 'tag-pass', label: 'パス', color: '#f59e0b', hotkey: '4' }
];

let _idSeq = 1;
function _newId(prefix) { return `${prefix}-${Date.now().toString(36)}-${_idSeq++}`; }

let _state = null;
let _keysBound = false;

// ==========================================
// Render Entry
// ==========================================
export function render(container) {
    _state = _createInitialState();
    container.innerHTML = _buildLayout();
    _bindGlobalUI(container);
    _bindKeyboardShortcuts();
    _renderTargetPanel('A');
    _renderTargetPanel('B');
    _renderToolPanes();
    _switchMode('single');
    _switchTool('draw');
    _startRenderLoop();
}

function _createInitialState() {
    return {
        mode: 'single',
        active: 'A',
        syncEnabled: true,
        syncOffset: 0,
        tagDefs: DEFAULT_TAG_DEFS.map(t => ({ ...t })),
        targets: {
            A: _createTargetState(),
            B: _createTargetState()
        },
        rafId: null,
        sampleListeners: { A: null, B: null }
    };
}

function _createTargetState() {
    return {
        file: null,
        url: null,
        mediaType: null,   // 'video' | 'image' | null
        media: null,       // 現在アクティブな <video> または <img>
        video: null,
        image: null,
        canvas: null,
        ctx: null,
        annotation: createAnnotationState(),
        tags: [],
        pose: {
            samples: [],
            latestPose: null,
            showOverlay: true,
            continuous: null,
            samplingMs: 200
        },
        detachDrawing: null
    };
}

function _isVideoFile(file) {
    if (!file) return false;
    if (ACCEPTED_VIDEO_TYPES.includes(file.type)) return true;
    if (file.type && file.type.startsWith('video/')) return true;
    return VIDEO_EXT.test(file.name || '');
}

function _isImageFile(file) {
    if (!file) return false;
    if (ACCEPTED_IMAGE_TYPES.includes(file.type)) return true;
    if (file.type && file.type.startsWith('image/')) return true;
    return IMAGE_EXT.test(file.name || '');
}

function _detectMediaType(file) {
    if (_isVideoFile(file)) return 'video';
    if (_isImageFile(file)) return 'image';
    return null;
}

function _getMediaTime(t) {
    if (!t) return 0;
    if (t.mediaType === 'video' && t.video) return t.video.currentTime;
    return 0;
}

// ==========================================
// Layout
// ==========================================
function _buildLayout() {
    return `
    <div class="va-root" style="--va-theme: ${THEME}; --va-theme-light: ${THEME_LIGHT}; --va-theme-border: ${THEME_BORDER};">
        <h2><i class="fas fa-film" style="color: ${THEME};"></i> 動画分析ツール</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            ローカル動画・画像をブラウザで読み込み、コマ送り再生・描画アノテーション・タグ記録・2メディア比較・MoveNet ポーズ推定をブラウザ内で完結します。
            ファイルは外部に送信されません。Splyza のような行動・スポーツ動画解析を、教育用途で手軽に試せます。
        </p>

        <div class="step-indicator-wrapper">${createStepIndicator(STEPS, 0)}</div>

        <!-- Step 1: メディア準備 -->
        <div class="model-config">
            <h3><i class="fas fa-folder-open" style="color: ${THEME};"></i> Step 1: メディアを開く</h3>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0.25rem 0 1rem;">
                動画 (mp4 / webm / mov / ogg) または画像 (jpg / png / webp / gif / bmp) を読み込みます。
                <strong>2メディア比較</strong>に切り替えると、左右並べて同時に分析できます。
            </p>
            <div class="va-mode-toggle" role="tablist" style="margin-bottom: 1rem;">
                <button class="va-mode-btn active" data-mode="single"><i class="fas fa-square"></i> 単一メディア</button>
                <button class="va-mode-btn" data-mode="compare"><i class="fas fa-clone"></i> 2メディア比較</button>
            </div>

            <div class="va-stage va-stage-single">
                ${_buildPanelHTML('A')}
                ${_buildPanelHTML('B')}
            </div>

            <div class="va-sync-bar" style="display:none; margin-top: 1rem;">
                <label class="va-inline">
                    <input type="checkbox" id="va-sync-enabled" checked>
                    同期再生
                </label>
                <label class="va-inline">
                    オフセット (B - A)
                    <input type="number" id="va-sync-offset" value="0" step="0.1" style="width:5rem;">
                    秒
                </label>
                <button class="va-btn" id="va-sync-play"><i class="fas fa-play"></i> 同期再生</button>
                <button class="va-btn" id="va-sync-pause"><i class="fas fa-pause"></i> 同期停止</button>
                <button class="va-btn" id="va-sync-zero"><i class="fas fa-undo"></i> 両方0秒へ</button>
                <span class="va-hint" id="va-sync-note">A の現在時刻を基準に、B はオフセットだけずらして再生します。</span>
            </div>
        </div>

        <!-- Step 2: 解析・記録 -->
        <div class="model-config">
            <h3><i class="fas fa-pen-to-square" style="color: ${THEME};"></i> Step 2: 解析・記録</h3>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0.25rem 0 1rem;">
                描画でフォームを示したり、タグでイベントを打刻したり、ポーズ推定で骨格を観察します。
                ツールはタブで切り替えできます。
            </p>
            <div class="tab-container" id="va-tool-tabs">
                <button class="tab-btn active" data-tool="draw"><i class="fas fa-pen"></i> 描画</button>
                <button class="tab-btn" data-tool="tag"><i class="fas fa-tags"></i> タグ / イベント</button>
                <button class="tab-btn" data-tool="pose"><i class="fas fa-person-running"></i> ポーズ推定</button>
            </div>
            <div class="tab-content active" data-tool="draw" id="va-pane-draw"></div>
            <div class="tab-content" data-tool="tag" id="va-pane-tag"></div>
            <div class="tab-content" data-tool="pose" id="va-pane-pose"></div>
        </div>

        <!-- Step 3: 出力・サマリー -->
        <div class="model-config">
            <h3><i class="fas fa-download" style="color: ${THEME};"></i> Step 3: サマリー・出力</h3>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0.25rem 0 1rem;">
                記録した描画・タグ・ポーズサンプルをファイルに書き出します。動画ファイルとは別に保管できます。
            </p>
            <div class="metrics-grid" id="va-summary-grid"></div>
            <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem;">
                <button class="btn-analysis" id="va-export-tags-csv" style="background: ${THEME}; flex: 1; min-width: 220px;">
                    <i class="fas fa-file-csv"></i> タグを CSV に保存
                </button>
                <button class="btn-analysis" id="va-export-anno-json" style="background: ${THEME}; flex: 1; min-width: 220px;">
                    <i class="fas fa-file-code"></i> 描画を JSON に保存
                </button>
                <button class="btn-analysis" id="va-export-pose-csv" style="background: ${THEME}; flex: 1; min-width: 220px;">
                    <i class="fas fa-file-csv"></i> 関節角度を CSV に保存
                </button>
            </div>
            <div class="va-keyboard-help" style="margin-top: 1.25rem; padding: 0.85rem 1rem; background: var(--surface); border: 1px solid var(--border-color); border-radius: 8px;">
                <p style="margin:0 0 0.4rem; font-size: 0.85rem; font-weight: 600; color: var(--text-primary);">
                    <i class="fas fa-keyboard" style="color: ${THEME};"></i> キーボードショートカット
                </p>
                <p style="margin:0; font-size: 0.8rem; color: var(--text-secondary);">
                    <kbd>Space</kbd> 再生/停止 ・ <kbd>,</kbd>/<kbd>.</kbd> コマ送り ・ <kbd>1</kbd>〜<kbd>9</kbd> タグ打刻 ・ <kbd>A</kbd>/<kbd>B</kbd> ツール対象切替
                </p>
            </div>
        </div>
    </div>
    `;
}

function _buildPanelHTML(target) {
    const label = target === 'A' ? 'メディアA' : 'メディアB';
    return `
    <div class="va-panel" data-target="${target}">
        <div class="va-panel-header">
            <div class="va-panel-titlewrap">
                <span class="va-panel-title"><i class="fas fa-photo-film"></i> ${label} <span class="va-media-tag" data-target="${target}"></span></span>
                <span class="va-media-info" data-target="${target}"></span>
            </div>
            <div class="va-panel-actions">
                <input type="file" accept="video/*,image/*" id="va-file-${target}" style="display:none;">
                <button class="va-btn va-btn-primary" data-action="open-file" data-target="${target}">
                    <i class="fas fa-folder-open"></i> ファイルを開く
                </button>
                <button class="va-btn va-active-toggle" data-action="set-active" data-target="${target}" title="このメディアを描画/タグ/ポーズ推定の対象にする">
                    <i class="fas fa-crosshairs"></i> ツール対象
                </button>
                <button class="va-btn va-btn-ghost" data-action="clear" data-target="${target}" disabled title="メディアと描画・タグを破棄">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>

        <div class="va-drop-zone" data-target="${target}">
            <div class="va-drop-inner">
                <div class="va-drop-icon"><i class="fas fa-cloud-upload-alt"></i></div>
                <p class="va-drop-headline">動画 / 画像をドラッグ＆ドロップ</p>
                <p class="va-drop-sub">mp4 / webm / mov / ogg ・ jpg / png / webp / gif</p>
                <button class="va-btn va-btn-primary va-drop-cta" data-action="open-file" data-target="${target}">
                    <i class="fas fa-folder-open"></i> ファイルを選択
                </button>
            </div>
        </div>

        <div class="va-video-wrap" data-target="${target}" style="display:none;">
            <video data-target="${target}" preload="metadata" playsinline></video>
            <img data-target="${target}" alt="" style="display:none;">
            <canvas data-target="${target}"></canvas>
            <div class="va-active-badge" data-target="${target}"><i class="fas fa-circle-dot"></i> アクティブ</div>
            <div class="va-image-badge" data-target="${target}" style="display:none;"><i class="fas fa-image"></i> 静止画</div>
        </div>

        <div class="va-controls" data-target="${target}" style="display:none;">
            <input type="range" class="va-seek" data-target="${target}" min="0" max="0" step="0.001" value="0">
            <div class="va-controls-row">
                <button class="va-btn va-btn-play" data-action="play-toggle" data-target="${target}" title="再生/一時停止 (Space)">
                    <i class="fas fa-play"></i>
                </button>
                <div class="va-controls-group va-only-video" data-target="${target}">
                    <button class="va-btn va-btn-sm" data-action="frame-prev" data-target="${target}" title="前フレーム (,)"><i class="fas fa-backward-step"></i></button>
                    <button class="va-btn va-btn-sm" data-action="frame-next" data-target="${target}" title="次フレーム (.)"><i class="fas fa-forward-step"></i></button>
                </div>
                <span class="va-time" data-target="${target}">00:00.000 / 00:00.000</span>
                <label class="va-inline va-only-video">
                    <i class="fas fa-gauge-high"></i>
                    <select data-action="speed" data-target="${target}" aria-label="再生速度">
                        ${PLAYBACK_SPEEDS.map(s => `<option value="${s}" ${s === 1 ? 'selected' : ''}>${s}x</option>`).join('')}
                    </select>
                </label>
                <label class="va-inline va-volume va-only-video">
                    <i class="fas fa-volume-high"></i>
                    <input type="range" min="0" max="1" step="0.05" value="1" data-action="volume" data-target="${target}" aria-label="音量">
                </label>
            </div>
            <div class="va-tag-strip" data-target="${target}"></div>
        </div>
    </div>
    `;
}

// ==========================================
// UI Binding
// ==========================================
function _bindGlobalUI(root) {
    root.querySelectorAll('.va-mode-btn').forEach(btn => {
        btn.addEventListener('click', () => _switchMode(btn.dataset.mode));
    });
    const exportTagsBtn = root.querySelector('#va-export-tags-csv');
    const exportAnnoBtn = root.querySelector('#va-export-anno-json');
    const exportPoseBtn = root.querySelector('#va-export-pose-csv');
    if (exportTagsBtn) exportTagsBtn.addEventListener('click', _exportTagsCSV);
    if (exportAnnoBtn) exportAnnoBtn.addEventListener('click', () => {
        const json = exportAnnotationsJSON(_activeAnnotationState());
        _downloadFile(`annotations_${_state.active}.json`, json, 'application/json');
    });
    if (exportPoseBtn) exportPoseBtn.addEventListener('click', () => {
        const samples = _state.targets[_state.active].pose.samples;
        if (samples.length === 0) {
            alert('まだサンプルがありません。連続推定または「現在フレームを推定」を実行してください。');
            return;
        }
        _downloadFile(`pose_angles_${_state.active}.csv`, exportPoseSeriesCSV(samples), 'text/csv');
    });
    root.querySelectorAll('#va-tool-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', () => _switchTool(btn.dataset.tool));
    });

    root.querySelector('#va-sync-enabled').addEventListener('change', (e) => {
        _state.syncEnabled = e.target.checked;
    });
    root.querySelector('#va-sync-offset').addEventListener('input', (e) => {
        const v = Number(e.target.value);
        _state.syncOffset = Number.isFinite(v) ? v : 0;
    });
    root.querySelector('#va-sync-play').addEventListener('click', () => {
        const tA = _state.targets.A;
        const tB = _state.targets.B;
        if (tA.mediaType !== 'video' || tB.mediaType !== 'video') return;
        const a = tA.video, b = tB.video;
        if (a) a.play().catch(() => {});
        if (b && _state.syncEnabled) {
            const t = Math.max(0, (a ? a.currentTime : 0) + _state.syncOffset);
            b.currentTime = Math.min(t, isFinite(b.duration) ? b.duration : t);
            b.play().catch(() => {});
        }
    });
    root.querySelector('#va-sync-pause').addEventListener('click', () => {
        const a = _state.targets.A.video, b = _state.targets.B.video;
        if (a && _state.targets.A.mediaType === 'video') a.pause();
        if (b && _state.targets.B.mediaType === 'video') b.pause();
    });
    root.querySelector('#va-sync-zero').addEventListener('click', () => {
        const a = _state.targets.A.video, b = _state.targets.B.video;
        if (a && _state.targets.A.mediaType === 'video') a.currentTime = 0;
        if (b && _state.targets.B.mediaType === 'video') b.currentTime = Math.max(0, _state.syncOffset);
    });
}

function _renderTargetPanel(target) {
    const root = document.querySelector('.va-root');

    const fileInput = root.querySelector(`#va-file-${target}`);
    const openBtns = root.querySelectorAll(`[data-action="open-file"][data-target="${target}"]`);
    const clearBtn = root.querySelector(`[data-action="clear"][data-target="${target}"]`);
    const dropZone = root.querySelector(`.va-drop-zone[data-target="${target}"]`);
    const videoEl = root.querySelector(`video[data-target="${target}"]`);
    const imageEl = root.querySelector(`img[data-target="${target}"]`);
    const canvasEl = root.querySelector(`canvas[data-target="${target}"]`);
    const seek = root.querySelector(`.va-seek[data-target="${target}"]`);
    const wrap = root.querySelector(`.va-video-wrap[data-target="${target}"]`);

    const getT = () => _state.targets[target];

    openBtns.forEach(btn => btn.addEventListener('click', () => fileInput.click()));
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) _loadMedia(target, e.target.files[0]);
        e.target.value = '';
    });
    clearBtn.addEventListener('click', () => _clearTarget(target));

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) _loadMedia(target, file);
    });

    // Video element listeners (attach ONCE - the same <video> is reused across loads)
    videoEl.addEventListener('loadedmetadata', () => {
        if (getT().mediaType !== 'video') return;
        canvasEl.width = videoEl.videoWidth || 1280;
        canvasEl.height = videoEl.videoHeight || 720;
        if (videoEl.videoWidth && videoEl.videoHeight) {
            const ratio = videoEl.videoWidth / videoEl.videoHeight;
            wrap.style.aspectRatio = `${videoEl.videoWidth} / ${videoEl.videoHeight}`;
            wrap.style.maxWidth = ratio < 1 ? `calc(70vh * ${ratio})` : '';
        }
        seek.max = String(videoEl.duration || 0);
        _updateTimeDisplay(target);
        _refreshTagStrip(target);
        _updateMediaInfo(target);
    });
    videoEl.addEventListener('timeupdate', () => {
        if (!getT().url || getT().mediaType !== 'video') return;
        _updateTimeDisplay(target);
        if (document.activeElement !== seek) seek.value = String(videoEl.currentTime);
        if (target === 'A' && _state.mode === 'compare' && _state.syncEnabled) _syncFromA();
    });
    videoEl.addEventListener('play', () => {
        _refreshPlayToggle(target);
        if (target === 'A' && _state.mode === 'compare' && _state.syncEnabled) {
            const tb = _state.targets.B;
            if (tb && tb.mediaType === 'video' && tb.video && tb.url) tb.video.play().catch(() => {});
        }
    });
    videoEl.addEventListener('pause', () => {
        _refreshPlayToggle(target);
        if (target === 'A' && _state.mode === 'compare' && _state.syncEnabled) {
            const tb = _state.targets.B;
            if (tb && tb.mediaType === 'video' && tb.video && tb.url) tb.video.pause();
        }
    });
    videoEl.addEventListener('ratechange', () => {
        if (target === 'A' && _state.mode === 'compare' && _state.syncEnabled) {
            const tb = _state.targets.B;
            if (tb && tb.mediaType === 'video' && tb.video && tb.url) tb.video.playbackRate = videoEl.playbackRate;
        }
    });

    // Image element listener (attach ONCE)
    imageEl.addEventListener('load', () => {
        if (getT().mediaType !== 'image') return;
        canvasEl.width = imageEl.naturalWidth || 1280;
        canvasEl.height = imageEl.naturalHeight || 720;
        if (imageEl.naturalWidth && imageEl.naturalHeight) {
            const ratio = imageEl.naturalWidth / imageEl.naturalHeight;
            wrap.style.aspectRatio = `${imageEl.naturalWidth} / ${imageEl.naturalHeight}`;
            // 縦長画像で横方向に余白が出すぎないよう、max-height(70vh) と整合する max-width を設定
            wrap.style.maxWidth = ratio < 1 ? `calc(70vh * ${ratio})` : '';
        } else {
            wrap.style.maxWidth = '';
        }
        _updateTimeDisplay(target);
        _updateMediaInfo(target);
    });

    // Control handlers (read state dynamically)
    const ctrl = (action) => root.querySelector(`[data-action="${action}"][data-target="${target}"]`);
    ctrl('play-toggle').addEventListener('click', () => {
        const t = getT();
        if (t.mediaType !== 'video' || !t.video) return;
        if (t.video.paused) t.video.play().catch(() => {});
        else t.video.pause();
    });
    ctrl('frame-prev').addEventListener('click', () => _stepFrame(target, -1));
    ctrl('frame-next').addEventListener('click', () => _stepFrame(target, +1));
    ctrl('speed').addEventListener('change', (e) => {
        const t = getT();
        if (t.video) t.video.playbackRate = Number(e.target.value);
    });
    ctrl('volume').addEventListener('input', (e) => {
        const t = getT();
        if (t.video) t.video.volume = Number(e.target.value);
    });
    ctrl('set-active').addEventListener('click', () => _setActiveTarget(target));

    seek.addEventListener('input', (e) => {
        const t = getT();
        if (t.mediaType === 'video' && t.video && Number.isFinite(t.video.duration)) {
            t.video.currentTime = Number(e.target.value);
        }
    });

    wrap.addEventListener('click', (e) => {
        if (e.target.tagName !== 'CANVAS') _setActiveTarget(target);
    });
}

// ==========================================
// File Loading
// ==========================================
function _loadMedia(target, file) {
    const mediaType = _detectMediaType(file);
    if (!mediaType) {
        alert('対応していないファイル形式です。動画 (mp4/webm/mov/ogg) または 画像 (jpg/png/webp/gif/bmp) をご利用ください。');
        return;
    }
    const t = _state.targets[target];
    if (t.url) URL.revokeObjectURL(t.url);
    if (t.detachDrawing) { t.detachDrawing(); t.detachDrawing = null; }
    if (t.pose.continuous) { t.pose.continuous.stop(); t.pose.continuous = null; }
    t.file = file;
    t.url = URL.createObjectURL(file);
    t.mediaType = mediaType;
    t.annotation = createAnnotationState();
    t.tags = [];
    t.pose.samples = [];
    t.pose.latestPose = null;

    const root = document.querySelector('.va-root');
    const videoEl = root.querySelector(`video[data-target="${target}"]`);
    const imageEl = root.querySelector(`img[data-target="${target}"]`);
    const canvasEl = root.querySelector(`canvas[data-target="${target}"]`);
    const wrap = root.querySelector(`.va-video-wrap[data-target="${target}"]`);
    const controls = root.querySelector(`.va-controls[data-target="${target}"]`);
    const dropZone = root.querySelector(`.va-drop-zone[data-target="${target}"]`);

    if (mediaType === 'video') {
        imageEl.style.display = 'none';
        imageEl.removeAttribute('src');
        videoEl.style.display = 'block';
        videoEl.pause();
        videoEl.src = t.url;
        videoEl.load();
        t.video = videoEl;
        t.image = null;
        t.media = videoEl;
        wrap.classList.remove('is-image');
        controls.classList.remove('is-image');
        controls.style.display = 'flex';
        wrap.style.aspectRatio = '16 / 9';
    } else {
        videoEl.style.display = 'none';
        videoEl.pause();
        videoEl.removeAttribute('src');
        videoEl.load();
        imageEl.style.display = 'block';
        imageEl.src = t.url;
        t.video = null;
        t.image = imageEl;
        t.media = imageEl;
        wrap.classList.add('is-image');
        controls.classList.add('is-image');
        // 画像時は再生コントロール一式を非表示にする（行ごと隠す）
        controls.style.display = 'none';
        wrap.style.aspectRatio = '';
    }

    t.canvas = canvasEl;
    t.ctx = canvasEl.getContext('2d');

    // Drawing handlers (re-attach for new annotation state)
    t.detachDrawing = attachDrawingHandlers(
        canvasEl,
        t.annotation,
        () => _getMediaTime(t),
        _refreshSummary
    );

    // UI swap
    dropZone.style.display = 'none';
    wrap.style.display = 'block';
    root.querySelector(`[data-action="clear"][data-target="${target}"]`).disabled = false;

    // メディアタグ表示
    const tagEl = root.querySelector(`.va-media-tag[data-target="${target}"]`);
    if (tagEl) {
        tagEl.textContent = mediaType === 'video' ? '動画' : '画像';
        tagEl.className = `va-media-tag va-media-tag-${mediaType}`;
        tagEl.dataset.target = target;
    }
    const imageBadge = root.querySelector(`.va-image-badge[data-target="${target}"]`);
    if (imageBadge) imageBadge.style.display = mediaType === 'image' ? 'inline-flex' : 'none';

    _setActiveTarget(target);
    _refreshPlayToggle(target);
    _refreshTagList();
    _refreshTagStrip(target);
    _refreshAnnotationPaneCounts();
    _refreshSyncBarState();
    _updateMediaInfo(target);
}

function _clearTarget(target) {
    const t = _state.targets[target];
    if (t.url) URL.revokeObjectURL(t.url);
    if (t.detachDrawing) { t.detachDrawing(); t.detachDrawing = null; }
    if (t.pose.continuous) { t.pose.continuous.stop(); t.pose.continuous = null; }
    if (t.video) {
        t.video.pause();
        t.video.removeAttribute('src');
        t.video.load();
    }
    if (t.image) {
        t.image.removeAttribute('src');
    }
    if (t.ctx && t.canvas) t.ctx.clearRect(0, 0, t.canvas.width, t.canvas.height);

    // Reset state fields in place (preserving _state.targets[target] reference)
    t.file = null;
    t.url = null;
    t.mediaType = null;
    t.media = null;
    t.video = null;
    t.image = null;
    t.canvas = null;
    t.ctx = null;
    t.annotation = createAnnotationState();
    t.tags = [];
    t.pose = { samples: [], latestPose: null, showOverlay: true, continuous: null, samplingMs: 200 };

    const root = document.querySelector('.va-root');
    root.querySelector(`.va-drop-zone[data-target="${target}"]`).style.display = 'flex';
    const wrap = root.querySelector(`.va-video-wrap[data-target="${target}"]`);
    wrap.style.display = 'none';
    wrap.style.maxWidth = '';
    wrap.style.aspectRatio = '';
    wrap.classList.remove('is-image');
    const controls = root.querySelector(`.va-controls[data-target="${target}"]`);
    controls.style.display = 'none';
    controls.classList.remove('is-image');
    root.querySelector(`[data-action="clear"][data-target="${target}"]`).disabled = true;
    const tagEl = root.querySelector(`.va-media-tag[data-target="${target}"]`);
    if (tagEl) tagEl.textContent = '';
    const imageBadge = root.querySelector(`.va-image-badge[data-target="${target}"]`);
    if (imageBadge) imageBadge.style.display = 'none';

    if (_state.active === target && _state.mode === 'compare') {
        _setActiveTarget(target === 'A' ? 'B' : 'A');
    }
    _refreshTagList();
    _refreshTagStrip(target);
    _refreshAnnotationPaneCounts();
    _refreshSyncBarState();
    _updateMediaInfo(target);
}

// ==========================================
// Modes & Active Target
// ==========================================
function _switchMode(mode) {
    _state.mode = mode;
    const root = document.querySelector('.va-root');
    root.querySelectorAll('.va-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    root.querySelector('.va-stage').classList.toggle('va-stage-compare', mode === 'compare');
    root.querySelector('.va-stage').classList.toggle('va-stage-single', mode === 'single');
    root.querySelector('.va-sync-bar').style.display = mode === 'compare' ? 'flex' : 'none';

    const panelB = root.querySelector('.va-panel[data-target="B"]');
    panelB.style.display = mode === 'compare' ? 'flex' : 'none';
    if (mode === 'single') _setActiveTarget('A');
    _refreshSyncBarState();
}

function _setActiveTarget(target) {
    _state.active = target;
    const root = document.querySelector('.va-root');
    root.querySelectorAll('.va-panel').forEach(panel => {
        panel.classList.toggle('va-panel-active', panel.dataset.target === target);
    });
    root.querySelectorAll('.va-active-badge').forEach(badge => {
        badge.classList.toggle('on', badge.dataset.target === target);
    });
    root.querySelectorAll('.va-active-toggle').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.target === target);
    });
    _refreshAnnotationPaneCounts();
    _refreshActiveDependentUI();
}

function _syncFromA() {
    const tA = _state.targets.A;
    const tB = _state.targets.B;
    if (!tA || !tB) return;
    if (tA.mediaType !== 'video' || tB.mediaType !== 'video') return;
    const a = tA.video;
    const b = tB.video;
    if (!a || !b) return;
    const desired = a.currentTime + _state.syncOffset;
    if (!isFinite(desired)) return;
    const clamped = Math.max(0, Math.min(b.duration || desired, desired));
    if (Math.abs(b.currentTime - clamped) > 0.25) b.currentTime = clamped;
}

function _refreshSyncBarState() {
    const root = document.querySelector('.va-root');
    if (!root) return;
    const bar = root.querySelector('.va-sync-bar');
    if (!bar) return;
    const tA = _state.targets.A;
    const tB = _state.targets.B;
    const bothVideo = tA.mediaType === 'video' && tB.mediaType === 'video';
    const note = root.querySelector('#va-sync-note');
    if (note) {
        note.textContent = bothVideo
            ? 'A の現在時刻を基準に、B はオフセットだけずらして再生します。'
            : '同期再生は両側が動画のときだけ有効です。';
        note.classList.toggle('va-note-warn', !bothVideo);
    }
    bar.querySelectorAll('button, input').forEach(el => {
        if (el.id === 'va-sync-enabled' || el.id === 'va-sync-offset') return;
        el.disabled = !bothVideo;
    });
}

function _refreshPlayToggle(target) {
    const root = document.querySelector('.va-root');
    if (!root) return;
    const btn = root.querySelector(`.va-btn-play[data-target="${target}"]`);
    if (!btn) return;
    const t = _state.targets[target];
    const playing = t.mediaType === 'video' && t.video && !t.video.paused && !t.video.ended;
    btn.classList.toggle('is-playing', !!playing);
    btn.innerHTML = playing
        ? '<i class="fas fa-pause"></i>'
        : '<i class="fas fa-play"></i>';
    btn.disabled = t.mediaType !== 'video';
}

// ==========================================
// Render Loop
// ==========================================
function _startRenderLoop() {
    cancelAnimationFrame(_state.rafId);
    const tick = () => {
        ['A', 'B'].forEach(target => {
            const t = _state.targets[target];
            if (!t.media || !t.canvas || !t.ctx) return;
            const ctx = t.ctx;
            const canvas = t.canvas;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const showAll = t.mediaType === 'image';
            renderAnnotations(ctx, canvas, t.annotation, _getMediaTime(t), { showAll });
            if (t.pose.showOverlay && t.pose.latestPose) {
                drawPoseOverlay(ctx, canvas, t.pose.latestPose, t.media);
            }
        });
        _state.rafId = requestAnimationFrame(tick);
    };
    _state.rafId = requestAnimationFrame(tick);
}

function _updateTimeDisplay(target) {
    const t = _state.targets[target];
    const root = document.querySelector('.va-root');
    const span = root.querySelector(`.va-time[data-target="${target}"]`);
    if (!span) return;
    if (t.mediaType === 'video' && t.video) {
        span.textContent = `${_fmtTime(t.video.currentTime)} / ${_fmtTime(t.video.duration || 0)}`;
    } else if (t.mediaType === 'image') {
        span.textContent = '静止画';
    } else {
        span.textContent = '';
    }
}

function _fmtTime(sec) {
    if (!isFinite(sec)) return '00:00.000';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec - Math.floor(sec)) * 1000);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

function _stepFrame(target, dir) {
    const t = _state.targets[target];
    if (t.mediaType !== 'video' || !t.video) return;
    t.video.pause();
    const next = Math.max(0, Math.min((t.video.duration || 0), t.video.currentTime + dir * FRAME_STEP_SEC));
    t.video.currentTime = next;
}

// ==========================================
// Tool Tabs
// ==========================================
function _switchTool(toolName) {
    const root = document.querySelector('.va-root');
    root.querySelectorAll('#va-tool-tabs .tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tool === toolName));
    root.querySelectorAll('.tab-content[data-tool]').forEach(p => p.classList.toggle('active', p.dataset.tool === toolName));
}

function _renderToolPanes() {
    _renderDrawPane();
    _renderTagPane();
    _renderPosePane();
}

// ==========================================
// Draw Pane
// ==========================================
function _renderDrawPane() {
    const pane = document.getElementById('va-pane-draw');
    pane.innerHTML = `
        <div id="va-draw-disabled-msg" class="va-info-banner" style="display:none;"><i class="fas fa-info-circle"></i> Step 1 でメディアを読み込むと描画を開始できます。</div>
        <div id="va-draw-block">
        <div class="va-tool-section">
            <h4>描画ツール</h4>
            <div class="va-draw-toolbar">
                ${ANNOTATION_TOOLS.map(tool => `
                    <button class="va-btn va-draw-tool ${tool === 'pen' ? 'active' : ''}" data-tool="${tool}" title="${TOOL_LABELS[tool] || tool}">
                        <i class="fas ${TOOL_ICONS[tool] || 'fa-pen'}"></i>
                        <span>${TOOL_LABELS[tool] || tool}</span>
                    </button>
                `).join('')}
            </div>
            <div class="va-tool-row">
                <label class="va-inline va-color-label">色
                    <input type="color" id="va-draw-color" value="${THEME}">
                </label>
                <label class="va-inline">太さ
                    <input type="range" id="va-draw-thickness" min="1" max="20" value="4">
                    <span id="va-draw-thickness-val" class="va-num-badge">4</span>px
                </label>
                <label class="va-inline" title="動画モードでのみ有効。画像はずっと表示されます。">表示秒数
                    <input type="number" id="va-draw-duration" value="2.0" min="0.3" step="0.1" style="width:5rem;">秒
                </label>
            </div>
            <div class="va-tool-row">
                <button class="va-btn" id="va-draw-undo" title="直前の描画を取り消し"><i class="fas fa-rotate-left"></i> 1つ戻す</button>
                <button class="va-btn" id="va-draw-clear" title="全描画を削除"><i class="fas fa-eraser"></i> 全消去</button>
                <button class="va-btn" id="va-draw-export"><i class="fas fa-download"></i> JSON保存</button>
                <button class="va-btn" id="va-draw-import"><i class="fas fa-upload"></i> JSON読込</button>
                <input type="file" id="va-draw-import-input" accept="application/json" style="display:none;">
                <span class="va-hint va-pill" id="va-draw-status">対象: メディアA</span>
            </div>
        </div>
        <div class="va-tool-help">
            <p><strong>使い方</strong>: 「ツール対象」で動画/画像を選んでから上のメディアをドラッグして描画します。
            動画では現在時刻から指定秒数だけ表示され、再生中に自動でフェードアウト。画像ではずっと表示されたままになります。</p>
        </div>
        </div>
    `;
    pane.querySelectorAll('.va-draw-tool').forEach(btn => {
        btn.addEventListener('click', () => {
            pane.querySelectorAll('.va-draw-tool').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            _activeAnnotationState().tool = btn.dataset.tool;
        });
    });
    pane.querySelector('#va-draw-color').addEventListener('input', (e) => {
        _activeAnnotationState().color = e.target.value;
    });
    pane.querySelector('#va-draw-thickness').addEventListener('input', (e) => {
        const v = Number(e.target.value);
        _activeAnnotationState().thickness = v;
        pane.querySelector('#va-draw-thickness-val').textContent = String(v);
    });
    pane.querySelector('#va-draw-duration').addEventListener('input', (e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v) && v > 0) _activeAnnotationState().durationSec = v;
    });
    pane.querySelector('#va-draw-undo').addEventListener('click', () => {
        undoLastAnnotation(_activeAnnotationState());
    });
    pane.querySelector('#va-draw-clear').addEventListener('click', () => {
        if (confirm('描画を全て削除しますか？')) clearAnnotations(_activeAnnotationState());
    });
    pane.querySelector('#va-draw-export').addEventListener('click', () => {
        const json = exportAnnotationsJSON(_activeAnnotationState());
        _downloadFile(`annotations_${_state.active}.json`, json, 'application/json');
    });
    const importInput = pane.querySelector('#va-draw-import-input');
    pane.querySelector('#va-draw-import').addEventListener('click', () => importInput.click());
    importInput.addEventListener('change', (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        const reader = new FileReader();
        reader.onload = () => {
            try {
                importAnnotationsJSON(_activeAnnotationState(), String(reader.result));
                alert(`${_activeAnnotationState().items.length} 件のアノテーションを読み込みました。`);
            } catch (err) {
                alert('JSON読み込みに失敗しました: ' + err.message);
            }
        };
        reader.readAsText(f);
    });
}

function _activeAnnotationState() {
    return _state.targets[_state.active].annotation;
}

function _refreshAnnotationPaneCounts() {
    const status = document.getElementById('va-draw-status');
    if (status) {
        const t = _state.targets[_state.active];
        const typeLabel = t.mediaType === 'image' ? '画像' : (t.mediaType === 'video' ? '動画' : '未読込');
        status.textContent = `対象: メディア${_state.active} (${typeLabel})`;
    }
    _refreshSummary();
}

function _updateMediaInfo(target) {
    const root = document.querySelector('.va-root');
    if (!root) return;
    const el = root.querySelector(`.va-media-info[data-target="${target}"]`);
    if (!el) return;
    const t = _state.targets[target];
    if (!t.file || !t.media) {
        el.textContent = '';
        return;
    }
    const name = t.file.name;
    let dims = '';
    if (t.mediaType === 'video' && t.video && t.video.videoWidth) {
        dims = `${t.video.videoWidth}×${t.video.videoHeight}`;
        if (isFinite(t.video.duration) && t.video.duration > 0) {
            dims += ` ・ ${_fmtTime(t.video.duration).split('.')[0]}`;
        }
    } else if (t.mediaType === 'image' && t.image && t.image.naturalWidth) {
        dims = `${t.image.naturalWidth}×${t.image.naturalHeight}`;
    }
    const sizeKB = t.file.size ? `${(t.file.size / 1024).toFixed(0)} KB` : '';
    el.innerHTML = `<i class="fas fa-file"></i> ${_escape(name)}${dims ? ' ・ ' + dims : ''}${sizeKB ? ' ・ ' + sizeKB : ''}`;
}

function _refreshSummary() {
    const grid = document.getElementById('va-summary-grid');
    if (!grid) return;
    const tA = _state.targets.A;
    const tB = _state.targets.B;
    const annoCount = tA.annotation.items.length + tB.annotation.items.length;
    const tagCount = tA.tags.length + tB.tags.length;
    const poseCount = tA.pose.samples.length + tB.pose.samples.length;
    const mediaA = tA.mediaType ? (tA.mediaType === 'image' ? '画像' : '動画') : '未読込';
    const mediaB = tB.mediaType ? (tB.mediaType === 'image' ? '画像' : '動画') : '未読込';
    const mediaAClass = tA.mediaType ? (tA.mediaType === 'image' ? 'good' : 'good') : '';
    const mediaBClass = tB.mediaType ? (tB.mediaType === 'image' ? 'good' : 'good') : '';
    grid.innerHTML = `
        <div class="metric-card ${mediaAClass}">
            <div class="metric-label">メディアA</div>
            <div class="metric-value" style="font-size: 1.25rem;">${mediaA}</div>
        </div>
        <div class="metric-card ${mediaBClass}">
            <div class="metric-label">メディアB</div>
            <div class="metric-value" style="font-size: 1.25rem;">${mediaB}</div>
        </div>
        <div class="metric-card ${annoCount > 0 ? 'good' : ''}">
            <div class="metric-label">描画件数</div>
            <div class="metric-value">${annoCount}</div>
        </div>
        <div class="metric-card ${tagCount > 0 ? 'good' : ''}">
            <div class="metric-label">タグ件数</div>
            <div class="metric-value">${tagCount}</div>
        </div>
        <div class="metric-card ${poseCount > 0 ? 'good' : ''}">
            <div class="metric-label">ポーズサンプル</div>
            <div class="metric-value">${poseCount}</div>
        </div>
    `;

    // エクスポートボタンの状態を内容と active target に応じて切替
    const tActive = _state.targets[_state.active];
    const activeAnno = tActive.annotation.items.length;
    const activeTags = tActive.tags.length;
    const activePoses = tActive.pose.samples.length;

    const tagsBtn = document.getElementById('va-export-tags-csv');
    const annoBtn = document.getElementById('va-export-anno-json');
    const poseBtn = document.getElementById('va-export-pose-csv');

    function updateBtn(btn, count, label, icon) {
        if (!btn) return;
        btn.disabled = count === 0;
        btn.innerHTML = `<i class="fas ${icon}"></i> ${label} <span class="va-badge-count">${count}</span>`;
    }
    updateBtn(tagsBtn, tagCount, 'タグを CSV に保存', 'fa-file-csv');
    updateBtn(annoBtn, activeAnno, `描画 (メディア${_state.active}) を JSON に保存`, 'fa-file-code');
    updateBtn(poseBtn, activePoses, `関節角度 (メディア${_state.active}) を CSV に保存`, 'fa-file-csv');
}

// ==========================================
// Tag Pane
// ==========================================
function _renderTagPane() {
    const pane = document.getElementById('va-pane-tag');
    pane.innerHTML = `
        <div id="va-tag-disabled-msg" class="va-info-banner" style="display:none;"><i class="fas fa-info-circle"></i> Step 1 で動画を読み込むとタグ／イベント記録が利用できます。</div>
        <div id="va-tag-block">
            <div class="va-tool-section">
                <h4>タグ種別</h4>
                <div id="va-tag-defs"></div>
                <div class="va-tool-row">
                    <input type="text" id="va-tag-new-label" placeholder="新しいタグ名" style="flex:1;min-width:8rem;">
                    <input type="color" id="va-tag-new-color" value="#3b82f6">
                    <button class="va-btn va-btn-primary" id="va-tag-add"><i class="fas fa-plus"></i> 追加</button>
                </div>
                <p class="va-hint">先頭9件のタグはホットキー <kbd>1</kbd>〜<kbd>9</kbd> で打鍵できます（テキスト入力中は無効）。</p>
            </div>
            <div class="va-tool-section">
                <h4>記録</h4>
                <div class="va-tool-row" id="va-tag-quickbar"></div>
                <div class="va-tool-row">
                    <button class="va-btn" id="va-tag-export"><i class="fas fa-file-csv"></i> CSV出力</button>
                    <button class="va-btn" id="va-tag-clear-all"><i class="fas fa-trash"></i> 全削除</button>
                    <span class="va-hint va-pill">対象: メディア${_state.active}</span>
                </div>
            </div>
            <div class="va-tool-section">
                <h4>タグ一覧</h4>
                <div class="va-table-wrap">
                    <table class="va-tag-table">
                        <thead><tr><th>時刻</th><th>動画</th><th>ラベル</th><th>コメント</th><th></th></tr></thead>
                        <tbody id="va-tag-tbody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    pane.querySelector('#va-tag-add').addEventListener('click', () => {
        const labelInput = pane.querySelector('#va-tag-new-label');
        const colorInput = pane.querySelector('#va-tag-new-color');
        const label = labelInput.value.trim();
        if (!label) return;
        _state.tagDefs.push({
            id: _newId('tagdef'),
            label,
            color: colorInput.value,
            hotkey: _state.tagDefs.length < 9 ? String(_state.tagDefs.length + 1) : ''
        });
        labelInput.value = '';
        _refreshTagDefs();
    });

    pane.querySelector('#va-tag-export').addEventListener('click', () => {
        _exportTagsCSV();
    });
    pane.querySelector('#va-tag-clear-all').addEventListener('click', () => {
        if (confirm(`動画${_state.active} のタグをすべて削除しますか？`)) {
            _state.targets[_state.active].tags = [];
            _refreshTagList();
            _refreshTagStrip(_state.active);
        }
    });

    _refreshTagDefs();
    _refreshTagList();
}

function _refreshTagDefs() {
    const container = document.getElementById('va-tag-defs');
    if (!container) return;
    container.innerHTML = _state.tagDefs.map(def => `
        <div class="va-tag-def" data-id="${def.id}" style="--c:${def.color}">
            <span class="va-tag-swatch" style="background:${def.color};"></span>
            <input type="text" class="va-tag-label-input" value="${_escape(def.label)}">
            <input type="color" class="va-tag-color-input" value="${def.color}">
            <input type="text" class="va-tag-hotkey-input" value="${def.hotkey || ''}" maxlength="1" placeholder="key" style="width:3rem;">
            <button class="va-btn va-tag-def-delete"><i class="fas fa-times"></i></button>
        </div>
    `).join('');

    container.querySelectorAll('.va-tag-def').forEach(row => {
        const id = row.dataset.id;
        const def = _state.tagDefs.find(d => d.id === id);
        row.querySelector('.va-tag-label-input').addEventListener('input', (e) => { def.label = e.target.value; _refreshQuickbar(); _refreshTagList(); });
        row.querySelector('.va-tag-color-input').addEventListener('input', (e) => {
            def.color = e.target.value;
            row.style.setProperty('--c', e.target.value);
            row.querySelector('.va-tag-swatch').style.background = e.target.value;
            _refreshQuickbar();
            _refreshTagList();
            ['A', 'B'].forEach(t => _refreshTagStrip(t));
        });
        row.querySelector('.va-tag-hotkey-input').addEventListener('input', (e) => { def.hotkey = e.target.value.trim().slice(0, 1); _refreshQuickbar(); });
        row.querySelector('.va-tag-def-delete').addEventListener('click', () => {
            if (confirm(`「${def.label}」を削除しますか？打鍵済みタグは残ります。`)) {
                _state.tagDefs = _state.tagDefs.filter(d => d.id !== id);
                _refreshTagDefs();
            }
        });
    });
    _refreshQuickbar();
}

function _refreshQuickbar() {
    const bar = document.getElementById('va-tag-quickbar');
    if (!bar) return;
    bar.innerHTML = _state.tagDefs.map(def => `
        <button class="va-btn va-tag-quick" style="background:${def.color};color:#fff;border-color:${def.color};" data-id="${def.id}">
            ${def.hotkey ? `<kbd style="background:rgba(255,255,255,0.25);">${_escape(def.hotkey)}</kbd>` : ''}
            ${_escape(def.label)}
        </button>
    `).join('');
    bar.querySelectorAll('.va-tag-quick').forEach(btn => {
        btn.addEventListener('click', () => {
            const def = _state.tagDefs.find(d => d.id === btn.dataset.id);
            if (def) _recordTag(def);
        });
    });
}

function _recordTag(def) {
    const target = _state.active;
    const t = _state.targets[target];
    if (!t.media) {
        alert('メディアが読み込まれていません。');
        return;
    }
    if (t.mediaType !== 'video') {
        alert('タグは動画のみ記録できます。画像にはタグ付けできません。');
        return;
    }
    t.tags.push({
        id: _newId('tag'),
        time: t.video.currentTime,
        defId: def.id,
        label: def.label,
        color: def.color,
        comment: ''
    });
    t.tags.sort((a, b) => a.time - b.time);
    _refreshTagList();
    _refreshTagStrip(target);
}

function _refreshTagList() {
    const tbody = document.getElementById('va-tag-tbody');
    if (!tbody) return;
    const all = [];
    ['A', 'B'].forEach(target => {
        _state.targets[target].tags.forEach(tag => all.push({ ...tag, target }));
    });
    all.sort((a, b) => a.target === b.target ? a.time - b.time : a.target.localeCompare(b.target));
    tbody.innerHTML = all.map(tag => `
        <tr data-target="${tag.target}" data-id="${tag.id}">
            <td><button class="va-link" data-action="jump">${_fmtTime(tag.time)}</button></td>
            <td>${tag.target}</td>
            <td><span class="va-tag-swatch" style="background:${tag.color};"></span> ${_escape(tag.label)}</td>
            <td><input type="text" class="va-tag-comment" value="${_escape(tag.comment || '')}"></td>
            <td><button class="va-btn va-tag-del"><i class="fas fa-times"></i></button></td>
        </tr>
    `).join('');

    tbody.querySelectorAll('tr').forEach(tr => {
        const target = tr.dataset.target;
        const id = tr.dataset.id;
        const tag = _state.targets[target].tags.find(t => t.id === id);
        if (!tag) return;
        tr.querySelector('[data-action="jump"]').addEventListener('click', () => {
            const v = _state.targets[target].video;
            if (v) { v.currentTime = tag.time; _setActiveTarget(target); }
        });
        tr.querySelector('.va-tag-comment').addEventListener('input', (e) => { tag.comment = e.target.value; });
        tr.querySelector('.va-tag-del').addEventListener('click', () => {
            _state.targets[target].tags = _state.targets[target].tags.filter(t => t.id !== id);
            _refreshTagList();
            _refreshTagStrip(target);
        });
    });
    _refreshSummary();
}

function _refreshTagStrip(target) {
    const root = document.querySelector('.va-root');
    if (!root) return;
    const strip = root.querySelector(`.va-tag-strip[data-target="${target}"]`);
    if (!strip) return;
    const t = _state.targets[target];
    if (t.mediaType !== 'video' || !t.video || !isFinite(t.video.duration) || t.video.duration <= 0) {
        strip.innerHTML = '';
        return;
    }
    const dur = t.video.duration;
    strip.innerHTML = t.tags.map(tag => {
        const left = (tag.time / dur) * 100;
        return `<button class="va-tag-mark" data-id="${tag.id}" style="left:${left}%;background:${tag.color};" title="${_escape(tag.label)} @ ${_fmtTime(tag.time)}"></button>`;
    }).join('');
    strip.querySelectorAll('.va-tag-mark').forEach(btn => {
        btn.addEventListener('click', () => {
            const tag = t.tags.find(x => x.id === btn.dataset.id);
            if (tag && t.video) t.video.currentTime = tag.time;
        });
    });
}

function _exportTagsCSV() {
    const rows = [['target', 'time_sec', 'time', 'label', 'color', 'comment']];
    ['A', 'B'].forEach(target => {
        _state.targets[target].tags.forEach(tag => {
            rows.push([
                target,
                tag.time.toFixed(3),
                _fmtTime(tag.time),
                tag.label,
                tag.color,
                (tag.comment || '').replace(/"/g, '""')
            ]);
        });
    });
    const csv = rows.map(r => r.map(c => /[",\n]/.test(String(c)) ? `"${c}"` : String(c)).join(',')).join('\n');
    _downloadFile('tags.csv', csv, 'text/csv');
}

// ==========================================
// Pose Pane
// ==========================================
function _renderPosePane() {
    const pane = document.getElementById('va-pane-pose');
    pane.innerHTML = `
        <div class="va-tool-section">
            <h4>ポーズ推定 (MoveNet Lightning)</h4>
            <p id="va-pose-mode-hint" class="va-hint" style="margin: 0 0 0.5rem 0;"><i class="fas fa-info-circle"></i> Step 1 でメディアを読み込んでから推定してください。</p>
            <div class="va-tool-row">
                <button class="va-btn va-btn-primary" id="va-pose-once" disabled><i class="fas fa-camera"></i> 現在フレームを推定</button>
                <button class="va-btn" id="va-pose-start" disabled><i class="fas fa-play-circle"></i> 連続推定 開始</button>
                <button class="va-btn" id="va-pose-stop" disabled><i class="fas fa-stop-circle"></i> 連続推定 停止</button>
                <label class="va-inline">サンプリング
                    <input type="number" id="va-pose-sampling" min="50" max="2000" step="50" value="200" style="width:5rem;"> ms
                </label>
                <label class="va-inline"><input type="checkbox" id="va-pose-overlay" checked> 骨格オーバーレイ</label>
            </div>
            <div class="va-tool-row">
                <span id="va-pose-status" class="va-hint">未推定</span>
                <button class="va-btn" id="va-pose-clear"><i class="fas fa-eraser"></i> サンプル削除</button>
                <button class="va-btn" id="va-pose-export"><i class="fas fa-file-csv"></i> 関節角度CSV</button>
            </div>
        </div>
        <div class="va-tool-section">
            <h4>関節角度の現在値</h4>
            <div id="va-pose-angles" class="va-pose-angles"></div>
        </div>
        <div class="va-tool-section" id="va-pose-chart-section">
            <h4>関節角度の時系列</h4>
            <div id="va-pose-chart" style="width:100%;height:320px;"></div>
            <p class="va-hint">連続推定で蓄積されたサンプル（時刻×関節角度）をプロットします。</p>
        </div>
        <div class="va-tool-help">
            <p><strong>MoveNetについて</strong>: 人物を1人検出し、17キーポイント（頭・肩・肘・手首・腰・膝・足首など）の2D座標を返します。
            背景や衣服、視点、解像度の影響を受けるため、ぶれる場合は信頼度が低くなり描画されません。教育用の目安としてご利用ください。</p>
        </div>
    `;
    _renderAngleSlots();

    pane.querySelector('#va-pose-once').addEventListener('click', _runSinglePose);
    pane.querySelector('#va-pose-start').addEventListener('click', _startContinuousPose);
    pane.querySelector('#va-pose-stop').addEventListener('click', _stopContinuousPose);
    pane.querySelector('#va-pose-overlay').addEventListener('change', (e) => {
        _state.targets[_state.active].pose.showOverlay = e.target.checked;
    });
    pane.querySelector('#va-pose-sampling').addEventListener('input', (e) => {
        const v = Math.max(50, Number(e.target.value) || 200);
        _state.targets[_state.active].pose.samplingMs = v;
    });
    pane.querySelector('#va-pose-clear').addEventListener('click', () => {
        _state.targets[_state.active].pose.samples = [];
        _refreshPoseChart();
        _setPoseStatus('サンプル削除');
    });
    pane.querySelector('#va-pose-export').addEventListener('click', () => {
        const samples = _state.targets[_state.active].pose.samples;
        if (samples.length === 0) {
            alert('まだサンプルがありません。連続推定を開始してください。');
            return;
        }
        _downloadFile(`pose_angles_${_state.active}.csv`, exportPoseSeriesCSV(samples), 'text/csv');
    });
}

function _renderAngleSlots() {
    const wrap = document.getElementById('va-pose-angles');
    if (!wrap) return;
    wrap.innerHTML = POSE_ANGLE_DEFS.map(def => `
        <div class="va-angle" data-key="${def.key}">
            <div class="va-angle-label">${def.label}</div>
            <div class="va-angle-value">— °</div>
        </div>
    `).join('');
}

async function _runSinglePose() {
    const target = _state.active;
    const t = _state.targets[target];
    if (!t.media) { alert('メディアを読み込んでください'); return; }
    _setPoseStatus('モデル準備中...');
    try {
        await ensurePoseDetector();
        _setPoseStatus('推定中...');
        const pose = await detectPoseOnce(t.media);
        if (!pose) { _setPoseStatus('人物を検出できませんでした'); return; }
        t.pose.latestPose = pose;
        const angles = computeJointAngles(pose);
        _updateAngleDisplay(angles);
        t.pose.samples.push({ time: _getMediaTime(t), pose, angles });
        _refreshPoseChart();
        _refreshSummary();
        const high = pose.keypoints.filter(k => (k.score || 0) >= 0.45).length;
        const med = pose.keypoints.filter(k => (k.score || 0) >= 0.20 && (k.score || 0) < 0.45).length;
        _setPoseStatus(`成功（高信頼: ${high} / 中信頼: ${med} / 17）`);
    } catch (err) {
        console.error(err);
        _setPoseStatus('エラー: ' + err.message);
    }
}

async function _startContinuousPose() {
    const target = _state.active;
    const t = _state.targets[target];
    if (!t.media) { alert('メディアを読み込んでください'); return; }
    if (t.mediaType !== 'video') {
        alert('連続推定は動画のみ対応です。画像は「現在フレームを推定」をご利用ください。');
        return;
    }
    if (t.pose.continuous) return;

    _setPoseStatus('モデル準備中...');
    try {
        await ensurePoseDetector();
    } catch (err) {
        _setPoseStatus('エラー: ' + err.message);
        return;
    }
    document.getElementById('va-pose-start').disabled = true;
    document.getElementById('va-pose-stop').disabled = false;
    _setPoseStatus('連続推定中...');

    t.pose.continuous = startContinuousDetection({
        video: t.video,
        samplingMs: t.pose.samplingMs,
        onSample: (sample) => {
            t.pose.latestPose = sample.pose;
            t.pose.samples.push(sample);
            _updateAngleDisplay(sample.angles);
            if (t.pose.samples.length % 5 === 0) {
                _refreshPoseChart();
                _refreshSummary();
            }
            _setPoseStatus(`連続推定中（${t.pose.samples.length} サンプル）`);
        }
    });
    if (t.video.paused) t.video.play().catch(() => {});
}

function _stopContinuousPose() {
    const target = _state.active;
    const t = _state.targets[target];
    if (t.pose.continuous) { t.pose.continuous.stop(); t.pose.continuous = null; }
    document.getElementById('va-pose-start').disabled = false;
    document.getElementById('va-pose-stop').disabled = true;
    _refreshPoseChart();
    _setPoseStatus(`停止（合計 ${t.pose.samples.length} サンプル）`);
}

function _updateAngleDisplay(angles) {
    POSE_ANGLE_DEFS.forEach(def => {
        const el = document.querySelector(`.va-angle[data-key="${def.key}"] .va-angle-value`);
        if (!el) return;
        const v = angles ? angles[def.key] : null;
        el.textContent = v == null ? '— °' : `${v.toFixed(1)} °`;
    });
}

function _setPoseStatus(text) {
    const el = document.getElementById('va-pose-status');
    if (el) el.textContent = text;
}

function _refreshPoseChart() {
    const target = _state.active;
    const t = _state.targets[target];
    const samples = t.pose.samples;
    const div = document.getElementById('va-pose-chart');
    const section = document.getElementById('va-pose-chart-section');
    if (!div) return;

    // 画像モードでは時系列チャート自体を隠す
    if (section) section.style.display = t.mediaType === 'image' ? 'none' : '';
    if (t.mediaType === 'image') {
        if (typeof Plotly !== 'undefined') Plotly.purge(div);
        return;
    }

    if (typeof Plotly === 'undefined') return;
    if (samples.length === 0) {
        Plotly.purge(div);
        div.innerHTML = '<div style="padding:1rem;color:#64748b;">まだサンプルがありません。「現在フレームを推定」または「連続推定」を実行してください。</div>';
        return;
    }
    const x = samples.map(s => s.time);
    const traces = POSE_ANGLE_DEFS.map(def => ({
        x,
        y: samples.map(s => s.angles ? s.angles[def.key] : null),
        type: 'scatter',
        mode: 'lines',
        name: def.label,
        connectgaps: false
    }));
    Plotly.react(div, traces, {
        margin: { t: 10, r: 10, b: 40, l: 40 },
        xaxis: { title: '時刻 (秒)' },
        yaxis: { title: '角度 (°)', range: [0, 200] },
        legend: { orientation: 'h' }
    }, { displayModeBar: false, responsive: true });
}

function _refreshActiveDependentUI() {
    const t = _state.targets[_state.active];
    const isImage = t.mediaType === 'image';
    const isVideo = t.mediaType === 'video';

    // ポーズ：連続推定とサンプリング設定は動画のみ
    const startBtn = document.getElementById('va-pose-start');
    const stopBtn = document.getElementById('va-pose-stop');
    const samplingInput = document.getElementById('va-pose-sampling');
    if (startBtn) startBtn.disabled = !isVideo;
    if (stopBtn) stopBtn.disabled = !t.pose.continuous;
    if (samplingInput) samplingInput.disabled = !isVideo;
    const onceBtn = document.getElementById('va-pose-once');
    if (onceBtn) onceBtn.disabled = !t.media;

    // ポーズタブの注意書き
    const poseHint = document.getElementById('va-pose-mode-hint');
    if (poseHint) {
        if (isImage) poseHint.innerHTML = '<i class="fas fa-info-circle"></i> 画像モードでは「現在フレームを推定」のみ利用できます。連続推定は動画のみ対応です。';
        else if (isVideo) poseHint.innerHTML = '<i class="fas fa-info-circle"></i> 連続推定で再生中の骨格・関節角度を時系列に記録できます。';
        else poseHint.innerHTML = '<i class="fas fa-info-circle"></i> Step 1 でメディアを読み込んでから推定してください。';
    }

    // タグタブ
    const tagBlock = document.getElementById('va-tag-block');
    const tagDisabled = document.getElementById('va-tag-disabled-msg');
    if (tagBlock) tagBlock.style.display = isVideo ? '' : 'none';
    if (tagDisabled) {
        tagDisabled.style.display = isVideo ? 'none' : '';
        if (!t.media) tagDisabled.innerHTML = '<i class="fas fa-info-circle"></i> Step 1 で動画を読み込むとタグ／イベント記録が利用できます。';
        else tagDisabled.innerHTML = '<i class="fas fa-info-circle"></i> 画像にはタグ付けできません。動画ファイルを読み込んでください。';
    }

    // 描画タブ：メディア未読込時はガイドのみ表示
    const drawBlock = document.getElementById('va-draw-block');
    const drawDisabled = document.getElementById('va-draw-disabled-msg');
    const hasMedia = !!t.media;
    if (drawBlock) drawBlock.style.display = hasMedia ? '' : 'none';
    if (drawDisabled) drawDisabled.style.display = hasMedia ? 'none' : '';

    // パネルヘッダのアクションボタン：メディア読込前は冗長なので隠す（ドロップゾーンのCTAを主導線にする）
    const root = document.querySelector('.va-root');
    if (root) {
        ['A', 'B'].forEach(tg => {
            const tt = _state.targets[tg];
            const actions = root.querySelector(`.va-panel[data-target="${tg}"] .va-panel-actions`);
            if (actions) actions.style.display = tt.media ? 'flex' : 'none';
        });
    }

    _refreshPoseChart();
    _refreshSummary();
}

// ==========================================
// Keyboard Shortcuts
// ==========================================
function _bindKeyboardShortcuts() {
    if (_keysBound) return;
    _keysBound = true;
    document.addEventListener('keydown', (e) => {
        // 現在の分析が動画分析である場合のみ処理
        if (!document.querySelector('.va-root')) return;
        const tag = (e.target && e.target.tagName) || '';
        const isInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(tag);
        if (isInput) return;

        if (e.code === 'Space') {
            e.preventDefault();
            const t = _state.targets[_state.active];
            if (t.mediaType === 'video' && t.video) {
                if (t.video.paused) t.video.play().catch(() => {});
                else t.video.pause();
            }
            return;
        }
        if (e.key === ',') { _stepFrame(_state.active, -1); e.preventDefault(); return; }
        if (e.key === '.') { _stepFrame(_state.active, +1); e.preventDefault(); return; }
        if (e.key === 'a' || e.key === 'A') { _setActiveTarget('A'); return; }
        if (e.key === 'b' || e.key === 'B') {
            if (_state.mode === 'compare') _setActiveTarget('B');
            return;
        }
        if (/^[1-9]$/.test(e.key)) {
            const def = _state.tagDefs.find(d => d.hotkey === e.key);
            if (def) { _recordTag(def); e.preventDefault(); }
        }
    });
}

// ==========================================
// Utilities
// ==========================================
function _downloadFile(name, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

function _escape(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
