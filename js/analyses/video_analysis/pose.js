// ==========================================
// 動画分析: ポーズ推定モジュール (MoveNet)
// TensorFlow.js + @tensorflow-models/pose-detection
// ==========================================

const KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
];

const SKELETON_PAIRS = [
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
    ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip'],
    ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
];

const ANGLE_DEFINITIONS = [
    { key: 'left_elbow', label: '左肘', a: 'left_shoulder', b: 'left_elbow', c: 'left_wrist' },
    { key: 'right_elbow', label: '右肘', a: 'right_shoulder', b: 'right_elbow', c: 'right_wrist' },
    { key: 'left_knee', label: '左膝', a: 'left_hip', b: 'left_knee', c: 'left_ankle' },
    { key: 'right_knee', label: '右膝', a: 'right_hip', b: 'right_knee', c: 'right_ankle' },
    { key: 'left_shoulder', label: '左肩', a: 'left_hip', b: 'left_shoulder', c: 'left_elbow' },
    { key: 'right_shoulder', label: '右肩', a: 'right_hip', b: 'right_shoulder', c: 'right_elbow' }
];

const KEYPOINT_DRAW_THRESHOLD = 0.20;  // 描画する最小信頼度
const SKELETON_DRAW_THRESHOLD = 0.25;  // 骨格線を結ぶ最小信頼度
const ANGLE_THRESHOLD = 0.20;          // 角度計算に使う最小信頼度
const KEYPOINT_HIGH_CONFIDENCE = 0.45; // この値以上は鮮やか色

let _detectorPromise = null;

function _getPoseLib() {
    if (typeof poseDetection === 'undefined') {
        throw new Error('pose-detection ライブラリが読み込まれていません。ページをリロードしてください。');
    }
    return poseDetection;
}

function _getTf() {
    if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js が読み込まれていません。');
    }
    return tf;
}

/**
 * MoveNet (lightning) のシングルトンを取得
 */
export async function ensurePoseDetector() {
    if (_detectorPromise) return _detectorPromise;
    const tfLib = _getTf();
    const poseLib = _getPoseLib();
    _detectorPromise = (async () => {
        try {
            await tfLib.setBackend('webgl');
        } catch (_) {
            await tfLib.setBackend('cpu');
        }
        await tfLib.ready();
        const model = poseLib.SupportedModels.MoveNet;
        return poseLib.createDetector(model, {
            modelType: poseLib.movenet.modelType.SINGLEPOSE_LIGHTNING,
            enableSmoothing: true
        });
    })();
    return _detectorPromise;
}

function _isMediaReady(media) {
    if (!media) return false;
    if (media.tagName === 'VIDEO') return media.readyState >= 2;
    if (media.tagName === 'IMG') return media.complete && media.naturalWidth > 0;
    return true;
}

function _mediaSize(media) {
    if (!media) return { w: 0, h: 0 };
    if (media.tagName === 'VIDEO') return { w: media.videoWidth || 0, h: media.videoHeight || 0 };
    if (media.tagName === 'IMG') return { w: media.naturalWidth || 0, h: media.naturalHeight || 0 };
    return { w: media.width || 0, h: media.height || 0 };
}

/**
 * 1フレーム分の推定。<video> または <img> を受け取り、natural サイズのオフスクリーン canvas に
 * 転写してから MoveNet へ渡す。これにより CSS の表示サイズに依存せず安定して推論できる。
 */
export async function detectPoseOnce(media) {
    if (!_isMediaReady(media)) {
        throw new Error('メディアの準備ができていません。動画は1度再生、画像は読み込み完了をお待ちください。');
    }
    const { w, h } = _mediaSize(media);
    if (!w || !h) {
        throw new Error('メディアサイズを取得できません。読み込みが完了しているか確認してください。');
    }
    const offCanvas = document.createElement('canvas');
    offCanvas.width = w;
    offCanvas.height = h;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(media, 0, 0, w, h);

    const detector = await ensurePoseDetector();
    const poses = await detector.estimatePoses(offCanvas, { maxPoses: 1, flipHorizontal: false });
    if (!poses || poses.length === 0) return null;
    return poses[0];
}

/**
 * 骨格をオーバーレイ描画する
 * 信頼度に応じて透明度・色を変えて視認性を高める
 */
export function drawPoseOverlay(ctx, canvas, pose, media) {
    if (!pose || !pose.keypoints) return;

    const { w: vw, h: vh } = _mediaSize(media);
    const scaleX = canvas.width / (vw || canvas.width);
    const scaleY = canvas.height / (vh || canvas.height);
    // 表示サイズに対して一貫した見え方になるよう、CSS表示サイズから canvas 単位への倍率を求める
    const displayRect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : { width: canvas.width, height: canvas.height };
    const displayScale = displayRect.width > 0 ? canvas.width / displayRect.width : 1;
    // 表示時に常に視認できるよう、表示px換算で線太さ4px・ドット半径6px・フォント16pxを目安にする
    const lineWidth = Math.max(3, 4 * displayScale);
    const dotRadius = Math.max(5, 6 * displayScale);
    const fontSize = Math.max(14, 16 * displayScale);

    const map = {};
    pose.keypoints.forEach(kp => { map[kp.name] = kp; });

    // 骨格線（信頼度の最小値で太さ・透明度を決める）
    SKELETON_PAIRS.forEach(([nameA, nameB]) => {
        const a = map[nameA];
        const b = map[nameB];
        if (!a || !b) return;
        const aScore = a.score || 0;
        const bScore = b.score || 0;
        if (aScore < SKELETON_DRAW_THRESHOLD || bScore < SKELETON_DRAW_THRESHOLD) return;
        const minScore = Math.min(aScore, bScore);
        const alpha = Math.min(1, 0.4 + minScore * 1.2);
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        // 縁取りで黒のシャドウを足し読みやすく
        ctx.shadowColor = 'rgba(0,0,0,0.55)';
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.moveTo(a.x * scaleX, a.y * scaleY);
        ctx.lineTo(b.x * scaleX, b.y * scaleY);
        ctx.stroke();
    });
    ctx.shadowBlur = 0;

    // キーポイント（高信頼=黄、中信頼=オレンジ、低信頼=半透明グレー）
    pose.keypoints.forEach(kp => {
        const score = kp.score || 0;
        if (score < KEYPOINT_DRAW_THRESHOLD) return;
        let fill;
        let alpha;
        if (score >= KEYPOINT_HIGH_CONFIDENCE) { fill = '#facc15'; alpha = 1.0; }
        else if (score >= SKELETON_DRAW_THRESHOLD) { fill = '#fb923c'; alpha = 0.9; }
        else { fill = '#cbd5e1'; alpha = 0.7; }
        ctx.globalAlpha = alpha;
        ctx.fillStyle = fill;
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(kp.x * scaleX, kp.y * scaleY, dotRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
    ctx.globalAlpha = 1.0;

    // 関節角度の弧と数値ラベルを描画（どの角度なのか視覚的に明示）
    const angles = computeJointAngles(pose);
    const arcRadius = Math.max(18, dotRadius * 3.2);
    ctx.font = `bold ${Math.round(fontSize)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ANGLE_DEFINITIONS.forEach(def => {
        const angle = angles[def.key];
        if (angle == null) return;
        const aKp = map[def.a];
        const bKp = map[def.b];
        const cKp = map[def.c];
        if (!aKp || !bKp || !cKp) return;

        const bx = bKp.x * scaleX;
        const by = bKp.y * scaleY;
        const v1x = aKp.x * scaleX - bx;
        const v1y = aKp.y * scaleY - by;
        const v2x = cKp.x * scaleX - bx;
        const v2y = cKp.y * scaleY - by;
        const v1Len = Math.hypot(v1x, v1y);
        const v2Len = Math.hypot(v2x, v2y);
        if (v1Len === 0 || v2Len === 0) return;

        const v1Angle = Math.atan2(v1y, v1x);
        const v2Angle = Math.atan2(v2y, v2x);
        let diff = v2Angle - v1Angle;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        while (diff <= -Math.PI) diff += 2 * Math.PI;
        const anticlockwise = diff < 0;

        // 弧の塗り（扇形）— 関節内側を半透明で塗って測定範囲を強調
        ctx.globalAlpha = 0.25;
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.arc(bx, by, arcRadius, v1Angle, v2Angle, anticlockwise);
        ctx.closePath();
        ctx.fill();

        // 弧の外周線
        ctx.globalAlpha = 0.95;
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = Math.max(2, lineWidth * 0.7);
        ctx.shadowColor = 'rgba(0,0,0,0.6)';
        ctx.shadowBlur = 3;
        ctx.beginPath();
        ctx.arc(bx, by, arcRadius, v1Angle, v2Angle, anticlockwise);
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1.0;

        // テキストは弧の二等分線上、円の少し外側に配置
        const u1x = v1x / v1Len, u1y = v1y / v1Len;
        const u2x = v2x / v2Len, u2y = v2y / v2Len;
        let bisX = u1x + u2x;
        let bisY = u1y + u2y;
        const bisLen = Math.hypot(bisX, bisY);
        if (bisLen < 0.001) {
            // 180°近辺は二等分線が定義しづらい → 弧の中点へ
            const mid = v1Angle + diff / 2;
            bisX = Math.cos(mid);
            bisY = Math.sin(mid);
        } else {
            bisX /= bisLen;
            bisY /= bisLen;
        }
        const labelDist = arcRadius + fontSize * 0.9;
        const tx = bx + bisX * labelDist;
        const ty = by + bisY * labelDist;

        const text = `${Math.round(angle)}°`;
        const textW = ctx.measureText(text).width;
        const padX = fontSize * 0.45;
        const padY = fontSize * 0.22;
        ctx.fillStyle = 'rgba(0,0,0,0.82)';
        _roundRect(ctx, tx - textW / 2 - padX, ty - fontSize / 2 - padY, textW + padX * 2, fontSize + padY * 2, fontSize * 0.32);
        ctx.fill();
        ctx.fillStyle = '#fef3c7';
        ctx.fillText(text, tx, ty);
    });
}

function _roundRect(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

/**
 * 3点から角度（度）を返す。Bが頂点。
 */
function _angleAt(a, b, c) {
    const v1 = { x: a.x - b.x, y: a.y - b.y };
    const v2 = { x: c.x - b.x, y: c.y - b.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
    if (mag === 0) return null;
    const cos = Math.max(-1, Math.min(1, dot / mag));
    return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * pose から関節角度を抽出
 */
export function computeJointAngles(pose) {
    if (!pose || !pose.keypoints) return {};
    const map = {};
    pose.keypoints.forEach(kp => { map[kp.name] = kp; });

    const result = {};
    ANGLE_DEFINITIONS.forEach(def => {
        const a = map[def.a];
        const b = map[def.b];
        const c = map[def.c];
        if (!a || !b || !c) { result[def.key] = null; return; }
        if ((a.score || 0) < ANGLE_THRESHOLD) { result[def.key] = null; return; }
        if ((b.score || 0) < ANGLE_THRESHOLD) { result[def.key] = null; return; }
        if ((c.score || 0) < ANGLE_THRESHOLD) { result[def.key] = null; return; }
        result[def.key] = _angleAt(a, b, c);
    });
    return result;
}

/**
 * 連続推定モード
 * - 再生中、サンプリング間隔ごとに pose 推定し samples へ追記
 * - returns: { stop: () => void }
 */
export function startContinuousDetection({ video, onSample, samplingMs = 100 }) {
    let stopped = false;
    let lastSampleAt = 0;
    let busy = false;

    async function loop(ts) {
        if (stopped) return;
        if (video.paused || video.ended) {
            requestAnimationFrame(loop);
            return;
        }
        if (!busy && ts - lastSampleAt >= samplingMs) {
            busy = true;
            lastSampleAt = ts;
            try {
                const pose = await detectPoseOnce(video);
                if (pose) {
                    onSample({
                        time: video.currentTime,
                        pose,
                        angles: computeJointAngles(pose)
                    });
                }
            } catch (err) {
                console.warn('連続ポーズ推定でエラー:', err);
            } finally {
                busy = false;
            }
        }
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    return { stop: () => { stopped = true; } };
}

export function exportPoseSeriesCSV(samples) {
    const angleKeys = ANGLE_DEFINITIONS.map(d => d.key);
    const angleLabels = ANGLE_DEFINITIONS.map(d => d.label);
    const header = ['time_sec', ...angleLabels].join(',');
    const lines = samples.map(s => {
        const cells = [s.time.toFixed(3), ...angleKeys.map(k => {
            const v = s.angles ? s.angles[k] : null;
            return v == null ? '' : v.toFixed(2);
        })];
        return cells.join(',');
    });
    return [header, ...lines].join('\n');
}

export const POSE_ANGLE_DEFS = ANGLE_DEFINITIONS.slice();
export const POSE_KEYPOINT_NAMES = KEYPOINT_NAMES.slice();
