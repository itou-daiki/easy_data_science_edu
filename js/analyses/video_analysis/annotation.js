// ==========================================
// 動画分析: 描画アノテーションモジュール
// 動画上に時刻ひも付けの図形を描画する
// ==========================================

const TOOLS = ['pen', 'line', 'arrow', 'circle', 'rect'];
const DEFAULT_DURATION_SEC = 2.0;
const DEFAULT_COLOR = '#ef4444';
const DEFAULT_THICKNESS = 4;

let _idCounter = 1;
function _newId() { return `anno-${Date.now().toString(36)}-${_idCounter++}`; }

/**
 * 描画状態のスロット（A/B 各動画用）を作る
 */
export function createAnnotationState() {
    return {
        items: [],
        tool: 'pen',
        color: DEFAULT_COLOR,
        thickness: DEFAULT_THICKNESS,
        durationSec: DEFAULT_DURATION_SEC,
        currentDraft: null
    };
}

/**
 * 正規化座標 → ピクセル
 */
function _toPx(point, width, height) {
    return { x: point.nx * width, y: point.ny * height };
}

/**
 * 図形を描画
 */
function _strokeShape(ctx, item, width, height) {
    ctx.lineWidth = item.thickness;
    ctx.strokeStyle = item.color;
    ctx.fillStyle = item.color;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const pts = item.points.map(p => _toPx(p, width, height));
    if (pts.length === 0) return;

    if (item.tool === 'pen') {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.stroke();
    } else if (item.tool === 'line' && pts.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
        ctx.stroke();
    } else if (item.tool === 'arrow' && pts.length >= 2) {
        const a = pts[0];
        const b = pts[pts.length - 1];
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        // 矢頭
        const angle = Math.atan2(b.y - a.y, b.x - a.x);
        const head = Math.max(10, item.thickness * 3);
        ctx.beginPath();
        ctx.moveTo(b.x, b.y);
        ctx.lineTo(b.x - head * Math.cos(angle - Math.PI / 7), b.y - head * Math.sin(angle - Math.PI / 7));
        ctx.lineTo(b.x - head * Math.cos(angle + Math.PI / 7), b.y - head * Math.sin(angle + Math.PI / 7));
        ctx.closePath();
        ctx.fill();
    } else if (item.tool === 'circle' && pts.length >= 2) {
        const a = pts[0];
        const b = pts[pts.length - 1];
        const cx = (a.x + b.x) / 2;
        const cy = (a.y + b.y) / 2;
        const rx = Math.abs(b.x - a.x) / 2;
        const ry = Math.abs(b.y - a.y) / 2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();
    } else if (item.tool === 'rect' && pts.length >= 2) {
        const a = pts[0];
        const b = pts[pts.length - 1];
        ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
    }
}

/**
 * 現在時刻に表示すべきアノテーションを描画
 * @param {Object} options - { showAll: boolean } 画像モードでは true で全件常時表示
 */
export function renderAnnotations(ctx, canvas, state, currentTime, options = {}) {
    const { items, currentDraft } = state;
    const w = canvas.width;
    const h = canvas.height;
    const showAll = options.showAll === true;

    items.forEach(item => {
        if (showAll) {
            ctx.globalAlpha = 1.0;
            _strokeShape(ctx, item, w, h);
            return;
        }
        if (currentTime >= item.time && currentTime <= item.time + item.durationSec) {
            const remaining = (item.time + item.durationSec) - currentTime;
            const fade = remaining < 0.3 ? Math.max(0.2, remaining / 0.3) : 1.0;
            ctx.globalAlpha = fade;
            _strokeShape(ctx, item, w, h);
        }
    });
    ctx.globalAlpha = 1.0;

    if (currentDraft) {
        ctx.globalAlpha = 0.85;
        _strokeShape(ctx, currentDraft, w, h);
        ctx.globalAlpha = 1.0;
    }
}

/**
 * 描画イベントの取り付け
 * @param {HTMLCanvasElement} canvas
 * @param {Object} state - createAnnotationState() の戻り値
 * @param {Function} getCurrentTime - () => number
 * @param {Function} onChange - 変更時に呼ぶ（再描画用）
 */
export function attachDrawingHandlers(canvas, state, getCurrentTime, onChange) {
    let isDrawing = false;

    function _normalize(evt) {
        const rect = canvas.getBoundingClientRect();
        const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
        const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
        return {
            nx: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
            ny: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height))
        };
    }

    function _start(evt) {
        evt.preventDefault();
        isDrawing = true;
        const point = _normalize(evt);
        state.currentDraft = {
            id: _newId(),
            tool: state.tool,
            color: state.color,
            thickness: state.thickness,
            durationSec: state.durationSec,
            time: getCurrentTime(),
            points: [point]
        };
        onChange();
    }

    function _move(evt) {
        if (!isDrawing || !state.currentDraft) return;
        evt.preventDefault();
        const point = _normalize(evt);
        if (state.currentDraft.tool === 'pen') {
            state.currentDraft.points.push(point);
        } else {
            // 直線・矢印・円・四角は始点と終点の2点のみ
            if (state.currentDraft.points.length === 1) {
                state.currentDraft.points.push(point);
            } else {
                state.currentDraft.points[1] = point;
            }
        }
        onChange();
    }

    function _end(evt) {
        if (!isDrawing) return;
        evt && evt.preventDefault && evt.preventDefault();
        isDrawing = false;
        if (state.currentDraft && state.currentDraft.points.length >= 1) {
            // 単発クリックで点だけは無視
            const isMeaningful = state.currentDraft.tool === 'pen'
                ? state.currentDraft.points.length >= 3
                : state.currentDraft.points.length >= 2;
            if (isMeaningful) {
                state.items.push(state.currentDraft);
            }
        }
        state.currentDraft = null;
        onChange();
    }

    canvas.addEventListener('mousedown', _start);
    canvas.addEventListener('mousemove', _move);
    canvas.addEventListener('mouseup', _end);
    canvas.addEventListener('mouseleave', _end);
    canvas.addEventListener('touchstart', _start, { passive: false });
    canvas.addEventListener('touchmove', _move, { passive: false });
    canvas.addEventListener('touchend', _end);

    return () => {
        canvas.removeEventListener('mousedown', _start);
        canvas.removeEventListener('mousemove', _move);
        canvas.removeEventListener('mouseup', _end);
        canvas.removeEventListener('mouseleave', _end);
        canvas.removeEventListener('touchstart', _start);
        canvas.removeEventListener('touchmove', _move);
        canvas.removeEventListener('touchend', _end);
    };
}

export function clearAnnotations(state) {
    state.items = [];
    state.currentDraft = null;
}

export function undoLastAnnotation(state) {
    state.items.pop();
}

export function exportAnnotationsJSON(state) {
    return JSON.stringify({ version: 1, items: state.items }, null, 2);
}

export function importAnnotationsJSON(state, jsonText) {
    const parsed = JSON.parse(jsonText);
    if (!parsed || !Array.isArray(parsed.items)) {
        throw new Error('アノテーションJSONの形式が不正です');
    }
    state.items = parsed.items.map(it => ({
        id: it.id || _newId(),
        tool: TOOLS.includes(it.tool) ? it.tool : 'pen',
        color: it.color || DEFAULT_COLOR,
        thickness: Number(it.thickness) || DEFAULT_THICKNESS,
        durationSec: Number(it.durationSec) || DEFAULT_DURATION_SEC,
        time: Number(it.time) || 0,
        points: Array.isArray(it.points) ? it.points.map(p => ({ nx: Number(p.nx) || 0, ny: Number(p.ny) || 0 })) : []
    }));
}

export const ANNOTATION_TOOLS = TOOLS.slice();
