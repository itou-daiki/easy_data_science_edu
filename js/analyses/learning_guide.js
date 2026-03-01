// ==========================================
// 機械学習 学習ガイド Module
// Interactive beginner's guide to ML concepts
// ==========================================
import { renderPlot } from '../utils.js';

// ==========================================
// Constants
// ==========================================

const THEME = Object.freeze({
    primary: '#4f46e5',
    primaryLight: '#6366f1',
    primaryBg: 'rgba(79, 70, 229, 0.08)',
    accent: '#06b6d4',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    textPrimary: '#1e293b',
    textSecondary: '#64748b',
    border: '#e2e8f0',
    cardBg: '#ffffff',
});

const TABS = Object.freeze([
    { id: 'ml-intro', label: '機械学習とは', icon: 'fa-brain' },
    { id: 'reg-cls', label: '回帰と分類', icon: 'fa-chart-line' },
    { id: 'evaluation', label: 'モデルの評価', icon: 'fa-clipboard-check' },
    { id: 'features', label: '特徴量と前処理', icon: 'fa-cogs' },
    { id: 'ensemble', label: 'アンサンブル学習', icon: 'fa-layer-group' },
    { id: 'app-guide', label: 'このアプリの使い方', icon: 'fa-book-open' },
]);

// ==========================================
// Styles (scoped with .lg- prefix)
// ==========================================

function buildStyles() {
    return `<style>
.lg-container { font-family: 'Inter', sans-serif; color: ${THEME.textPrimary}; }

.lg-tabs {
    display: flex; gap: 0.25rem; flex-wrap: wrap;
    border-bottom: 2px solid ${THEME.border}; margin-bottom: 1.5rem; padding-bottom: 0;
}
.lg-tab {
    padding: 0.6rem 1rem; border: none; background: none; cursor: pointer;
    font-size: 0.85rem; font-weight: 500; color: ${THEME.textSecondary};
    border-bottom: 2px solid transparent; margin-bottom: -2px;
    transition: all 0.2s ease; border-radius: 6px 6px 0 0;
}
.lg-tab:hover { color: ${THEME.primary}; background: ${THEME.primaryBg}; }
.lg-tab.active {
    color: ${THEME.primary}; border-bottom-color: ${THEME.primary};
    background: ${THEME.primaryBg}; font-weight: 600;
}
.lg-tab i { margin-right: 0.4rem; }

.lg-panel { display: none; animation: lgFadeIn 0.3s ease; }
.lg-panel.active { display: block; }
@keyframes lgFadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

.lg-section { margin-bottom: 2rem; }
.lg-section h3 {
    font-size: 1.15rem; font-weight: 700; color: ${THEME.primary};
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.lg-section p { line-height: 1.8; color: ${THEME.textSecondary}; margin-bottom: 0.75rem; }

.lg-card {
    background: ${THEME.cardBg}; border: 1px solid ${THEME.border};
    border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.lg-card-header {
    font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.5rem;
}

.lg-takeaway {
    background: linear-gradient(135deg, ${THEME.primaryBg}, rgba(6,182,212,0.06));
    border-left: 4px solid ${THEME.primary}; border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem; margin-top: 1.25rem;
}
.lg-takeaway-title {
    font-weight: 700; font-size: 0.9rem; color: ${THEME.primary};
    margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;
}
.lg-takeaway p { margin: 0; font-size: 0.9rem; line-height: 1.7; }

.lg-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.lg-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
@media (max-width: 768px) {
    .lg-grid-2, .lg-grid-3 { grid-template-columns: 1fr; }
}

.lg-badge {
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; color: white;
}
.lg-badge-blue { background: ${THEME.primary}; }
.lg-badge-green { background: ${THEME.success}; }
.lg-badge-orange { background: ${THEME.warning}; }
.lg-badge-red { background: ${THEME.danger}; }
.lg-badge-cyan { background: ${THEME.accent}; }

.lg-plot { min-height: 350px; margin: 1rem 0; border-radius: 8px; }

.lg-btn {
    padding: 0.5rem 1.2rem; border: none; border-radius: 8px; cursor: pointer;
    font-size: 0.85rem; font-weight: 600; transition: all 0.2s ease;
    display: inline-flex; align-items: center; gap: 0.4rem;
}
.lg-btn-primary { background: ${THEME.primary}; color: white; }
.lg-btn-primary:hover { background: ${THEME.primaryLight}; transform: translateY(-1px); }
.lg-btn-outline {
    background: transparent; color: ${THEME.primary};
    border: 1.5px solid ${THEME.primary};
}
.lg-btn-outline:hover { background: ${THEME.primaryBg}; }
.lg-btn-group { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.75rem 0; }

.lg-slider-row {
    display: flex; align-items: center; gap: 1rem; margin: 0.75rem 0;
}
.lg-slider-row label { font-size: 0.85rem; font-weight: 500; min-width: 120px; }
.lg-slider-row input[type="range"] { flex: 1; accent-color: ${THEME.primary}; }
.lg-slider-row .lg-slider-val {
    min-width: 50px; text-align: right; font-weight: 600;
    font-size: 0.85rem; color: ${THEME.primary};
}

.lg-diagram {
    display: flex; align-items: center; justify-content: center;
    gap: 1rem; flex-wrap: wrap; padding: 1.5rem; margin: 1rem 0;
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border-radius: 12px; border: 1px solid ${THEME.border};
}
.lg-diagram-box {
    padding: 0.75rem 1.25rem; border-radius: 10px; text-align: center;
    font-size: 0.85rem; font-weight: 600; min-width: 100px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.lg-diagram-arrow {
    font-size: 1.5rem; color: ${THEME.primary}; font-weight: bold;
}

.lg-step-list { counter-reset: lg-step; list-style: none; padding: 0; }
.lg-step-list li {
    counter-increment: lg-step; position: relative;
    padding: 0.75rem 0 0.75rem 3rem; border-bottom: 1px solid ${THEME.border};
}
.lg-step-list li:last-child { border-bottom: none; }
.lg-step-list li::before {
    content: counter(lg-step); position: absolute; left: 0; top: 0.65rem;
    width: 2rem; height: 2rem; background: ${THEME.primary}; color: white;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
}
.lg-step-list li strong { color: ${THEME.primary}; }

.lg-cm-grid {
    display: grid; grid-template-columns: auto 1fr 1fr; gap: 0;
    max-width: 300px; margin: 1rem auto; text-align: center;
    border: 1px solid ${THEME.border}; border-radius: 8px; overflow: hidden;
}
.lg-cm-cell {
    padding: 0.6rem; font-size: 0.85rem; font-weight: 600;
    border: 1px solid ${THEME.border};
}
.lg-cm-header { background: #f1f5f9; font-weight: 700; font-size: 0.75rem; }
.lg-cm-tp { background: rgba(16,185,129,0.15); color: #065f46; }
.lg-cm-tn { background: rgba(59,130,246,0.12); color: #1e40af; }
.lg-cm-fp { background: rgba(239,68,68,0.12); color: #991b1b; }
.lg-cm-fn { background: rgba(245,158,11,0.12); color: #92400e; }
</style>`;
}

// ==========================================
// Data Generation Utilities (pure functions)
// ==========================================

function generateLinearData(n, noise) {
    return Array.from({ length: n }, () => {
        const x = Math.random() * 10;
        const y = 2.5 * x + 3 + (Math.random() - 0.5) * noise;
        return { x, y };
    });
}

function generateClusterData(n) {
    const clusters = [
        { cx: 2, cy: 2, label: 'A' },
        { cx: 7, cy: 7, label: 'B' },
        { cx: 7, cy: 2, label: 'C' },
    ];
    return Array.from({ length: n }, () => {
        const cluster = clusters[Math.floor(Math.random() * clusters.length)];
        const x = cluster.cx + (Math.random() - 0.5) * 3;
        const y = cluster.cy + (Math.random() - 0.5) * 3;
        return { x, y, label: cluster.label };
    });
}

function generateClassificationData(n) {
    return Array.from({ length: n }, () => {
        const x = Math.random() * 10;
        const y = Math.random() * 10;
        const boundary = 0.5 * x + 2;
        const label = y > boundary ? 1 : 0;
        return { x, y, label };
    });
}

function fitSimpleLinearRegression(points) {
    const n = points.length;
    const sumX = points.reduce((s, p) => s + p.x, 0);
    const sumY = points.reduce((s, p) => s + p.y, 0);
    const sumXY = points.reduce((s, p) => s + p.x * p.y, 0);
    const sumX2 = points.reduce((s, p) => s + p.x * p.x, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    return Object.freeze({ slope, intercept });
}

function computeR2(points, slope, intercept) {
    const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
    const ssTot = points.reduce((s, p) => s + (p.y - meanY) ** 2, 0);
    const ssRes = points.reduce((s, p) => s + (p.y - (slope * p.x + intercept)) ** 2, 0);
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

function computeMAE(points, slope, intercept) {
    return points.reduce((s, p) => s + Math.abs(p.y - (slope * p.x + intercept)), 0) / points.length;
}

function computeRMSE(points, slope, intercept) {
    const mse = points.reduce((s, p) => s + (p.y - (slope * p.x + intercept)) ** 2, 0) / points.length;
    return Math.sqrt(mse);
}

// ==========================================
// Tab Renderers
// ==========================================

function renderTabMLIntro(panel) {
    const plotId1 = 'lg-intro-pattern-plot';
    const plotId2 = 'lg-intro-cluster-plot';

    panel.innerHTML = `
        <div class="lg-section">
            <h3><i class="fas fa-brain"></i> 機械学習とは？</h3>
            <p>機械学習とは、<strong>データからパターンを自動的に学習し、未知のデータに対して予測や判断を行う</strong>技術です。
               人間がルールを一つ一つプログラムするのではなく、コンピュータがデータの中にある規則性を見つけ出します。</p>

            <div class="lg-diagram">
                <div class="lg-diagram-box" style="background: rgba(79,70,229,0.12); color: ${THEME.primary};">
                    <i class="fas fa-database"></i><br>データ
                </div>
                <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                <div class="lg-diagram-box" style="background: rgba(6,182,212,0.12); color: ${THEME.accent};">
                    <i class="fas fa-cog"></i><br>学習アルゴリズム
                </div>
                <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                <div class="lg-diagram-box" style="background: rgba(16,185,129,0.12); color: ${THEME.success};">
                    <i class="fas fa-magic"></i><br>モデル
                </div>
                <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                <div class="lg-diagram-box" style="background: rgba(245,158,11,0.12); color: ${THEME.warning};">
                    <i class="fas fa-bullseye"></i><br>予測
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-sitemap"></i> 機械学習の種類</h3>
            <div class="lg-grid-3">
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-blue">教師あり学習</span>
                    </div>
                    <p>正解ラベル付きデータで学習します。</p>
                    <p><strong>回帰</strong>: 連続値を予測（例: 住宅価格）<br>
                       <strong>分類</strong>: カテゴリを予測（例: スパム判定）</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-green">教師なし学習</span>
                    </div>
                    <p>正解ラベルなしでデータの構造を発見します。</p>
                    <p><strong>クラスタリング</strong>: データのグループ分け<br>
                       <strong>次元削減</strong>: 情報を圧縮</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-orange">強化学習</span>
                    </div>
                    <p>試行錯誤で最適な行動を学習します。</p>
                    <p><strong>ゲームAI</strong>、<strong>ロボット制御</strong>、<strong>自動運転</strong>などに使用</p>
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-play-circle"></i> インタラクティブ体験: パターンを見つけよう</h3>
            <p>下のボタンをクリックすると、ランダムなデータが生成されます。データにどのようなパターンがあるか観察してみましょう。</p>
            <div class="lg-btn-group">
                <button class="lg-btn lg-btn-primary" id="lg-gen-linear">
                    <i class="fas fa-chart-line"></i> 直線パターンを生成
                </button>
                <button class="lg-btn lg-btn-outline" id="lg-gen-cluster">
                    <i class="fas fa-braille"></i> クラスターパターンを生成
                </button>
            </div>
            <div class="lg-grid-2">
                <div id="${plotId1}" class="lg-plot"></div>
                <div id="${plotId2}" class="lg-plot"></div>
            </div>
        </div>

        <div class="lg-takeaway">
            <div class="lg-takeaway-title"><i class="fas fa-lightbulb"></i> ポイント</div>
            <p>機械学習は「データの中に隠れたパターンを自動で発見する技術」です。
               予測したい値（目的変数）の種類に応じて、回帰・分類・クラスタリングなどの手法を使い分けます。</p>
        </div>
    `;

    const genLinearBtn = panel.querySelector('#lg-gen-linear');
    const genClusterBtn = panel.querySelector('#lg-gen-cluster');

    const drawLinear = () => {
        const points = generateLinearData(60, 5);
        const fit = fitSimpleLinearRegression(points);
        renderPlot(plotId1, [
            {
                x: points.map(p => p.x), y: points.map(p => p.y),
                mode: 'markers', type: 'scatter', name: 'データ点',
                marker: { color: THEME.primary, size: 7, opacity: 0.7 },
            },
            {
                x: [0, 10], y: [fit.intercept, fit.slope * 10 + fit.intercept],
                mode: 'lines', name: '学習した直線',
                line: { color: THEME.danger, width: 2, dash: 'dash' },
            },
        ], { title: '教師あり学習 (回帰)', xaxis: { title: '特徴量 X' }, yaxis: { title: '目的変数 Y' }, height: 340 });
    };

    const drawCluster = () => {
        const points = generateClusterData(90);
        const colorMap = { A: THEME.primary, B: THEME.success, C: THEME.warning };
        const groups = ['A', 'B', 'C'];
        const traces = groups.map(label => {
            const subset = points.filter(p => p.label === label);
            return {
                x: subset.map(p => p.x), y: subset.map(p => p.y),
                mode: 'markers', type: 'scatter', name: `クラスター ${label}`,
                marker: { color: colorMap[label], size: 8, opacity: 0.7 },
            };
        });
        renderPlot(plotId2, traces, {
            title: '教師なし学習 (クラスタリング)',
            xaxis: { title: 'X' }, yaxis: { title: 'Y' }, height: 340,
        });
    };

    genLinearBtn.addEventListener('click', drawLinear);
    genClusterBtn.addEventListener('click', drawCluster);
    drawLinear();
    drawCluster();
}

// ------------------------------------------
// Tab 2: 回帰と分類
// ------------------------------------------

function renderTabRegCls(panel) {
    const plotId = 'lg-regcls-plot';

    panel.innerHTML = `
        <div class="lg-section">
            <h3><i class="fas fa-chart-line"></i> 回帰と分類の違い</h3>
            <p>教師あり学習は大きく「回帰」と「分類」に分かれます。
               目的変数（予測したい値）が<strong>連続値なら回帰</strong>、<strong>カテゴリなら分類</strong>です。</p>
            <div class="lg-grid-2">
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-chart-area" style="color:${THEME.primary};"></i> 回帰 (Regression)
                    </div>
                    <p>連続的な数値を予測します。</p>
                    <p><strong>例</strong>: 住宅価格、気温、売上高<br>
                       <strong>出力</strong>: 実数値 (例: 3500万円)</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-tags" style="color:${THEME.success};"></i> 分類 (Classification)
                    </div>
                    <p>離散的なカテゴリを予測します。</p>
                    <p><strong>例</strong>: スパム/非スパム、犬/猫、病気の有無<br>
                       <strong>出力</strong>: クラスラベル (例: 陽性)</p>
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-exchange-alt"></i> インタラクティブ: 回帰 vs 分類を切り替えよう</h3>
            <p>同じデータセットでも、目的に応じて回帰と分類を切り替えられます。ボタンで表示を切り替えてみましょう。</p>
            <div class="lg-btn-group">
                <button class="lg-btn lg-btn-primary active" id="lg-mode-reg">
                    <i class="fas fa-chart-line"></i> 回帰モード
                </button>
                <button class="lg-btn lg-btn-outline" id="lg-mode-cls">
                    <i class="fas fa-border-all"></i> 分類モード
                </button>
            </div>
            <div class="lg-slider-row">
                <label>ノイズ量:</label>
                <input type="range" id="lg-noise-slider" min="1" max="15" value="5" step="1">
                <span class="lg-slider-val" id="lg-noise-val">5</span>
            </div>
            <div id="${plotId}" class="lg-plot" style="min-height: 380px;"></div>
            <div id="lg-regcls-info" style="margin-top: 0.5rem;"></div>
        </div>

        <div class="lg-takeaway">
            <div class="lg-takeaway-title"><i class="fas fa-lightbulb"></i> ポイント</div>
            <p>回帰は「どれくらい？」、分類は「どちらに属する？」を予測します。
               適切な問題設定が良いモデルの第一歩です。</p>
        </div>
    `;

    let currentMode = 'regression';

    const drawRegression = (noise) => {
        const points = generateLinearData(80, noise);
        const fit = fitSimpleLinearRegression(points);
        const r2 = computeR2(points, fit.slope, fit.intercept);
        renderPlot(plotId, [
            {
                x: points.map(p => p.x), y: points.map(p => p.y),
                mode: 'markers', type: 'scatter', name: 'データ点',
                marker: { color: THEME.primary, size: 7, opacity: 0.7 },
            },
            {
                x: [0, 10], y: [fit.intercept, fit.slope * 10 + fit.intercept],
                mode: 'lines', name: '回帰直線',
                line: { color: THEME.danger, width: 3 },
            },
        ], {
            title: '回帰: 連続値を予測', xaxis: { title: '特徴量 X' },
            yaxis: { title: '目的変数 Y' }, height: 380,
        });
        panel.querySelector('#lg-regcls-info').innerHTML =
            `<div class="lg-card"><strong>回帰直線</strong>: Y = ${fit.slope.toFixed(2)} * X + ${fit.intercept.toFixed(2)} &nbsp; | &nbsp; R&sup2; = ${r2.toFixed(3)}</div>`;
    };

    const drawClassification = (noise) => {
        const points = generateClassificationData(120);
        const addedNoise = noise / 15;
        const noisyPoints = points.map(p => {
            const flip = Math.random() < addedNoise * 0.3;
            return { ...p, label: flip ? (1 - p.label) : p.label };
        });
        const cls0 = noisyPoints.filter(p => p.label === 0);
        const cls1 = noisyPoints.filter(p => p.label === 1);
        renderPlot(plotId, [
            {
                x: cls0.map(p => p.x), y: cls0.map(p => p.y),
                mode: 'markers', type: 'scatter', name: 'クラス 0',
                marker: { color: THEME.primary, size: 7, symbol: 'circle', opacity: 0.7 },
            },
            {
                x: cls1.map(p => p.x), y: cls1.map(p => p.y),
                mode: 'markers', type: 'scatter', name: 'クラス 1',
                marker: { color: THEME.danger, size: 7, symbol: 'diamond', opacity: 0.7 },
            },
            {
                x: [0, 10], y: [2, 7],
                mode: 'lines', name: '決定境界',
                line: { color: THEME.success, width: 3, dash: 'dash' },
            },
        ], {
            title: '分類: カテゴリを予測', xaxis: { title: '特徴量 X', range: [0, 10] },
            yaxis: { title: '特徴量 Y', range: [0, 10] }, height: 380,
        });
        const total = noisyPoints.length;
        const correct = noisyPoints.filter(p =>
            (p.label === 1 && p.y > 0.5 * p.x + 2) ||
            (p.label === 0 && p.y <= 0.5 * p.x + 2)
        ).length;
        panel.querySelector('#lg-regcls-info').innerHTML =
            `<div class="lg-card"><strong>決定境界</strong>: Y = 0.5 * X + 2 &nbsp; | &nbsp; 正解率 = ${(correct / total * 100).toFixed(1)}%</div>`;
    };

    const redraw = () => {
        const noise = Number(panel.querySelector('#lg-noise-slider').value);
        if (currentMode === 'regression') { drawRegression(noise); }
        else { drawClassification(noise); }
    };

    panel.querySelector('#lg-mode-reg').addEventListener('click', () => {
        currentMode = 'regression';
        panel.querySelector('#lg-mode-reg').className = 'lg-btn lg-btn-primary';
        panel.querySelector('#lg-mode-cls').className = 'lg-btn lg-btn-outline';
        redraw();
    });
    panel.querySelector('#lg-mode-cls').addEventListener('click', () => {
        currentMode = 'classification';
        panel.querySelector('#lg-mode-cls').className = 'lg-btn lg-btn-primary';
        panel.querySelector('#lg-mode-reg').className = 'lg-btn lg-btn-outline';
        redraw();
    });
    panel.querySelector('#lg-noise-slider').addEventListener('input', (e) => {
        panel.querySelector('#lg-noise-val').textContent = e.target.value;
        redraw();
    });
    redraw();
}

// ------------------------------------------
// Tab 3: モデルの評価
// ------------------------------------------

function renderTabEvaluation(panel) {
    const splitPlotId = 'lg-eval-split-plot';
    const metricPlotId = 'lg-eval-metric-plot';
    const curvePlotId = 'lg-eval-curve-plot';

    panel.innerHTML = `
        <div class="lg-section">
            <h3><i class="fas fa-cut"></i> 訓練データとテストデータの分割</h3>
            <p>モデルの本当の実力を測るために、データを<strong>訓練用</strong>と<strong>テスト用</strong>に分けます。
               訓練データで学習し、テストデータで評価します。</p>
            <div class="lg-slider-row">
                <label>訓練データの割合:</label>
                <input type="range" id="lg-split-slider" min="50" max="90" value="70" step="5">
                <span class="lg-slider-val" id="lg-split-val">70%</span>
            </div>
            <div id="${splitPlotId}" class="lg-plot" style="min-height: 300px;"></div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-ruler"></i> 回帰の評価指標</h3>
            <p>スライダーで予測の精度を変化させ、評価指標がどう変わるか観察しましょう。</p>
            <div class="lg-slider-row">
                <label>予測のノイズ:</label>
                <input type="range" id="lg-pred-noise" min="0" max="20" value="5" step="1">
                <span class="lg-slider-val" id="lg-pred-noise-val">5</span>
            </div>
            <div id="${metricPlotId}" class="lg-plot" style="min-height: 350px;"></div>
            <div id="lg-eval-metrics" class="lg-grid-3" style="margin-top: 0.75rem;"></div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-th"></i> 分類の評価指標: 混同行列</h3>
            <p>分類の評価は<strong>混同行列</strong>が基本です。予測と実際の組み合わせを4つのセルで整理します。</p>
            <div class="lg-slider-row">
                <label>分類の精度:</label>
                <input type="range" id="lg-cls-accuracy" min="50" max="99" value="80" step="1">
                <span class="lg-slider-val" id="lg-cls-accuracy-val">80%</span>
            </div>
            <div id="lg-cm-display" style="margin: 1rem 0;"></div>
            <div id="lg-cls-metrics" class="lg-grid-3" style="margin-top: 0.75rem;"></div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-wave-square"></i> 過学習と未学習</h3>
            <p>モデルの複雑さと汎化性能の関係を学習曲線で見てみましょう。</p>
            <div class="lg-btn-group">
                <button class="lg-btn lg-btn-outline" id="lg-curve-underfit">未学習 (Underfitting)</button>
                <button class="lg-btn lg-btn-primary" id="lg-curve-good">適切なモデル</button>
                <button class="lg-btn lg-btn-outline" id="lg-curve-overfit">過学習 (Overfitting)</button>
            </div>
            <div id="${curvePlotId}" class="lg-plot" style="min-height: 350px;"></div>
        </div>

        <div class="lg-takeaway">
            <div class="lg-takeaway-title"><i class="fas fa-lightbulb"></i> ポイント</div>
            <p>モデルの評価は「未知のデータに対する予測精度」が重要です。訓練データだけでの評価は過学習を見逃します。
               複数の指標を総合的に判断し、過学習と未学習のバランスを取りましょう。</p>
        </div>
    `;

    // --- Split visualization ---
    const drawSplit = () => {
        const ratio = Number(panel.querySelector('#lg-split-slider').value);
        panel.querySelector('#lg-split-val').textContent = `${ratio}%`;
        const n = 40;
        const trainN = Math.round(n * ratio / 100);
        const xAll = Array.from({ length: n }, (_, i) => i + 1);
        renderPlot(splitPlotId, [
            {
                x: xAll.slice(0, trainN), y: Array(trainN).fill(1),
                type: 'bar', name: `訓練データ (${trainN}件)`,
                marker: { color: THEME.primary },
            },
            {
                x: xAll.slice(trainN), y: Array(n - trainN).fill(1),
                type: 'bar', name: `テストデータ (${n - trainN}件)`,
                marker: { color: THEME.warning },
            },
        ], {
            title: 'データ分割', barmode: 'stack',
            xaxis: { title: 'サンプル番号' }, yaxis: { visible: false, range: [0, 1.5] },
            height: 250, showlegend: true,
        });
    };
    panel.querySelector('#lg-split-slider').addEventListener('input', drawSplit);
    drawSplit();

    // --- Regression metrics ---
    const drawRegressionMetrics = () => {
        const noise = Number(panel.querySelector('#lg-pred-noise').value);
        panel.querySelector('#lg-pred-noise-val').textContent = noise;
        const points = generateLinearData(60, 2);
        const fit = fitSimpleLinearRegression(points);
        const predicted = points.map(p => ({
            ...p,
            yPred: fit.slope * p.x + fit.intercept + (Math.random() - 0.5) * noise,
        }));
        const r2 = computeR2(points, fit.slope, fit.intercept);
        const mae = predicted.reduce((s, p) => s + Math.abs(p.y - p.yPred), 0) / predicted.length;
        const rmse = Math.sqrt(predicted.reduce((s, p) => s + (p.y - p.yPred) ** 2, 0) / predicted.length);

        renderPlot(metricPlotId, [
            {
                x: predicted.map(p => p.y), y: predicted.map(p => p.yPred),
                mode: 'markers', type: 'scatter', name: '予測 vs 実測',
                marker: { color: THEME.primary, size: 7, opacity: 0.7 },
            },
            {
                x: [0, 30], y: [0, 30], mode: 'lines', name: '理想線 (y=x)',
                line: { color: THEME.danger, dash: 'dash', width: 2 },
            },
        ], {
            title: '実測値 vs 予測値',
            xaxis: { title: '実測値' }, yaxis: { title: '予測値' }, height: 350,
        });

        panel.querySelector('#lg-eval-metrics').innerHTML = `
            <div class="lg-card" style="text-align:center;">
                <div style="font-size:0.8rem;color:${THEME.textSecondary};">R&sup2; (決定係数)</div>
                <div style="font-size:1.5rem;font-weight:700;color:${THEME.primary};">${Math.max(0, r2 - noise * 0.02).toFixed(3)}</div>
                <div style="font-size:0.75rem;color:${THEME.textSecondary};">1に近いほど良い</div>
            </div>
            <div class="lg-card" style="text-align:center;">
                <div style="font-size:0.8rem;color:${THEME.textSecondary};">MAE (平均絶対誤差)</div>
                <div style="font-size:1.5rem;font-weight:700;color:${THEME.warning};">${mae.toFixed(3)}</div>
                <div style="font-size:0.75rem;color:${THEME.textSecondary};">0に近いほど良い</div>
            </div>
            <div class="lg-card" style="text-align:center;">
                <div style="font-size:0.8rem;color:${THEME.textSecondary};">RMSE (二乗平均平方根誤差)</div>
                <div style="font-size:1.5rem;font-weight:700;color:${THEME.danger};">${rmse.toFixed(3)}</div>
                <div style="font-size:0.75rem;color:${THEME.textSecondary};">0に近いほど良い</div>
            </div>
        `;
    };
    panel.querySelector('#lg-pred-noise').addEventListener('input', drawRegressionMetrics);
    drawRegressionMetrics();

    // --- Confusion matrix ---
    const drawConfusionMatrix = () => {
        const accuracy = Number(panel.querySelector('#lg-cls-accuracy').value) / 100;
        panel.querySelector('#lg-cls-accuracy-val').textContent = `${Math.round(accuracy * 100)}%`;
        const n = 200;
        const nPos = 100;
        const nNeg = 100;
        const tp = Math.round(nPos * (accuracy + (Math.random() - 0.5) * 0.05));
        const fn = nPos - tp;
        const tn = Math.round(nNeg * (accuracy + (Math.random() - 0.5) * 0.05));
        const fp = nNeg - tn;

        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
        const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        const acc = (tp + tn) / n;

        panel.querySelector('#lg-cm-display').innerHTML = `
            <div class="lg-cm-grid">
                <div class="lg-cm-cell lg-cm-header"></div>
                <div class="lg-cm-cell lg-cm-header">予測: 陽性</div>
                <div class="lg-cm-cell lg-cm-header">予測: 陰性</div>
                <div class="lg-cm-cell lg-cm-header">実際: 陽性</div>
                <div class="lg-cm-cell lg-cm-tp">TP: ${tp}</div>
                <div class="lg-cm-cell lg-cm-fn">FN: ${fn}</div>
                <div class="lg-cm-cell lg-cm-header">実際: 陰性</div>
                <div class="lg-cm-cell lg-cm-fp">FP: ${fp}</div>
                <div class="lg-cm-cell lg-cm-tn">TN: ${tn}</div>
            </div>
        `;
        panel.querySelector('#lg-cls-metrics').innerHTML = `
            <div class="lg-card" style="text-align:center;">
                <div style="font-size:0.8rem;color:${THEME.textSecondary};">Accuracy (正解率)</div>
                <div style="font-size:1.4rem;font-weight:700;color:${THEME.primary};">${(acc * 100).toFixed(1)}%</div>
            </div>
            <div class="lg-card" style="text-align:center;">
                <div style="font-size:0.8rem;color:${THEME.textSecondary};">Precision (適合率)</div>
                <div style="font-size:1.4rem;font-weight:700;color:${THEME.success};">${(precision * 100).toFixed(1)}%</div>
                <div style="font-size:0.7rem;color:${THEME.textSecondary};">陽性と予測した中の正解率</div>
            </div>
            <div class="lg-card" style="text-align:center;">
                <div style="font-size:0.8rem;color:${THEME.textSecondary};">Recall (再現率)</div>
                <div style="font-size:1.4rem;font-weight:700;color:${THEME.warning};">${(recall * 100).toFixed(1)}%</div>
                <div style="font-size:0.7rem;color:${THEME.textSecondary};">実際の陽性を正しく検出した率</div>
            </div>
        `;
    };
    panel.querySelector('#lg-cls-accuracy').addEventListener('input', drawConfusionMatrix);
    drawConfusionMatrix();

    // --- Learning curves ---
    const drawLearningCurve = (type) => {
        const sizes = Array.from({ length: 8 }, (_, i) => (i + 1) * 50);
        let trainScores, testScores;
        if (type === 'underfit') {
            trainScores = sizes.map(() => 0.4 + Math.random() * 0.05);
            testScores = sizes.map(() => 0.35 + Math.random() * 0.05);
        } else if (type === 'overfit') {
            trainScores = sizes.map(() => 0.98 + Math.random() * 0.02);
            testScores = sizes.map((_, i) => 0.55 + i * 0.01 + Math.random() * 0.03);
        } else {
            trainScores = sizes.map((_, i) => 0.95 - (7 - i) * 0.01 + Math.random() * 0.01);
            testScores = sizes.map((_, i) => 0.85 + i * 0.008 + Math.random() * 0.01);
        }

        renderPlot(curvePlotId, [
            {
                x: sizes, y: trainScores, mode: 'lines+markers', name: '訓練スコア',
                line: { color: THEME.primary, width: 2 }, marker: { size: 6 },
            },
            {
                x: sizes, y: testScores, mode: 'lines+markers', name: 'テストスコア',
                line: { color: THEME.danger, width: 2 }, marker: { size: 6 },
            },
        ], {
            title: `学習曲線 - ${type === 'underfit' ? '未学習' : type === 'overfit' ? '過学習' : '適切'}`,
            xaxis: { title: '訓練サンプル数' }, yaxis: { title: 'スコア', range: [0, 1.05] },
            height: 350,
            shapes: type === 'overfit' ? [{
                type: 'rect', xref: 'paper', yref: 'paper',
                x0: 0, y0: 0, x1: 1, y1: 1,
                fillcolor: 'rgba(239,68,68,0.03)', line: { width: 0 },
            }] : [],
        });
    };

    panel.querySelector('#lg-curve-underfit').addEventListener('click', () => {
        drawLearningCurve('underfit');
        setActiveBtn(panel, 'lg-curve-underfit', ['lg-curve-underfit', 'lg-curve-good', 'lg-curve-overfit']);
    });
    panel.querySelector('#lg-curve-good').addEventListener('click', () => {
        drawLearningCurve('good');
        setActiveBtn(panel, 'lg-curve-good', ['lg-curve-underfit', 'lg-curve-good', 'lg-curve-overfit']);
    });
    panel.querySelector('#lg-curve-overfit').addEventListener('click', () => {
        drawLearningCurve('overfit');
        setActiveBtn(panel, 'lg-curve-overfit', ['lg-curve-underfit', 'lg-curve-good', 'lg-curve-overfit']);
    });
    drawLearningCurve('good');
}

// ------------------------------------------
// Tab 4: 特徴量とデータ前処理
// ------------------------------------------

function renderTabFeatures(panel) {
    const plotId = 'lg-feat-importance-plot';

    panel.innerHTML = `
        <div class="lg-section">
            <h3><i class="fas fa-columns"></i> 特徴量とは？</h3>
            <p><strong>特徴量 (Feature)</strong> とは、モデルが予測に使用するデータの各属性のことです。
               例えば住宅価格予測なら「面積」「築年数」「駅からの距離」などが特徴量です。</p>
            <div class="lg-diagram">
                <div class="lg-diagram-box" style="background:rgba(79,70,229,0.1);color:${THEME.primary};">
                    <i class="fas fa-home"></i><br>面積
                </div>
                <div class="lg-diagram-box" style="background:rgba(6,182,212,0.1);color:${THEME.accent};">
                    <i class="fas fa-calendar"></i><br>築年数
                </div>
                <div class="lg-diagram-box" style="background:rgba(16,185,129,0.1);color:${THEME.success};">
                    <i class="fas fa-train"></i><br>駅距離
                </div>
                <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                <div class="lg-diagram-box" style="background:rgba(245,158,11,0.1);color:${THEME.warning};">
                    <i class="fas fa-yen-sign"></i><br>価格予測
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-broom"></i> データ前処理の重要性</h3>
            <div class="lg-grid-3">
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-question-circle" style="color:${THEME.danger};"></i> 欠損値処理
                    </div>
                    <p>データに穴がある場合の対処法:</p>
                    <p><span class="lg-badge lg-badge-blue">平均値で埋める</span>
                       <span class="lg-badge lg-badge-green">中央値で埋める</span>
                       <span class="lg-badge lg-badge-red">行を削除</span></p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-balance-scale" style="color:${THEME.primary};"></i> スケーリング
                    </div>
                    <p>特徴量の値の範囲を揃えます:</p>
                    <p><strong>標準化</strong>: 平均0、分散1に変換<br>
                       <strong>正規化</strong>: 0~1の範囲に変換</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-code" style="color:${THEME.success};"></i> エンコーディング
                    </div>
                    <p>カテゴリを数値に変換:</p>
                    <p><strong>ラベル</strong>: A=0, B=1, C=2<br>
                       <strong>One-Hot</strong>: A=[1,0,0], B=[0,1,0]</p>
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-sort-amount-down"></i> 特徴量重要度</h3>
            <p>すべての特徴量が等しく重要とは限りません。下のチャートは、住宅価格予測における各特徴量の重要度のシミュレーションです。
               ボタンをクリックして異なるシナリオを確認しましょう。</p>
            <div class="lg-btn-group">
                <button class="lg-btn lg-btn-primary" id="lg-feat-house">
                    <i class="fas fa-home"></i> 住宅価格
                </button>
                <button class="lg-btn lg-btn-outline" id="lg-feat-health">
                    <i class="fas fa-heartbeat"></i> 健康診断
                </button>
                <button class="lg-btn lg-btn-outline" id="lg-feat-sales">
                    <i class="fas fa-shopping-cart"></i> 売上予測
                </button>
            </div>
            <div id="${plotId}" class="lg-plot" style="min-height: 350px;"></div>
        </div>

        <div class="lg-takeaway">
            <div class="lg-takeaway-title"><i class="fas fa-lightbulb"></i> ポイント</div>
            <p>良い特徴量はモデルの精度を大きく向上させます。前処理で欠損値を適切に扱い、
               スケーリングやエンコーディングでデータを整えることが、成功への近道です。</p>
        </div>
    `;

    const featureScenarios = Object.freeze({
        house: {
            names: ['面積(m2)', '駅距離(分)', '築年数', '階数', '部屋数', '日当たり', '治安スコア', '周辺施設数'],
            values: [0.35, 0.22, 0.18, 0.08, 0.07, 0.04, 0.04, 0.02],
        },
        health: {
            names: ['年齢', 'BMI', '血圧', '血糖値', 'コレステロール', '運動頻度', '喫煙歴', '飲酒量'],
            values: [0.15, 0.28, 0.20, 0.12, 0.10, 0.07, 0.05, 0.03],
        },
        sales: {
            names: ['広告費', '季節', '価格', '競合数', '在庫量', 'SNSフォロワー', '天気', '曜日'],
            values: [0.30, 0.18, 0.20, 0.10, 0.08, 0.07, 0.04, 0.03],
        },
    });

    const drawFeatureImportance = (key) => {
        const scenario = featureScenarios[key];
        const indices = scenario.values.map((_, i) => i).sort((a, b) => scenario.values[a] - scenario.values[b]);
        renderPlot(plotId, [{
            type: 'bar', orientation: 'h',
            x: indices.map(i => scenario.values[i]),
            y: indices.map(i => scenario.names[i]),
            marker: {
                color: indices.map(i => {
                    const v = scenario.values[i];
                    return v > 0.2 ? THEME.primary : v > 0.1 ? THEME.accent : THEME.textSecondary;
                }),
            },
        }], {
            title: '特徴量重要度', xaxis: { title: '重要度' },
            margin: { l: 120 }, height: 350,
        });
    };

    panel.querySelector('#lg-feat-house').addEventListener('click', () => {
        drawFeatureImportance('house');
        setActiveBtn(panel, 'lg-feat-house', ['lg-feat-house', 'lg-feat-health', 'lg-feat-sales']);
    });
    panel.querySelector('#lg-feat-health').addEventListener('click', () => {
        drawFeatureImportance('health');
        setActiveBtn(panel, 'lg-feat-health', ['lg-feat-house', 'lg-feat-health', 'lg-feat-sales']);
    });
    panel.querySelector('#lg-feat-sales').addEventListener('click', () => {
        drawFeatureImportance('sales');
        setActiveBtn(panel, 'lg-feat-sales', ['lg-feat-house', 'lg-feat-health', 'lg-feat-sales']);
    });
    drawFeatureImportance('house');
}

// ------------------------------------------
// Tab 5: アンサンブル学習
// ------------------------------------------

function renderTabEnsemble(panel) {
    const plotId = 'lg-ensemble-plot';

    panel.innerHTML = `
        <div class="lg-section">
            <h3><i class="fas fa-layer-group"></i> アンサンブル学習とは？</h3>
            <p>複数のモデルを組み合わせることで、単一モデルよりも高い予測精度を達成する手法です。
               「三人寄れば文殊の知恵」と同じ考え方です。</p>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-boxes"></i> 主なアンサンブル手法</h3>
            <div class="lg-grid-2">
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-blue">Bagging</span> ランダムフォレスト
                    </div>
                    <p>データをランダムに抽出して複数のモデルを<strong>並列</strong>に学習し、結果を平均（回帰）または多数決（分類）で統合します。</p>
                    <div class="lg-diagram" style="padding:1rem;">
                        <div style="display:flex;flex-direction:column;gap:0.3rem;align-items:center;">
                            <div class="lg-diagram-box" style="background:rgba(79,70,229,0.1);color:${THEME.primary};font-size:0.75rem;">モデル1</div>
                            <div class="lg-diagram-box" style="background:rgba(79,70,229,0.1);color:${THEME.primary};font-size:0.75rem;">モデル2</div>
                            <div class="lg-diagram-box" style="background:rgba(79,70,229,0.1);color:${THEME.primary};font-size:0.75rem;">モデル3</div>
                        </div>
                        <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                        <div class="lg-diagram-box" style="background:rgba(16,185,129,0.15);color:${THEME.success};">
                            多数決 / 平均
                        </div>
                    </div>
                    <p><strong>特徴</strong>: 分散を減らし、過学習を抑制</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-green">Boosting</span> 勾配ブースティング
                    </div>
                    <p>前のモデルの<strong>誤差を修正する</strong>ように次のモデルを<strong>逐次的</strong>に学習します。</p>
                    <div class="lg-diagram" style="padding:1rem;">
                        <div class="lg-diagram-box" style="background:rgba(79,70,229,0.1);color:${THEME.primary};font-size:0.75rem;">モデル1</div>
                        <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                        <div class="lg-diagram-box" style="background:rgba(6,182,212,0.1);color:${THEME.accent};font-size:0.75rem;">誤差修正</div>
                        <div class="lg-diagram-arrow"><i class="fas fa-arrow-right"></i></div>
                        <div class="lg-diagram-box" style="background:rgba(16,185,129,0.15);color:${THEME.success};font-size:0.75rem;">モデル2</div>
                    </div>
                    <p><strong>特徴</strong>: バイアスを減らし、高精度を実現</p>
                </div>
            </div>
        </div>

        <div class="lg-section">
            <div class="lg-grid-2">
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-cyan">Stacking</span> スタッキング
                    </div>
                    <p>複数のモデルの予測を<strong>別のモデル（メタモデル）</strong>への入力として使い、最終予測を行います。
                       異なる種類のモデルの強みを組み合わせられます。</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <span class="lg-badge lg-badge-orange">Blending</span> ブレンディング
                    </div>
                    <p>Stackingに似ていますが、ホールドアウトデータの予測値を使ってメタモデルを訓練する、
                       よりシンプルな手法です。データリークのリスクが低いのが利点です。</p>
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-chart-bar"></i> インタラクティブ: アンサンブルの効果</h3>
            <p>個別モデルとアンサンブルの精度を比較してみましょう。ボタンでモデル数を変更できます。</p>
            <div class="lg-slider-row">
                <label>モデル数:</label>
                <input type="range" id="lg-ensemble-count" min="1" max="20" value="5" step="1">
                <span class="lg-slider-val" id="lg-ensemble-count-val">5</span>
            </div>
            <div id="${plotId}" class="lg-plot" style="min-height: 380px;"></div>
            <div id="lg-ensemble-info" style="margin-top:0.5rem;"></div>
        </div>

        <div class="lg-takeaway">
            <div class="lg-takeaway-title"><i class="fas fa-lightbulb"></i> ポイント</div>
            <p>アンサンブル学習は多くのコンペティションで上位入賞の鍵となる手法です。
               Baggingは過学習防止に、Boostingは精度向上に強く、
               Stackingで異なるモデルの長所を組み合わせることで更なる改善が期待できます。</p>
        </div>
    `;

    const drawEnsembleEffect = () => {
        const nModels = Number(panel.querySelector('#lg-ensemble-count').value);
        panel.querySelector('#lg-ensemble-count-val').textContent = nModels;

        const trueVal = 50;
        const individualPreds = Array.from({ length: nModels }, () =>
            trueVal + (Math.random() - 0.5) * 30
        );
        const runningAvgs = individualPreds.map((_, i) => {
            const subset = individualPreds.slice(0, i + 1);
            return subset.reduce((a, b) => a + b, 0) / subset.length;
        });
        const modelLabels = Array.from({ length: nModels }, (_, i) => `モデル${i + 1}`);
        const ensembleError = Math.abs(runningAvgs[runningAvgs.length - 1] - trueVal);
        const avgIndividualError = individualPreds.reduce((s, v) => s + Math.abs(v - trueVal), 0) / nModels;

        renderPlot(plotId, [
            {
                x: modelLabels, y: individualPreds,
                type: 'bar', name: '個別モデルの予測',
                marker: { color: THEME.primaryLight, opacity: 0.7 },
            },
            {
                x: modelLabels, y: runningAvgs,
                mode: 'lines+markers', name: 'アンサンブル平均',
                line: { color: THEME.success, width: 3 }, marker: { size: 8 },
            },
            {
                x: modelLabels, y: Array(nModels).fill(trueVal),
                mode: 'lines', name: '真の値',
                line: { color: THEME.danger, dash: 'dash', width: 2 },
            },
        ], {
            title: 'アンサンブルによる予測の安定化',
            xaxis: { title: 'モデル' }, yaxis: { title: '予測値' }, height: 380,
            legend: { x: 0.65, y: 1 },
        });

        panel.querySelector('#lg-ensemble-info').innerHTML = `
            <div class="lg-grid-2">
                <div class="lg-card" style="text-align:center;">
                    <div style="font-size:0.8rem;color:${THEME.textSecondary};">個別モデルの平均誤差</div>
                    <div style="font-size:1.3rem;font-weight:700;color:${THEME.warning};">${avgIndividualError.toFixed(2)}</div>
                </div>
                <div class="lg-card" style="text-align:center;">
                    <div style="font-size:0.8rem;color:${THEME.textSecondary};">アンサンブルの誤差</div>
                    <div style="font-size:1.3rem;font-weight:700;color:${THEME.success};">${ensembleError.toFixed(2)}</div>
                </div>
            </div>
        `;
    };

    panel.querySelector('#lg-ensemble-count').addEventListener('input', drawEnsembleEffect);
    drawEnsembleEffect();
}

// ------------------------------------------
// Tab 6: このアプリの使い方
// ------------------------------------------

function renderTabAppGuide(panel) {
    panel.innerHTML = `
        <div class="lg-section">
            <h3><i class="fas fa-rocket"></i> easyDataScience の使い方</h3>
            <p>このアプリは、CSVファイルをアップロードするだけで、データ分析から機械学習モデルの構築まで、
               すべてブラウザ上で完結するツールです。</p>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-list-ol"></i> ステップバイステップ ワークフロー</h3>
            <ol class="lg-step-list">
                <li>
                    <strong>データのアップロード</strong><br>
                    CSVファイルをドラッグ＆ドロップ、またはクリックしてアップロードします。
                    数値変数とカテゴリ変数は自動判別されます。
                </li>
                <li>
                    <strong>探索的データ分析 (EDA)</strong><br>
                    データの概要、分布、相関、欠損値を確認します。
                    この段階でデータの全体像を把握することが重要です。
                </li>
                <li>
                    <strong>データ前処理</strong><br>
                    欠損値の補完、スケーリング、エンコーディングなどを行います。
                    適切な前処理がモデルの精度を大きく左右します。
                </li>
                <li>
                    <strong>分析タイプの選択</strong><br>
                    目的変数の種類に応じて「回帰」か「分類」を選択します。
                    連続値なら回帰、カテゴリなら分類です。
                </li>
                <li>
                    <strong>モデルの学習と比較</strong><br>
                    複数のアルゴリズムが自動的に比較され、最適なモデルが推薦されます。
                    交差検証により信頼性の高い評価が行われます。
                </li>
                <li>
                    <strong>結果の解釈</strong><br>
                    評価指標、特徴量重要度、SHAP値などでモデルの振る舞いを理解します。
                    予測結果はCSVでダウンロードできます。
                </li>
            </ol>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-question-circle"></i> どの分析を選べばいい？</h3>
            <div class="lg-grid-2">
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-search" style="color:${THEME.primary};"></i> 探索的データ分析 (EDA)
                    </div>
                    <p><strong>いつ使う？</strong> まずはこれから始めましょう。データの全体像を把握するために必ず最初に行います。</p>
                    <p><strong>わかること:</strong> 分布、相関、欠損値の状況</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-wrench" style="color:${THEME.accent};"></i> データ前処理
                    </div>
                    <p><strong>いつ使う？</strong> EDAで問題が見つかった場合（欠損値、外れ値、スケールの違いなど）。</p>
                    <p><strong>できること:</strong> 欠損値処理、スケーリング、エンコーディング</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-chart-line" style="color:${THEME.success};"></i> 回帰分析
                    </div>
                    <p><strong>いつ使う？</strong> 目的変数が<strong>連続値</strong>（価格、温度、売上など）の場合。</p>
                    <p><strong>アルゴリズム:</strong> 線形回帰、Ridge、Lasso、決定木、ランダムフォレスト、勾配ブースティングなど</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">
                        <i class="fas fa-tags" style="color:${THEME.warning};"></i> 分類分析
                    </div>
                    <p><strong>いつ使う？</strong> 目的変数が<strong>カテゴリ</strong>（Yes/No、A/B/C など）の場合。</p>
                    <p><strong>アルゴリズム:</strong> ロジスティック回帰、SVM、決定木、ランダムフォレスト、勾配ブースティングなど</p>
                </div>
            </div>
        </div>

        <div class="lg-section">
            <h3><i class="fas fa-chart-pie"></i> 結果の読み方</h3>
            <div class="lg-grid-2">
                <div class="lg-card">
                    <div class="lg-card-header">回帰の場合</div>
                    <p><strong>R&sup2; (決定係数)</strong>: 1に近いほど良い予測。0.7以上なら良好。<br>
                       <strong>RMSE</strong>: 予測誤差の大きさ。値が小さいほど良い。<br>
                       <strong>残差プロット</strong>: ランダムに散らばっていれば良い。パターンがあれば改善の余地あり。</p>
                </div>
                <div class="lg-card">
                    <div class="lg-card-header">分類の場合</div>
                    <p><strong>Accuracy</strong>: 全体の正解率。クラスが不均衡な場合は注意。<br>
                       <strong>F1スコア</strong>: PrecisionとRecallのバランス指標。<br>
                       <strong>混同行列</strong>: どのクラスを間違えやすいかがわかる。</p>
                </div>
            </div>
        </div>

        <div class="lg-takeaway">
            <div class="lg-takeaway-title"><i class="fas fa-lightbulb"></i> ポイント</div>
            <p>分析は「EDA → 前処理 → モデル構築 → 評価」の順で進めましょう。
               急いでモデルを作るより、データをしっかり理解することが成功の鍵です。
               このアプリはすべてブラウザ上で動作し、データがサーバーに送信されることはありません。</p>
        </div>
    `;
}

// ==========================================
// Utility: active button toggle
// ==========================================

function setActiveBtn(panel, activeId, allIds) {
    allIds.forEach(id => {
        const btn = panel.querySelector(`#${id}`);
        if (!btn) return;
        if (id === activeId) {
            btn.className = 'lg-btn lg-btn-primary';
        } else {
            btn.className = 'lg-btn lg-btn-outline';
        }
    });
}

// ==========================================
// Tab Navigation Setup
// ==========================================

function setupTabNavigation(container) {
    const tabButtons = container.querySelectorAll('.lg-tab');
    const panels = container.querySelectorAll('.lg-panel');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetPanel = container.querySelector(`#lg-panel-${btn.dataset.tab}`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}

// ==========================================
// Tab Content Renderers Map
// ==========================================

const TAB_RENDERERS = Object.freeze({
    'ml-intro': renderTabMLIntro,
    'reg-cls': renderTabRegCls,
    'evaluation': renderTabEvaluation,
    'features': renderTabFeatures,
    'ensemble': renderTabEnsemble,
    'app-guide': renderTabAppGuide,
});

// ==========================================
// Main Render Function
// ==========================================

export function render(container, _data, _characteristics) {
    const tabsHtml = TABS.map((tab, i) =>
        `<button class="lg-tab ${i === 0 ? 'active' : ''}" data-tab="${tab.id}">
            <i class="fas ${tab.icon}"></i> ${tab.label}
        </button>`
    ).join('');

    const panelsHtml = TABS.map((tab, i) =>
        `<div id="lg-panel-${tab.id}" class="lg-panel ${i === 0 ? 'active' : ''}"></div>`
    ).join('');

    container.innerHTML = `
        ${buildStyles()}
        <div class="lg-container">
            <h2 style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
                <i class="fas fa-graduation-cap" style="color:${THEME.primary};"></i>
                機械学習 学習ガイド
            </h2>
            <p style="color:${THEME.textSecondary};margin-bottom:1.5rem;">
                インタラクティブに機械学習の基礎を学びましょう。各タブをクリックして進めてください。
            </p>
            <div class="lg-tabs">${tabsHtml}</div>
            ${panelsHtml}
        </div>
    `;

    setupTabNavigation(container);

    // Render all tabs (first tab is visible, others are hidden but rendered)
    TABS.forEach(tab => {
        const panelEl = container.querySelector(`#lg-panel-${tab.id}`);
        const renderer = TAB_RENDERERS[tab.id];
        if (panelEl && renderer) {
            renderer(panelEl);
        }
    });
}
