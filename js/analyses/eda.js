// ==========================================
// 探索的データ分析 (EDA) Module
// ==========================================
import { bindAccessibleTabs, renderPlot, createSelect, escapeHtml, formatNumber } from '../utils.js';
import { buildAnalysisContext, renderAIAssistPanel } from '../ai_assistant.js';

export function render(container, data, characteristics) {
    const numCols = characteristics.numericColumns;
    const catCols = characteristics.categoricalColumns;
    const allCols = characteristics.allColumns || Object.keys(data[0]);

    container.innerHTML = `
        <h2><i class="fas fa-search" style="color: #3182ce;"></i> 探索的データ分析 (EDA)</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            データの分布・相関・欠損値を可視化し、機械学習の前にデータの全体像を把握します。
        </p>

        <div class="eda-tabs">
            <div class="tab-container" role="tablist" aria-label="EDA表示">
                <button id="eda-tab-overview" class="tab-btn active" data-tab="overview" role="tab" aria-controls="tab-overview" aria-selected="true">概要</button>
                <button id="eda-tab-distribution" class="tab-btn" data-tab="distribution" role="tab" aria-controls="tab-distribution" aria-selected="false" tabindex="-1">分布</button>
                <button id="eda-tab-correlation" class="tab-btn" data-tab="correlation" role="tab" aria-controls="tab-correlation" aria-selected="false" tabindex="-1">相関</button>
                <button id="eda-tab-missing" class="tab-btn" data-tab="missing" role="tab" aria-controls="tab-missing" aria-selected="false" tabindex="-1">欠損値</button>
            </div>

            <div id="tab-overview" class="tab-content active" role="tabpanel" aria-labelledby="eda-tab-overview">
                <div id="overview-content"></div>
            </div>
            <div id="tab-distribution" class="tab-content" role="tabpanel" aria-labelledby="eda-tab-distribution" hidden>
                <div style="margin-bottom: 1rem;">
                    <label style="font-weight: 600;">変数を選択:</label>
                    ${createSelect('dist-var-select', numCols)}
                </div>
                <div id="distribution-plot" style="min-height: 400px;"></div>
                <div id="distribution-stats" style="margin-top: 1rem;"></div>
            </div>
            <div id="tab-correlation" class="tab-content" role="tabpanel" aria-labelledby="eda-tab-correlation" hidden>
                <div id="correlation-plot" style="min-height: 500px;"></div>
                <div id="correlation-table" style="margin-top: 1rem;"></div>
            </div>
            <div id="tab-missing" class="tab-content" role="tabpanel" aria-labelledby="eda-tab-missing" hidden>
                <div id="missing-content"></div>
            </div>
        </div>
    `;

    bindAccessibleTabs(container.querySelector('.eda-tabs'));

    renderOverview(data, characteristics);
    renderMissing(data, allCols);
    if (numCols.length >= 2) renderCorrelation(data, numCols);

    const distSelect = container.querySelector('#dist-var-select');
    distSelect.addEventListener('change', () => {
        if (distSelect.value) renderDistribution(data, distSelect.value);
    });
    if (numCols.length > 0) {
        distSelect.value = numCols[0];
        renderDistribution(data, numCols[0]);
    }

    renderAIAssistPanel({
        context: buildAnalysisContext({
            data,
            characteristics,
            method: '探索的データ分析 (EDA)',
            resultSummary: createEDAResultSummary(data, characteristics)
        })
    });
}

function createEDAResultSummary(data, chars) {
    const cols = chars.allColumns || Object.keys(data[0] || {});
    const totalCells = data.length * cols.length;
    const missingCount = cols.reduce((sum, col) => {
        return sum + data.filter(row => row[col] == null || row[col] === '').length;
    }, 0);

    return {
        sampleCount: data.length,
        variableCount: cols.length,
        numericVariableCount: chars.numericColumns.length,
        categoricalVariableCount: chars.categoricalColumns.length,
        textVariableCount: chars.textColumns.length,
        missingCells: missingCount,
        missingRate: totalCells > 0 ? Number((missingCount / totalCells * 100).toFixed(2)) : 0,
        duplicateRows: countDuplicates(data)
    };
}

function renderOverview(data, chars) {
    const container = document.getElementById('overview-content');
    const n = data.length;
    const cols = Object.keys(data[0]);

    let missingCount = 0;
    cols.forEach(col => {
        data.forEach(row => {
            if (row[col] == null || row[col] === '') missingCount++;
        });
    });
    const missingRate = ((missingCount / (n * cols.length)) * 100).toFixed(1);

    container.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">サンプル数</div>
                <div class="metric-value">${n.toLocaleString()}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">特徴量数</div>
                <div class="metric-value">${cols.length}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">数値変数</div>
                <div class="metric-value">${chars.numericColumns.length}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">カテゴリ変数</div>
                <div class="metric-value">${chars.categoricalColumns.length}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">欠損率</div>
                <div class="metric-value">${missingRate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">重複行</div>
                <div class="metric-value">${countDuplicates(data)}</div>
            </div>
        </div>

        <h3 style="margin-top: 2rem;">変数の型</h3>
        <div class="table-container">
            <table class="table">
                <thead><tr><th>変数名</th><th>型</th><th>ユニーク数</th><th>欠損数</th><th>サンプル値</th></tr></thead>
                <tbody>
                    ${cols.map(col => {
                        const values = data.map(r => r[col]).filter(v => v != null && v !== '');
                        const unique = new Set(values).size;
                        const missing = n - values.length;
                        const type = chars.numericColumns.includes(col) ? '数値' :
                                     chars.categoricalColumns.includes(col) ? 'カテゴリ' : 'テキスト';
                        const sample = values.slice(0, 3).join(', ');
                        return `<tr><td><strong data-i18n-ignore>${escapeHtml(col)}</strong></td><td>${type}</td><td>${unique}</td><td>${missing}</td><td data-i18n-ignore style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(sample)}</td></tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderDistribution(data, colName) {
    const values = data.map(r => r[colName])
        .filter(v => v != null && v !== '' && !isNaN(Number(v)))
        .map(Number);
    if (values.length === 0) return;

    const plotData = [{
        x: values,
        type: 'histogram',
        marker: { color: '#1e90ff', line: { color: '#1873cc', width: 1 } },
        opacity: 0.8,
        name: colName
    }];

    renderPlot('distribution-plot', plotData, {
        title: `${colName} の分布`,
        xaxis: { title: colName },
        yaxis: { title: '頻度' },
        bargap: 0.05
    });

    const sorted = [...values].sort((a, b) => a - b);
    const n = values.length;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const std = n > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
    const skewness = std > 0 ? (values.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n) : null;
    const kurtosis = std > 0 ? (values.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / n) - 3 : null;
    const median = quantile(sorted, 0.5);

    document.getElementById('distribution-stats').innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">平均</div><div class="metric-value">${formatNumber(mean)}</div></div>
            <div class="metric-card"><div class="metric-label">標準偏差</div><div class="metric-value">${formatNumber(std)}</div></div>
            <div class="metric-card"><div class="metric-label">中央値</div><div class="metric-value">${formatNumber(median)}</div></div>
            <div class="metric-card"><div class="metric-label">歪度</div><div class="metric-value">${skewness == null ? '-' : formatNumber(skewness)}</div></div>
            <div class="metric-card"><div class="metric-label">尖度</div><div class="metric-value">${kurtosis == null ? '-' : formatNumber(kurtosis)}</div></div>
        </div>
    `;
}

function renderCorrelation(data, numCols) {
    const matrix = [];

    for (let i = 0; i < numCols.length; i++) {
        const row = [];
        for (let j = 0; j < numCols.length; j++) {
            const paired = data
                .map(r => [Number(r[numCols[i]]), Number(r[numCols[j]])])
                .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
            row.push(paired.length >= 2
                ? pearsonCorrelation(paired.map(p => p[0]), paired.map(p => p[1]))
                : 0);
        }
        matrix.push(row);
    }

    const plotData = [{
        z: matrix,
        x: numCols,
        y: numCols,
        type: 'heatmap',
        colorscale: 'RdBu',
        reversescale: true,
        zmin: -1,
        zmax: 1,
        text: matrix.map(row => row.map(v => v.toFixed(2))),
        texttemplate: '%{text}',
        textfont: { size: 10 },
        hoverongaps: false,
        showscale: true
    }];

    renderPlot('correlation-plot', plotData, {
        title: '相関行列',
        height: Math.max(400, numCols.length * 35),
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' }
    });
}

function renderMissing(data, allCols) {
    const container = document.getElementById('missing-content');
    const n = data.length;

    const missingInfo = allCols.map(col => {
        const missing = data.filter(r => r[col] == null || r[col] === '').length;
        return { col, missing, rate: (missing / n * 100) };
    }).sort((a, b) => b.missing - a.missing);

    const hasMissing = missingInfo.some(m => m.missing > 0);

    if (!hasMissing) {
        container.innerHTML = `<div style="text-align: center; padding: 2rem; color: #10b981;">
            <i class="fas fa-check-circle fa-3x" style="margin-bottom: 1rem;"></i>
            <h3>欠損値はありません</h3>
            <p>すべての変数にデータが揃っています。</p>
        </div>`;
        return;
    }

    const missingCols = missingInfo.filter(m => m.missing > 0);

    const barData = [{
        x: missingCols.map(m => m.col),
        y: missingCols.map(m => m.rate),
        type: 'bar',
        marker: { color: missingCols.map(m => m.rate > 50 ? '#ef4444' : m.rate > 20 ? '#f59e0b' : '#1e90ff') },
        text: missingCols.map(m => `${m.missing} (${m.rate.toFixed(1)}%)`),
        textposition: 'auto'
    }];

    container.innerHTML = `
        <div id="missing-plot" style="min-height: 400px;"></div>
        <div class="table-container" style="margin-top: 1rem;">
            <table class="table">
                <thead><tr><th>変数</th><th>欠損数</th><th>欠損率</th><th>状況</th></tr></thead>
                <tbody>
                    ${missingCols.map(m => `
                        <tr>
                            <td><strong data-i18n-ignore>${escapeHtml(m.col)}</strong></td>
                            <td>${m.missing}</td>
                            <td>${m.rate.toFixed(1)}%</td>
                            <td>${m.rate > 50 ? '<span style="color:#ef4444;">要注意</span>' : m.rate > 20 ? '<span style="color:#f59e0b;">注意</span>' : '<span style="color:#10b981;">軽微</span>'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;

    renderPlot('missing-plot', barData, {
        title: '欠損値の状況',
        xaxis: { title: '変数', tickangle: -45 },
        yaxis: { title: '欠損率 (%)', range: [0, 100] }
    });
}

function pearsonCorrelation(x, y) {
    const n = x.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }
    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
}

function quantile(sorted, q) {
    if (sorted.length === 0) return NaN;
    if (sorted.length === 1) return sorted[0];
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    return sorted[base + 1] !== undefined
        ? sorted[base] + rest * (sorted[base + 1] - sorted[base])
        : sorted[base];
}

function countDuplicates(data) {
    const seen = new Set();
    let dupes = 0;
    data.forEach(row => {
        const key = JSON.stringify(row);
        if (seen.has(key)) dupes++;
        else seen.add(key);
    });
    return dupes;
}
