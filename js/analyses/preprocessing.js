// ==========================================
// データ前処理 Module
// ==========================================
import { createSelect, formatNumber, renderPlot } from '../utils.js';

export function render(container, data, characteristics) {
    const numCols = characteristics.numericColumns;
    const catCols = characteristics.categoricalColumns;
    const allCols = characteristics.allColumns || Object.keys(data[0]);

    container.innerHTML = `
        <h2><i class="fas fa-cogs" style="color: #805ad5;"></i> データ前処理</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            機械学習の前にデータを整えます。欠損値補完・スケーリング・エンコーディングの効果を確認できます。
        </p>

        <div class="tab-container">
            <button class="tab-btn active" data-tab="missing">欠損値処理</button>
            <button class="tab-btn" data-tab="scaling">スケーリング</button>
            <button class="tab-btn" data-tab="encoding">エンコーディング</button>
            <button class="tab-btn" data-tab="outliers">外れ値検出</button>
        </div>

        <div id="tab-missing" class="tab-content active">
            ${renderMissingTab(data, allCols)}
        </div>
        <div id="tab-scaling" class="tab-content">
            ${renderScalingTab(data, numCols)}
        </div>
        <div id="tab-encoding" class="tab-content">
            ${renderEncodingTab(data, catCols, numCols)}
        </div>
        <div id="tab-outliers" class="tab-content">
            <div style="margin-bottom: 1rem;">
                <label style="font-weight: 600;">変数を選択:</label>
                ${createSelect('outlier-var-select', numCols)}
            </div>
            <div id="outlier-plot" style="min-height: 400px;"></div>
            <div id="outlier-info"></div>
        </div>
    `;

    // Tab switching
    container.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            container.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            container.querySelector(`#tab-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Outlier detection
    const outlierSelect = container.querySelector('#outlier-var-select');
    if (outlierSelect) {
        outlierSelect.addEventListener('change', () => {
            if (outlierSelect.value) renderOutlierDetection(data, outlierSelect.value);
        });
        if (numCols.length > 0) {
            outlierSelect.value = numCols[0];
            renderOutlierDetection(data, numCols[0]);
        }
    }
}

function renderMissingTab(data, allCols) {
    const n = data.length;
    const missingInfo = allCols.map(col => {
        const missing = data.filter(r => r[col] == null || r[col] === '').length;
        return { col, missing, rate: (missing / n * 100) };
    });

    const totalMissing = missingInfo.reduce((a, b) => a + b.missing, 0);
    const missingCols = missingInfo.filter(m => m.missing > 0);

    if (missingCols.length === 0) {
        return `<div style="text-align: center; padding: 2rem; color: #10b981;">
            <i class="fas fa-check-circle fa-3x" style="margin-bottom: 1rem;"></i>
            <h3>欠損値はありません</h3>
            <p>すべてのセルにデータが入っています。前処理は不要です。</p>
        </div>`;
    }

    return `
        <div class="metrics-grid" style="margin-bottom: 1.5rem;">
            <div class="metric-card">
                <div class="metric-label">総欠損セル数</div>
                <div class="metric-value">${totalMissing}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">欠損のある変数</div>
                <div class="metric-value">${missingCols.length} / ${allCols.length}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">全体欠損率</div>
                <div class="metric-value">${((totalMissing / (n * allCols.length)) * 100).toFixed(1)}%</div>
            </div>
        </div>

        <h4>補完方法の推奨</h4>
        <div class="table-container">
            <table class="table">
                <thead><tr><th>変数</th><th>欠損数</th><th>欠損率</th><th>推奨補完方法</th><th>理由</th></tr></thead>
                <tbody>
                    ${missingCols.map(m => {
                        const isNumeric = !isNaN(Number(data.find(r => r[m.col] != null)?.[m.col]));
                        let method, reason;
                        if (m.rate > 50) {
                            method = '変数の除外';
                            reason = '欠損率50%超で信頼性が低い';
                        } else if (isNumeric) {
                            method = '中央値で補完';
                            reason = '外れ値の影響を受けにくい';
                        } else {
                            method = '最頻値で補完';
                            reason = 'カテゴリ変数の標準的な方法';
                        }
                        return `<tr>
                            <td><strong>${m.col}</strong></td>
                            <td>${m.missing}</td>
                            <td>${m.rate.toFixed(1)}%</td>
                            <td><span style="color: #1e90ff; font-weight: 600;">${method}</span></td>
                            <td style="color: var(--text-secondary);">${reason}</td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 1rem;">
            <i class="fas fa-info-circle"></i> AutoML 機能を使う際、欠損値は自動的に補完されます。
        </p>
    `;
}

function renderScalingTab(data, numCols) {
    if (numCols.length === 0) {
        return '<p style="color: var(--text-secondary);">数値変数がありません。</p>';
    }

    const stats = numCols.map(col => {
        const values = data.map(r => r[col]).filter(v => v != null && !isNaN(Number(v))).map(Number);
        if (values.length === 0) return { col, min: 0, max: 0, mean: 0, std: 0, range: 0 };
        const min = Math.min(...values);
        const max = Math.max(...values);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1));
        return { col, min, max, mean, std, range: max - min };
    });

    const ranges = stats.map(s => s.range);
    const maxRange = Math.max(...ranges);
    const minRange = Math.min(...ranges);
    const needsScaling = maxRange / (minRange || 1) > 10;

    return `
        <div style="background: ${needsScaling ? '#fef3c7' : '#d1fae5'}; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid ${needsScaling ? '#f59e0b' : '#10b981'};">
            <strong>${needsScaling ? '<i class="fas fa-exclamation-triangle"></i> スケーリング推奨' : '<i class="fas fa-check-circle"></i> スケールは概ね均一'}</strong>
            <p style="margin-top: 0.5rem; color: var(--text-secondary);">
                ${needsScaling ? '変数間でスケールに大きな差があります。KNN や SVM などの距離ベースのアルゴリズムではスケーリングが重要です。' : '変数間のスケール差は比較的小さいです。ただし、スケーリングは一般的に推奨されます。'}
            </p>
        </div>

        <h4>各変数のスケール</h4>
        <div class="table-container">
            <table class="table">
                <thead><tr><th>変数</th><th>最小値</th><th>最大値</th><th>範囲</th><th>平均</th><th>標準偏差</th></tr></thead>
                <tbody>
                    ${stats.map(s => `
                        <tr>
                            <td><strong>${s.col}</strong></td>
                            <td>${formatNumber(s.min)}</td>
                            <td>${formatNumber(s.max)}</td>
                            <td>${formatNumber(s.range)}</td>
                            <td>${formatNumber(s.mean)}</td>
                            <td>${formatNumber(s.std)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>

        <h4 style="margin-top: 1.5rem;">スケーリング手法の比較</h4>
        <div class="table-container">
            <table class="table">
                <thead><tr><th>手法</th><th>変換式</th><th>特徴</th><th>推奨場面</th></tr></thead>
                <tbody>
                    <tr><td><strong>StandardScaler</strong></td><td>(x - mean) / std</td><td>平均0、標準偏差1</td><td>線形回帰、SVM、PCA</td></tr>
                    <tr><td><strong>MinMaxScaler</strong></td><td>(x - min) / (max - min)</td><td>[0, 1]に変換</td><td>ニューラルネット、KNN</td></tr>
                </tbody>
            </table>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 1rem;">
            <i class="fas fa-info-circle"></i> AutoML 機能では StandardScaler が自動適用されます。
        </p>
    `;
}

function renderEncodingTab(data, catCols, numCols) {
    const pureCatCols = catCols.filter(c => !numCols.includes(c));

    if (pureCatCols.length === 0 && catCols.length === 0) {
        return '<p style="color: var(--text-secondary);">カテゴリ変数がありません。エンコーディングは不要です。</p>';
    }

    const n = data.length;
    const catInfo = catCols.map(col => {
        const values = data.map(r => r[col]).filter(v => v != null);
        const unique = [...new Set(values)];
        const isNumericCoded = numCols.includes(col);
        return { col, uniqueCount: unique.length, values: unique.slice(0, 5), isNumericCoded };
    });

    return `
        <h4>カテゴリ変数のエンコーディング</h4>
        <div class="table-container">
            <table class="table">
                <thead><tr><th>変数</th><th>ユニーク数</th><th>サンプル値</th><th>現在の型</th><th>推奨エンコーディング</th></tr></thead>
                <tbody>
                    ${catInfo.map(c => {
                        let method;
                        if (c.isNumericCoded) {
                            method = 'そのまま使用可能（数値コード）';
                        } else if (c.uniqueCount === 2) {
                            method = 'Label Encoding (2値)';
                        } else if (c.uniqueCount <= 10) {
                            method = 'Label Encoding';
                        } else {
                            method = 'Label Encoding（高カーディナリティ）';
                        }
                        return `<tr>
                            <td><strong>${c.col}</strong></td>
                            <td>${c.uniqueCount}</td>
                            <td>${c.values.join(', ')}${c.uniqueCount > 5 ? '...' : ''}</td>
                            <td>${c.isNumericCoded ? '数値' : '文字列'}</td>
                            <td><span style="color: #1e90ff; font-weight: 600;">${method}</span></td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 1rem;">
            <i class="fas fa-info-circle"></i> AutoML 機能ではカテゴリ変数は自動的にエンコードされます。
        </p>
    `;
}

function renderOutlierDetection(data, colName) {
    const values = data.map(r => r[colName]).filter(v => v != null && !isNaN(Number(v))).map(Number);
    if (values.length === 0) return;

    const sorted = [...values].sort((a, b) => a - b);
    const n = sorted.length;
    const q1 = sorted[Math.floor(n * 0.25)];
    const q3 = sorted[Math.floor(n * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    const outliers = values.filter(v => v < lowerBound || v > upperBound);

    const plotData = [{
        y: values,
        type: 'box',
        name: colName,
        marker: { color: '#1e90ff' },
        boxpoints: 'outliers'
    }];

    renderPlot('outlier-plot', plotData, {
        title: `${colName} の箱ひげ図と外れ値`,
        yaxis: { title: colName }
    });

    document.getElementById('outlier-info').innerHTML = `
        <div class="metrics-grid" style="margin-top: 1rem;">
            <div class="metric-card">
                <div class="metric-label">Q1 (25%点)</div>
                <div class="metric-value">${formatNumber(q1)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Q3 (75%点)</div>
                <div class="metric-value">${formatNumber(q3)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">IQR</div>
                <div class="metric-value">${formatNumber(iqr)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">外れ値の数</div>
                <div class="metric-value">${outliers.length} (${(outliers.length / n * 100).toFixed(1)}%)</div>
            </div>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 1rem;">
            外れ値の判定基準: IQR法 (Q1 - 1.5*IQR 未満 または Q3 + 1.5*IQR 超)
        </p>
    `;
}
