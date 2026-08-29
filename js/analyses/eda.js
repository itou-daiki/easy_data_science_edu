// ==========================================
// 探索的データ分析 (EDA) Module
// ==========================================
import { bindAccessibleTabs, renderPlot, createSelect, createBeginnerGuide, escapeHtml, formatNumber } from '../utils.js';
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

        ${createBeginnerGuide({
            purpose: {
                ja: 'EDAは、予測を始める前に「どんなデータか」「直す点があるか」を調べる観察の段階です。',
                en: 'EDA is the observation stage for learning what the data looks like and what may need attention before prediction.'
            },
            lookFor: {
                ja: '「概要」→「分布」→「相関」→「欠損値」の順に見ます。特に欠損、極端な値、偏り、重複へ注目します。',
                en: 'Review Overview, Distribution, Correlation, then Missing values. Focus on missing data, extreme values, skew, and duplicates.'
            },
            nextAction: {
                ja: '気になる列が見つかったら前処理で対応を確認し、その後に予測したい列の型に合わせて回帰か分類へ進みます。',
                en: 'If a column needs attention, review it under Preprocessing, then choose regression or classification based on the target type.'
            },
            caution: {
                ja: 'EDAで見つけた関係は手がかりです。相関やグラフだけで原因・結果を断定できません。',
                en: 'EDA reveals clues. Correlations and charts alone do not establish cause and effect.'
            },
            terms: [{
                term: { ja: 'EDA', en: 'EDA' },
                meaning: { ja: '探索的データ分析。モデルを作る前に、データの特徴や問題を見つける作業です。', en: 'Exploratory data analysis: examining data characteristics and possible problems before modeling.' }
            }]
        })}

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
    if (numCols.length >= 2) {
        renderCorrelation(data, numCols);
    } else {
        document.getElementById('correlation-plot').style.minHeight = '0';
        document.getElementById('correlation-table').innerHTML = createBeginnerGuide({
            title: { ja: '相関を見るには', en: 'To inspect correlation' },
            purpose: { ja: '2つの数値列が一緒に変化する傾向を調べます。', en: 'Correlation examines whether two numeric columns tend to change together.' },
            lookFor: { ja: '現在は数値列が2つ未満のため、相関を計算できません。', en: 'There are fewer than two numeric columns, so correlation cannot be calculated.' },
            nextAction: { ja: '元データで数値列に文字や単位が混ざっていないか確認します。', en: 'Check whether text or units are mixed into numeric columns in the source data.' }
        });
    }

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
        ${createBeginnerGuide({
            title: { ja: '概要の読み方', en: 'How to read the overview' },
            purpose: { ja: 'データの大きさ、列の型、欠損、重複をまとめて確認します。', en: 'Review dataset size, column types, missing values, and duplicates.' },
            lookFor: {
                ja: `このデータは${n}行${cols.length}列で、欠損率は${missingRate}%、重複行は${countDuplicates(data)}件です。次に列ごとの型が想定どおりか見ます。`,
                en: `This dataset has ${n} rows and ${cols.length} columns, a ${missingRate}% missing rate, and ${countDuplicates(data)} duplicate rows. Next, check each detected column type.`
            },
            nextAction: { ja: '数値列を「分布」で1列ずつ確認します。欠損や重複があれば、その扱いも決めます。', en: 'Inspect numeric columns one at a time under Distribution. Decide how to handle any missing or duplicate observations.' },
            caution: { ja: '行数が多いだけでデータの質が高いとは限りません。集め方や対象の偏りも別に確認します。', en: 'More rows do not automatically mean higher-quality data. Also review how the data was collected and who or what it represents.' }
        })}
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
        ${createBeginnerGuide({
            title: { ja: `${colName}の分布の読み方`, en: `How to read the distribution of ${colName}` },
            purpose: { ja: '値がどこに集まり、どのくらい散らばっているかを確認します。', en: 'See where values cluster and how widely they spread.' },
            lookFor: {
                ja: `平均は${formatNumber(mean)}、中央値は${formatNumber(median)}、歪度は${skewness == null ? '計算不可' : formatNumber(skewness)}です。棒の山、空白、端に離れた値も見ます。`,
                en: `The mean is ${formatNumber(mean)}, the median is ${formatNumber(median)}, and skewness is ${skewness == null ? 'unavailable' : formatNumber(skewness)}. Also inspect peaks, gaps, and isolated values near the ends.`
            },
            nextAction: Math.abs(skewness ?? 0) >= 1
                ? {
                    ja: '偏りが目立つため、前処理の「外れ値検出」で確認し、値の意味や入力ミスを調べます。',
                    en: 'Because the distribution is notably skewed, inspect Outlier detection and check the meaning of extreme values or possible entry errors.'
                }
                : {
                    ja: '他の数値列も選び、目的変数候補の分布と比べます。',
                    en: 'Select other numeric columns and compare them with the distribution of your possible target.'
                },
            caution: { ja: '歪度が大きいこと自体は誤りではありません。所得や価格のように、本来偏るデータもあります。', en: 'A large skewness value is not automatically an error. Variables such as income or price can be naturally skewed.' },
            terms: [{
                term: { ja: '歪度', en: 'Skewness' },
                meaning: { ja: '分布の左右の偏りを表す数値です。0に近いほど左右対称ですが、良し悪しの点数ではありません。', en: 'A measure of left-right asymmetry. Values near zero are more symmetric, but skewness is not a quality score.' }
            }]
        })}
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
    let strongest = { first: '', second: '', value: 0 };

    for (let i = 0; i < numCols.length; i++) {
        const row = [];
        for (let j = 0; j < numCols.length; j++) {
            const paired = data
                .map(r => [Number(r[numCols[i]]), Number(r[numCols[j]])])
                .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
            row.push(paired.length >= 2
                ? pearsonCorrelation(paired.map(p => p[0]), paired.map(p => p[1]))
                : 0);
            if (i < j && (!strongest.first || Math.abs(row[j]) > Math.abs(strongest.value))) {
                strongest = { first: numCols[i], second: numCols[j], value: row[j] };
            }
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

    document.getElementById('correlation-table').innerHTML = createBeginnerGuide({
        title: { ja: '相関行列の読み方', en: 'How to read the correlation matrix' },
        purpose: { ja: '2つの数値列が一緒に増減する傾向を、-1から1の値と色で比べます。', en: 'Compare how pairs of numeric columns move together using values and colors from -1 to 1.' },
        lookFor: {
            ja: `対角線以外で絶対値が最も大きい組は「${strongest.first}」と「${strongest.second}」で、r = ${formatNumber(strongest.value)}です。`,
            en: `The largest absolute off-diagonal correlation is between "${strongest.first}" and "${strongest.second}", with r = ${formatNumber(strongest.value)}.`
        },
        nextAction: { ja: '強い組み合わせは両方の分布と欠損を確認し、予測に使う意味があるかを考えます。', en: 'For a strong pair, inspect both distributions and missing values, then consider whether the relationship is meaningful for prediction.' },
        caution: { ja: '相関は因果関係を示しません。外れ値、別の変数、同じ内容を表す重複列でも大きくなります。', en: 'Correlation does not show causation. It can be inflated by outliers, another variable, or duplicate measures of the same concept.' },
        terms: [{
            term: { ja: '相関係数 r', en: 'Correlation coefficient r' },
            meaning: { ja: '1に近いと同方向、-1に近いと反対方向、0に近いと直線的な関係が弱いことを表します。', en: 'Values near 1 indicate movement in the same direction, near -1 the opposite direction, and near 0 a weak linear relationship.' }
        }]
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
        container.innerHTML = `${createBeginnerGuide({
            title: { ja: '欠損値の読み方', en: 'How to read missing values' },
            purpose: { ja: '記録されていないセルがどの列にどれだけあるかを確認します。', en: 'Check how many cells are unrecorded in each column.' },
            lookFor: { ja: 'このデータでは空欄として検出された値はありません。', en: 'No blank values were detected in this dataset.' },
            nextAction: { ja: '「分布」で極端な値や入力ミスを確認します。0や「不明」が欠損の代わりに使われていないかも確認します。', en: 'Inspect distributions for extreme values or entry errors, and check whether 0 or labels such as “unknown” were used in place of missing values.' },
            caution: { ja: '欠損0件はデータが正しいことの証明ではありません。', en: 'Zero detected missing values does not prove that the data is correct.' }
        })}<div style="text-align: center; padding: 2rem; color: #10b981;">
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
        ${createBeginnerGuide({
            title: { ja: '欠損値の読み方', en: 'How to read missing values' },
            purpose: { ja: '欠損が多い列と、その割合を確認します。', en: 'Identify columns with missing values and compare their rates.' },
            lookFor: {
                ja: `最も欠損が多いのは「${missingCols[0].col}」で、${missingCols[0].missing}件（${missingCols[0].rate.toFixed(1)}%）です。`,
                en: `The most missing values occur in "${missingCols[0].col}": ${missingCols[0].missing} rows (${missingCols[0].rate.toFixed(1)}%).`
            },
            nextAction: { ja: '前処理の「欠損値処理」で推奨方法を確認し、なぜ欠損したのかも元データで調べます。', en: 'Review Missing-value handling under Preprocessing and investigate why the values are missing in the source data.' },
            caution: { ja: '空欄を埋めても、失われた情報が戻るわけではありません。欠損の理由によっては結果が偏ります。', en: 'Filling blanks does not restore lost information. Results can remain biased depending on why values are missing.' }
        })}
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
