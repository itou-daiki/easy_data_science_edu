// ==========================================
// データ前処理 Module
// ==========================================
import { bindAccessibleTabs, createSelect, createBeginnerGuide, escapeHtml, formatNumber, renderPlot } from '../utils.js';
import { buildAnalysisContext, renderAIAssistPanel } from '../ai_assistant.js';

export function render(container, data, characteristics) {
    const numCols = characteristics.numericColumns;
    const catCols = characteristics.categoricalColumns;
    const allCols = characteristics.allColumns || Object.keys(data[0]);

    container.innerHTML = `
        <h2><i class="fas fa-cogs" style="color: #805ad5;"></i> データ前処理</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            機械学習の前にデータを整えます。欠損値補完・スケーリング・エンコーディングの効果を確認できます。
        </p>

        ${createBeginnerGuide({
            purpose: { ja: '前処理は、モデルが比較できる形へデータを整える準備です。この画面では必要性を確認します。', en: 'Preprocessing prepares data in a form models can compare. This view helps you decide what preparation is needed.' },
            lookFor: { ja: '欠損値、数値の単位差、カテゴリ列、外れ値をタブごとに確認します。', en: 'Review missing values, differences in numeric scale, categorical columns, and outliers in each tab.' },
            nextAction: { ja: '理由を確認したら回帰または分類へ進みます。AutoMLでは前処理を各訓練区画の中だけで自動学習します。', en: 'After reviewing the reasons, continue to regression or classification. AutoML learns preprocessing only within each training fold.' },
            caution: { ja: '自動推奨は出発点です。列の意味やデータの集め方を確認して、削除・補完を決めます。', en: 'Automatic recommendations are a starting point. Use column meaning and collection context when deciding whether to remove or impute values.' },
            terms: [{
                term: { ja: '前処理', en: 'Preprocessing' },
                meaning: { ja: '欠損を扱い、文字を数値化し、尺度をそろえるなど、学習前にデータを整えることです。', en: 'Preparing data before training, such as handling missing values, encoding labels, and aligning scales.' }
            }]
        })}

        <div class="tab-container" role="tablist" aria-label="前処理表示">
            <button id="pre-tab-missing" class="tab-btn active" data-tab="missing" role="tab" aria-controls="tab-missing" aria-selected="true">欠損値処理</button>
            <button id="pre-tab-scaling" class="tab-btn" data-tab="scaling" role="tab" aria-controls="tab-scaling" aria-selected="false" tabindex="-1">スケーリング</button>
            <button id="pre-tab-encoding" class="tab-btn" data-tab="encoding" role="tab" aria-controls="tab-encoding" aria-selected="false" tabindex="-1">エンコーディング</button>
            <button id="pre-tab-outliers" class="tab-btn" data-tab="outliers" role="tab" aria-controls="tab-outliers" aria-selected="false" tabindex="-1">外れ値検出</button>
        </div>

        <div id="tab-missing" class="tab-content active" role="tabpanel" aria-labelledby="pre-tab-missing">
            ${renderMissingTab(data, allCols)}
        </div>
        <div id="tab-scaling" class="tab-content" role="tabpanel" aria-labelledby="pre-tab-scaling" hidden>
            ${renderScalingTab(data, numCols)}
        </div>
        <div id="tab-encoding" class="tab-content" role="tabpanel" aria-labelledby="pre-tab-encoding" hidden>
            ${renderEncodingTab(data, catCols, numCols)}
        </div>
        <div id="tab-outliers" class="tab-content" role="tabpanel" aria-labelledby="pre-tab-outliers" hidden>
            <div style="margin-bottom: 1rem;">
                <label style="font-weight: 600;">変数を選択:</label>
                ${createSelect('outlier-var-select', numCols)}
            </div>
            <div id="outlier-plot" style="min-height: 400px;"></div>
            <div id="outlier-info"></div>
        </div>
    `;

    bindAccessibleTabs(container);

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

    renderAIAssistPanel({
        context: buildAnalysisContext({
            data,
            characteristics,
            method: 'データ前処理',
            resultSummary: createPreprocessingResultSummary(data, characteristics)
        })
    });
}

function createPreprocessingResultSummary(data, chars) {
    const allCols = chars.allColumns || Object.keys(data[0] || {});
    const missingInfo = allCols.map(col => {
        const missing = data.filter(row => row[col] == null || row[col] === '').length;
        return { column: col, missing, missingRate: data.length > 0 ? Number((missing / data.length * 100).toFixed(2)) : 0 };
    });
    const totalMissing = missingInfo.reduce((sum, item) => sum + item.missing, 0);
    const numericRanges = chars.numericColumns.map(col => {
        const values = data.map(row => row[col]).filter(value => value != null && Number.isFinite(Number(value))).map(Number);
        if (values.length === 0) return null;
        return {
            column: col,
            min: Math.min(...values),
            max: Math.max(...values),
            range: Math.max(...values) - Math.min(...values)
        };
    }).filter(Boolean);
    const ranges = numericRanges.map(item => item.range).filter(range => range > 0);
    const needsScaling = ranges.length >= 2 ? Math.max(...ranges) / Math.min(...ranges) > 10 : false;

    return {
        missingCells: totalMissing,
        missingColumns: missingInfo.filter(item => item.missing > 0).slice(0, 10),
        numericColumns: chars.numericColumns.length,
        categoricalColumns: chars.categoricalColumns.length,
        scalingRecommendation: needsScaling ? '変数間のスケール差が大きいためスケーリング推奨' : 'スケール差は比較的小さい',
        encodingTargets: chars.categoricalColumns.slice(0, 10)
    };
}

function renderMissingTab(data, allCols) {
    const n = data.length;
    const missingInfo = allCols.map(col => {
        const missing = data.filter(r => r[col] == null || r[col] === '').length;
        return { col, missing, rate: (missing / n * 100) };
    });

    const totalMissing = missingInfo.reduce((a, b) => a + b.missing, 0);
    const missingCols = missingInfo.filter(m => m.missing > 0);
    const highestMissing = [...missingCols].sort((a, b) => b.rate - a.rate)[0];

    if (missingCols.length === 0) {
        return `${createBeginnerGuide({
            title: { ja: '欠損値処理の判断', en: 'Deciding how to handle missing values' },
            purpose: { ja: '空欄を補う必要がある列を探します。', en: 'Find columns that need missing values handled.' },
            lookFor: { ja: 'このデータでは空欄として検出されたセルはありません。', en: 'No blank cells were detected in this dataset.' },
            nextAction: { ja: '次に「スケーリング」を開き、数値列の単位差を確認します。', en: 'Next, open Scaling and review differences in numeric units.' },
            caution: { ja: '0、「不明」、999などが欠損の代わりに入力されていないかは元データで確認します。', en: 'Check the source data for codes such as 0, “unknown,” or 999 that may represent missing values.' }
        })}<div style="text-align: center; padding: 2rem; color: #10b981;">
            <i class="fas fa-check-circle fa-3x" style="margin-bottom: 1rem;"></i>
            <h3>欠損値はありません</h3>
            <p>すべてのセルにデータが入っています。前処理は不要です。</p>
        </div>`;
    }

    return `
        ${createBeginnerGuide({
            title: { ja: '欠損値処理の判断', en: 'Deciding how to handle missing values' },
            purpose: { ja: '空欄の量を確認し、列ごとに補完するか除外するかを考えます。', en: 'Review the amount of missing data and consider whether to impute or exclude each column.' },
            lookFor: {
                ja: `欠損セルは合計${totalMissing}件で、${missingCols.length}列にあります。まず欠損率が最も高い「${highestMissing.col}」を確認します。`,
                en: `There are ${totalMissing} missing cells across ${missingCols.length} columns. Start with "${highestMissing.col}", which has the highest missing rate.`
            },
            nextAction: { ja: '推奨方法と理由を確認し、欠損した理由が分からない場合は元データの作成者に確認します。', en: 'Review the recommended method and reason. If the cause of missingness is unknown, ask whoever created the data.' },
            caution: { ja: '50%という基準だけで自動削除はしません。重要な列なら、追加収集や別の扱いを検討します。', en: 'Do not remove a column automatically based only on the 50% threshold. For an important column, consider collecting more data or another approach.' }
        })}
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
                            <td><strong data-i18n-ignore>${escapeHtml(m.col)}</strong></td>
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
        const std = values.length > 1
            ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1))
            : 0;
        return { col, min, max, mean, std, range: max - min };
    });

    const ranges = stats.map(s => s.range).filter(range => range > 0);
    const maxRange = ranges.length > 0 ? Math.max(...ranges) : 0;
    const minRange = ranges.length > 0 ? Math.min(...ranges) : 0;
    const needsScaling = ranges.length >= 2 && maxRange / minRange > 10;

    return `
        ${createBeginnerGuide({
            title: { ja: 'スケーリングの判断', en: 'Deciding whether to scale' },
            purpose: { ja: '単位の違う数値を、モデルが公平に比べやすい尺度へそろえる必要があるかを確認します。', en: 'Decide whether numeric variables with different units need a comparable scale for modeling.' },
            lookFor: needsScaling
                ? { ja: `最大の範囲は最小の範囲の約${formatNumber(maxRange / (minRange || 1))}倍です。距離や係数を使うモデルでは差の影響を受けます。`, en: `The largest range is about ${formatNumber(maxRange / (minRange || 1))} times the smallest. Models based on distances or coefficients can be affected by this difference.` }
                : { ja: '列どうしの範囲差は大きくありません。各列の単位と標準偏差も確認します。', en: 'The column ranges are not very different. Also review each column’s unit and standard deviation.' },
            nextAction: { ja: 'AutoMLではStandardScalerが自動適用されるため、ここでは必要性と変換後の意味を理解して次へ進みます。', en: 'AutoML applies StandardScaler automatically, so understand why scaling is used and what transformed values mean, then continue.' },
            caution: { ja: 'スケーリングは順位や情報を増やす処理ではありません。決定木系モデルは尺度差の影響をほとんど受けません。', en: 'Scaling does not add information or improve rankings by itself. Tree-based models are mostly unaffected by differences in scale.' },
            terms: [{
                term: { ja: 'StandardScaler', en: 'StandardScaler' },
                meaning: { ja: '平均を0、標準偏差を1にする変換です。元の単位ではなく「平均から何標準偏差か」で表します。', en: 'A transformation to mean 0 and standard deviation 1, expressing values by their distance from the mean in standard deviations.' }
            }]
        })}
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
                            <td><strong data-i18n-ignore>${s.col}</strong></td>
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
        ${createBeginnerGuide({
            title: { ja: 'エンコーディングの判断', en: 'Deciding how to encode categories' },
            purpose: { ja: '文字やカテゴリ番号を、順序を勝手に付けずモデルへ渡せる数値に変えます。', en: 'Convert labels or category codes into numbers without inventing an order between categories.' },
            lookFor: { ja: `${catInfo.length}列がカテゴリとして扱われます。種類数が多い列は、変換後の列数も増えます。`, en: `${catInfo.length} columns are treated as categorical. Columns with many categories will create more encoded features.` },
            nextAction: { ja: 'サンプル値を見て表記ゆれ（例: A / a / A ）を直し、IDのような列は特徴量から外すか検討します。', en: 'Review example values for inconsistent labels (for example A / a / A with a space) and consider excluding identifier-like columns.' },
            caution: { ja: '数値で書かれたカテゴリも量ではありません。例として「1組・2組」の2が1の2倍という意味にはなりません。', en: 'A category written as a number is not necessarily a quantity. For example, class 2 is not twice class 1.' },
            terms: [{
                term: { ja: 'One-Hot Encoding', en: 'One-hot encoding' },
                meaning: { ja: 'カテゴリごとに0/1の列を作る方法です。カテゴリ間に大小関係を付けません。', en: 'Creates a 0/1 column for each category without imposing an order between categories.' }
            }]
        })}
        <h4>カテゴリ変数のエンコーディング</h4>
        <div class="table-container">
            <table class="table">
                <thead><tr><th>変数</th><th>ユニーク数</th><th>サンプル値</th><th>現在の型</th><th>推奨エンコーディング</th></tr></thead>
                <tbody>
                    ${catInfo.map(c => {
                        let method;
                        if (c.isNumericCoded) {
                            method = 'One-Hot Encoding（数値コードをカテゴリ扱い）';
                        } else if (c.uniqueCount === 2) {
                            method = 'One-Hot Encoding (2値)';
                        } else if (c.uniqueCount <= 10) {
                            method = 'One-Hot Encoding';
                        } else {
                            method = 'One-Hot Encoding（高カーディナリティに注意）';
                        }
                        return `<tr>
                            <td><strong data-i18n-ignore>${escapeHtml(c.col)}</strong></td>
                            <td>${c.uniqueCount}</td>
                            <td data-i18n-ignore>${escapeHtml(c.values.join(', '))}${c.uniqueCount > 5 ? '...' : ''}</td>
                            <td>${c.isNumericCoded ? '数値' : '文字列'}</td>
                            <td><span style="color: #1e90ff; font-weight: 600;">${method}</span></td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 1rem;">
            <i class="fas fa-info-circle"></i> AutoML 機能ではカテゴリ変数をfold内の訓練データでOne-Hot Encodingし、未知カテゴリは全0として扱います。
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
        ${createBeginnerGuide({
            title: { ja: `${colName}の外れ値の見方`, en: `How to inspect outliers in ${colName}` },
            purpose: { ja: '他の値から離れた観測を、箱ひげ図とIQR基準で見つけます。', en: 'Use a box plot and the IQR rule to identify observations far from most values.' },
            lookFor: { ja: `IQR基準では${outliers.length}件（${(outliers.length / n * 100).toFixed(1)}%）が外れ値候補です。点の位置と元の値を確認します。`, en: `The IQR rule flags ${outliers.length} observations (${(outliers.length / n * 100).toFixed(1)}%) as possible outliers. Review their positions and original values.` },
            nextAction: outliers.length > 0
                ? { ja: '入力ミスか、本当に珍しい事例かを元データで調べます。理由なしに削除せず、残した場合と処理した場合を比較します。', en: 'Check the source data to determine whether these are entry errors or genuinely rare cases. Do not remove them without a reason; compare results with and without treatment.' }
                : { ja: '他の数値列も選び、目的変数と主要な特徴量を確認します。', en: 'Select other numeric columns and inspect the target and important features.' },
            caution: { ja: 'IQRの外側にある値は「誤り」ではなく候補です。分野の知識と記録方法で判断します。', en: 'A value outside the IQR fence is a candidate, not proof of an error. Decide using domain knowledge and collection context.' },
            terms: [{
                term: { ja: 'IQR', en: 'IQR' },
                meaning: { ja: 'Q3からQ1を引いた値で、中央50%の広がりを表します。', en: 'Q3 minus Q1, describing the spread of the middle 50% of observations.' }
            }]
        })}
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
