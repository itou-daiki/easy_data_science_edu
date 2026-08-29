// ==========================================
// UI Helpers for easyDataScience
// ==========================================
import { tr } from './i18n.js';

/**
 * Escape untrusted text before inserting it into an HTML template.
 * @param {*} value
 * @returns {string}
 */
export function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

function createBilingualText(value) {
    const japanese = typeof value === 'object' && value !== null ? value.ja : value;
    const english = typeof value === 'object' && value !== null ? value.en : value;
    return `<span data-i18n-en="${escapeHtml(english ?? japanese ?? '')}">${escapeHtml(japanese ?? '')}</span>`;
}

/**
 * Creates a compact learning scaffold that tells beginners what a view shows,
 * where to look, and what to do next.
 * @param {{
 *   title?: string|{ja: string, en: string},
 *   purpose: string|{ja: string, en: string},
 *   lookFor: string|{ja: string, en: string},
 *   nextAction: string|{ja: string, en: string},
 *   caution?: string|{ja: string, en: string},
 *   terms?: Array<{term: string|{ja: string, en: string}, meaning: string|{ja: string, en: string}}>
 * }} guide
 * @returns {string}
 */
export function createBeginnerGuide({
    title = { ja: '迷わないための見方', en: 'How to read this view' },
    purpose,
    lookFor,
    nextAction,
    caution = '',
    terms = []
}) {
    const steps = [
        {
            label: { ja: 'ここで分かること', en: 'What this shows' },
            text: purpose
        },
        {
            label: { ja: 'まず見る場所', en: 'Where to look first' },
            text: lookFor
        },
        {
            label: { ja: '次にすること', en: 'What to do next' },
            text: nextAction
        }
    ];
    const termList = Array.isArray(terms) ? terms.filter(item => item?.term && item?.meaning) : [];

    return `
        <aside class="beginner-guide">
            <div class="beginner-guide-heading">
                <i class="fas fa-compass" aria-hidden="true"></i>
                <strong>${createBilingualText(title)}</strong>
            </div>
            <div class="beginner-guide-steps">
                ${steps.map((step, index) => `
                    <div class="beginner-guide-step">
                        <span class="beginner-guide-number" aria-hidden="true">${index + 1}</span>
                        <div>
                            <strong>${createBilingualText(step.label)}</strong>
                            <p>${createBilingualText(step.text)}</p>
                        </div>
                    </div>
                `).join('')}
            </div>
            ${caution ? `
                <p class="beginner-guide-caution">
                    <i class="fas fa-triangle-exclamation" aria-hidden="true"></i>
                    ${createBilingualText(caution)}
                </p>
            ` : ''}
            ${termList.length > 0 ? `
                <details class="beginner-guide-terms">
                    <summary>${createBilingualText({ ja: 'ことばの意味', en: 'Key terms' })}</summary>
                    <dl>
                        ${termList.map(item => `
                            <div>
                                <dt>${createBilingualText(item.term)}</dt>
                                <dd>${createBilingualText(item.meaning)}</dd>
                            </div>
                        `).join('')}
                    </dl>
                </details>
            ` : ''}
        </aside>
    `;
}

/**
 * Bind an ARIA tab set, including arrow, Home, and End key navigation.
 * Tabs and panels are paired through aria-controls.
 * @param {HTMLElement} root
 * @param {{onActivate?: ((tab: HTMLElement) => void) | null}} options
 */
export function bindAccessibleTabs(root, { onActivate = null } = {}) {
    const tabs = Array.from(root.querySelectorAll('[role="tab"]'));
    if (tabs.length === 0) return;

    const activate = (tab, { focus = false } = {}) => {
        tabs.forEach(candidate => {
            const selected = candidate === tab;
            candidate.classList.toggle('active', selected);
            candidate.setAttribute('aria-selected', String(selected));
            candidate.tabIndex = selected ? 0 : -1;
            const panelId = candidate.getAttribute('aria-controls');
            const panel = panelId ? root.querySelector(`#${CSS.escape(panelId)}`) : null;
            if (panel) {
                panel.classList.toggle('active', selected);
                panel.hidden = !selected;
            }
        });
        if (focus) tab.focus();
        if (onActivate) onActivate(tab);
    };

    tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => activate(tab));
        tab.addEventListener('keydown', event => {
            let nextIndex = null;
            if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
            if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
            if (event.key === 'Home') nextIndex = 0;
            if (event.key === 'End') nextIndex = tabs.length - 1;
            if (nextIndex == null) return;
            event.preventDefault();
            activate(tabs[nextIndex], { focus: true });
        });
    });

    activate(tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
}

/**
 * Toggles the visibility of a collapsible section.
 * @param {HTMLElement} header
 */
export function toggleCollapsible(header) {
    const content = header.nextElementSibling;
    const willOpen = header.classList.contains('collapsed');
    header.classList.toggle('collapsed', !willOpen);
    content.classList.toggle('collapsed', !willOpen);
    header.setAttribute('aria-expanded', String(willOpen));
    content.hidden = !willOpen;
}

/**
 * Displays a loading message in the upload area.
 * @param {string} message
 */
export function showLoadingMessage(message) {
    const uploadText = document.querySelector('.upload-text');
    if (uploadText) {
        uploadText.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${message}`;
    }
}

/**
 * Hides the loading message in the upload area.
 */
export function hideLoadingMessage() {
    const uploadText = document.querySelector('.upload-text');
    if (uploadText) {
        uploadText.textContent = tr('ここにファイルをドラッグ＆ドロップ');
    }
}

/**
 * Shows an error message using a simple alert.
 * @param {string} message
 */
export function showError(message) {
    alert(`${tr('エラー:')} ${tr(message)}`);
    hideLoadingMessage();
}

/**
 * Creates and returns an HTML table from data.
 * @param {string[]} headers
 * @param {string[]} rowLabels
 * @param {Array<Array<number|string>>} data
 * @returns {string}
 */
export function toHtmlTable(headers, rowLabels, data) {
    let table = '<table class="table"><thead><tr><th></th>';
    headers.forEach(h => table += `<th>${escapeHtml(h)}</th>`);
    table += '</tr></thead><tbody>';
    rowLabels.forEach((r, i) => {
        table += `<tr><th>${escapeHtml(r)}</th>`;
        data[i].forEach(d => table += `<td>${escapeHtml(typeof d === 'number' ? d.toFixed(4) : d)}</td>`);
        table += '</tr>';
    });
    table += '</tbody></table>';
    return table;
}

/**
 * Renders a data preview table.
 * @param {string} containerId
 * @param {Object[]} data
 * @param {string} title
 * @param {number} maxRows
 */
export function renderDataPreview(containerId, data, title = 'データプレビュー', maxRows = 10) {
    const container = document.getElementById(containerId);
    if (!container || !data || data.length === 0) return;

    const columns = Object.keys(data[0]);
    const displayData = data.slice(0, maxRows);

    let html = createBeginnerGuide({
        title: { ja: `${title}の見方`, en: 'How to read the data preview' },
        purpose: {
            ja: '1行が1件、1列が1つの項目になっているかを確かめる画面です。ここに見えるのは先頭部分だけです。',
            en: 'Check that each row is one observation and each column is one item. Only the first rows are shown here.'
        },
        lookFor: {
            ja: '列名、値の単位、空欄（N/A）、数値列に混ざった文字を見ます。個人を特定できる情報がないかも確認します。',
            en: 'Check column names, units, blanks (N/A), text mixed into numeric columns, and any information that could identify a person.'
        },
        nextAction: {
            ja: '表の形が想定どおりなら要約統計量を開き、その後にEDAで欠損・分布・外れ値を確認します。',
            en: 'If the table looks as expected, open Summary statistics, then use EDA to check missing values, distributions, and outliers.'
        },
        terms: [
            {
                term: { ja: '行', en: 'Row' },
                meaning: { ja: '生徒1人、商品1個など、1件分の記録です。', en: 'One observation, such as one student or one product.' }
            },
            {
                term: { ja: '列', en: 'Column' },
                meaning: { ja: '点数、価格、種類など、記録する項目です。', en: 'One recorded item, such as a score, price, or category.' }
            }
        ]
    });
    html += `<div class="table-container"><table class="table">`;
    html += '<thead data-i18n-ignore><tr>';
    html += '<th>#</th>';
    columns.forEach(col => html += `<th>${escapeHtml(col)}</th>`);
    html += '</tr></thead><tbody data-i18n-ignore>';

    displayData.forEach((row, i) => {
        html += `<tr><td>${i + 1}</td>`;
        columns.forEach(col => {
            const val = row[col];
            html += `<td>${val != null && val !== '' ? escapeHtml(val) : '<span style="color:#64748b;">N/A</span>'}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    if (data.length > maxRows) {
        html += `<p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;"><span data-i18n-en="Showing the first ${maxRows} rows (${data.length} rows total)">先頭 ${maxRows} 行を表示（全 ${data.length} 行）</span></p>`;
    }
    container.innerHTML = html;
}

/**
 * Renders summary statistics for the dataset.
 * @param {string} containerId
 * @param {Object[]} data
 * @param {Object} characteristics
 * @param {string} title
 */
export function renderSummaryStatistics(containerId, data, characteristics, title = '要約統計量') {
    const container = document.getElementById(containerId);
    if (!container || !data || !characteristics) return;

    const numCols = characteristics.numericColumns;
    if (numCols.length === 0) {
        container.innerHTML = createBeginnerGuide({
            title: { ja: `${title}の見方`, en: 'How to read summary statistics' },
            purpose: { ja: '数値の列を短くまとめる画面です。', en: 'This view gives a compact summary of numeric columns.' },
            lookFor: { ja: 'このデータには数値として認識された列がありません。', en: 'No columns in this dataset were recognized as numeric.' },
            nextAction: { ja: '数値を分析したい場合は、単位や文字が混ざっていないか元データを確認します。', en: 'To analyze numeric values, check the source data for units or text mixed into the values.' }
        });
        return;
    }

    const stats = numCols.map(col => {
        const values = data.map(row => row[col]).filter(v => v != null && Number.isFinite(Number(v))).map(Number);
        if (values.length === 0) return { col, count: 0, mean: '-', std: '-', min: '-', q1: '-', median: '-', q3: '-', max: '-', missing: data.length };
        const sorted = [...values].sort((a, b) => a - b);
        const n = values.length;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const variance = n > 1
            ? values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
            : 0;
        const std = Math.sqrt(variance);
        const q1 = sorted[Math.floor(n * 0.25)];
        const median = sorted[Math.floor(n * 0.5)];
        const q3 = sorted[Math.floor(n * 0.75)];
        const missing = data.length - n;
        return { col, count: n, mean, std, min: sorted[0], q1, median, q3, max: sorted[n - 1], missing };
    });

    let html = createBeginnerGuide({
        title: { ja: `${title}の見方`, en: 'How to read summary statistics' },
        purpose: {
            ja: '各数値列の中心、ばらつき、範囲、欠損を1つの表で比べる画面です。',
            en: 'Compare the center, spread, range, and missing values of each numeric column in one table.'
        },
        lookFor: {
            ja: '平均と中央値が大きく違う列、標準偏差や最小・最大の幅が大きい列、欠損がある列に注目します。',
            en: 'Look for columns where mean and median differ, spread or range is large, or missing values are present.'
        },
        nextAction: {
            ja: '気になる列名を覚えてEDAの「分布」と「欠損値」で形を確かめます。数字だけで異常と決めつけないでください。',
            en: 'Note any concerning columns and inspect them under Distribution and Missing values in EDA. Do not label a value abnormal from this table alone.'
        },
        terms: [
            {
                term: { ja: '中央値', en: 'Median' },
                meaning: { ja: '小さい順に並べた中央の値です。極端な値の影響を平均より受けにくい指標です。', en: 'The middle value after sorting. It is less affected by extreme values than the mean.' }
            },
            {
                term: { ja: '標準偏差', en: 'Standard deviation' },
                meaning: { ja: '値の散らばり方の目安です。大きいほど平均から広く散らばっています。', en: 'A measure of spread. Larger values indicate observations are more widely dispersed around the mean.' }
            },
            {
                term: { ja: 'Q1 / Q3', en: 'Q1 / Q3' },
                meaning: { ja: '小さい側から25%点と75%点の値で、中央50%の範囲を見るために使います。', en: 'The 25th and 75th percentiles, used to describe the middle 50% of values.' }
            }
        ]
    });
    html += '<div class="table-container"><table class="table">';
    html += '<thead><tr><th>変数</th><th>件数</th><th>平均</th><th>標準偏差</th><th>最小</th><th>Q1</th><th>中央値</th><th>Q3</th><th>最大</th><th>欠損</th></tr></thead><tbody>';
    stats.forEach(s => {
        const fmt = v => typeof v === 'number' ? v.toFixed(3) : v;
        html += `<tr><td><strong data-i18n-ignore>${escapeHtml(s.col)}</strong></td><td>${s.count}</td><td>${fmt(s.mean)}</td><td>${fmt(s.std)}</td><td>${fmt(s.min)}</td><td>${fmt(s.q1)}</td><td>${fmt(s.median)}</td><td>${fmt(s.q3)}</td><td>${fmt(s.max)}</td><td>${s.missing}</td></tr>`;
    });
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

/**
 * Creates a select element with options.
 * @param {string} id
 * @param {string[]} options
 * @param {string} placeholder
 * @returns {string}
 */
export function createSelect(id, options, placeholder = '選択してください') {
    let html = `<select id="${escapeHtml(id)}" class="form-select">`;
    html += `<option value="">${escapeHtml(placeholder)}</option>`;
    options.forEach(opt => html += `<option value="${escapeHtml(opt)}" data-i18n-ignore>${escapeHtml(opt)}</option>`);
    html += '</select>';
    return html;
}

/**
 * Creates a multi-select checkbox group for variable selection.
 * @param {string} name
 * @param {string[]} options
 * @param {string[]} selected
 * @returns {string}
 */
export function createVariableChips(name, options, selected = []) {
    let html = '<div class="variable-chips">';
    options.forEach(opt => {
        const isSelected = selected.includes(opt);
        html += `<label class="variable-chip ${isSelected ? 'selected' : ''}" data-name="${escapeHtml(name)}" data-value="${escapeHtml(opt)}">
            <input type="checkbox" name="${escapeHtml(name)}" value="${escapeHtml(opt)}" ${isSelected ? 'checked' : ''} style="display:none;">
            <span data-i18n-ignore>${escapeHtml(opt)}</span>
        </label>`;
    });
    html += '</div>';
    return html;
}

/**
 * Initializes variable chip click handlers within a container.
 * @param {HTMLElement} container
 */
export function initVariableChips(container) {
    container.querySelectorAll('.variable-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const checkbox = chip.querySelector('input[type="checkbox"]');
            checkbox.checked = !checkbox.checked;
            chip.classList.toggle('selected', checkbox.checked);
        });
    });
}

/**
 * Gets selected values from variable chips.
 * @param {HTMLElement} container
 * @param {string} name
 * @returns {string[]}
 */
export function getSelectedChips(container, name) {
    const checkboxes = container.querySelectorAll(`input[name="${name}"]:checked`);
    return Array.from(checkboxes).map(cb => cb.value);
}

/**
 * Creates a step indicator for the ML workflow.
 * @param {string[]} steps
 * @param {number} activeIndex
 * @returns {string}
 */
export function createStepIndicator(steps, activeIndex = 0) {
    let html = '<div class="step-indicator">';
    steps.forEach((step, i) => {
        const state = i < activeIndex ? 'completed' : i === activeIndex ? 'active' : '';
        html += `<div class="step ${state}">
            <div class="step-number">${i < activeIndex ? '<i class="fas fa-check"></i>' : i + 1}</div>
            <div class="step-label">${step}</div>
        </div>`;
        if (i < steps.length - 1) {
            html += '<div class="step-connector"></div>';
        }
    });
    html += '</div>';
    return html;
}

/**
 * Formats a number for display.
 * @param {number} value
 * @param {number} decimals
 * @returns {string}
 */
export function formatNumber(value, decimals = 4) {
    if (value == null || isNaN(value)) return '-';
    if (Math.abs(value) < 0.0001 && value !== 0) return value.toExponential(2);
    return value.toFixed(decimals);
}

/**
 * Creates a metric card HTML.
 * @param {string} label
 * @param {number} value
 * @param {string} description
 * @param {boolean} higherIsBetter
 * @returns {string}
 */
export function createMetricCard(label, value, description = '', higherIsBetter = true) {
    const formattedValue = Number.isInteger(value) ? value.toLocaleString() : formatNumber(value);
    return `<div class="metric-card">
        <div class="metric-label">${label}</div>
        <div class="metric-value">${formattedValue}</div>
        ${description ? `<div class="metric-description">${description}</div>` : ''}
    </div>`;
}

/**
 * Creates a progress bar HTML.
 * @param {number} progress - 0 to 100
 * @param {string} label
 * @returns {string}
 */
export function createProgressBar(progress, label = '') {
    return `<div class="progress-container">
        ${label ? `<div class="progress-label">${label}</div>` : ''}
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${progress}%"></div>
        </div>
        <div class="progress-text">${Math.round(progress)}%</div>
    </div>`;
}

/**
 * Renders a Plotly chart safely.
 * @param {string} containerId
 * @param {Object[]} data
 * @param {Object} layout
 * @param {Object} config
 */
export function renderPlot(containerId, data, layout = {}, config = {}) {
    const defaultLayout = {
        font: { family: 'Inter, sans-serif' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 40, r: 20, b: 50, l: 60 },
        ...layout
    };
    const defaultConfig = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        ...config
    };
    Plotly.newPlot(containerId, data, defaultLayout, defaultConfig);
}

/**
 * Renders a confusion matrix as a Plotly heatmap.
 * @param {string} containerId
 * @param {number[][]} matrix
 * @param {string[]} labels
 */
export function renderConfusionMatrix(containerId, matrix, labels) {
    const data = [{
        z: matrix,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: true,
        text: matrix.map(row => row.map(v => v.toString())),
        texttemplate: '%{text}',
        textfont: { size: 14 },
        hoverongaps: false
    }];
    const layout = {
        title: '混同行列',
        xaxis: { title: '予測値', side: 'bottom' },
        yaxis: { title: '実測値', autorange: 'reversed' },
        autosize: true,
        height: 400
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders feature importance as a horizontal bar chart.
 * @param {string} containerId
 * @param {string[]} featureNames
 * @param {number[]} importances
 */
export function renderFeatureImportance(containerId, featureNames, importances) {
    const indices = importances.map((v, i) => i).sort((a, b) => importances[a] - importances[b]);
    const sortedNames = indices.map(i => featureNames[i]);
    const sortedValues = indices.map(i => importances[i]);

    const data = [{
        type: 'bar',
        x: sortedValues,
        y: sortedNames,
        orientation: 'h',
        marker: { color: '#1e90ff' }
    }];
    const layout = {
        title: '特徴量重要度',
        xaxis: { title: '重要度' },
        margin: { l: 150 },
        height: Math.max(300, sortedNames.length * 25)
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders actual vs predicted scatter plot.
 * @param {string} containerId
 * @param {number[]} yTrue
 * @param {number[]} yPred
 */
export function renderActualVsPredicted(containerId, yTrue, yPred) {
    const minVal = Math.min(...yTrue, ...yPred);
    const maxVal = Math.max(...yTrue, ...yPred);
    const data = [
        {
            x: yTrue,
            y: yPred,
            mode: 'markers',
            type: 'scatter',
            name: 'データ点',
            marker: { color: '#1e90ff', size: 6, opacity: 0.6 }
        },
        {
            x: [minVal, maxVal],
            y: [minVal, maxVal],
            mode: 'lines',
            name: '理想線 (y=x)',
            line: { color: '#ef4444', dash: 'dash', width: 2 }
        }
    ];
    const layout = {
        title: '実測値 vs 予測値',
        xaxis: { title: '実測値' },
        yaxis: { title: '予測値' }
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders residual plot.
 * @param {string} containerId
 * @param {number[]} yTrue
 * @param {number[]} yPred
 */
export function renderResidualPlot(containerId, yTrue, yPred) {
    const residuals = yTrue.map((v, i) => v - yPred[i]);
    const data = [
        {
            x: yPred,
            y: residuals,
            mode: 'markers',
            type: 'scatter',
            name: '残差',
            marker: { color: '#1e90ff', size: 6, opacity: 0.6 }
        },
        {
            x: [Math.min(...yPred), Math.max(...yPred)],
            y: [0, 0],
            mode: 'lines',
            name: 'ゼロライン',
            line: { color: '#ef4444', dash: 'dash', width: 2 }
        }
    ];
    const layout = {
        title: '残差プロット',
        xaxis: { title: '予測値' },
        yaxis: { title: '残差' }
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders ROC curve.
 * @param {string} containerId
 * @param {number[]} yTrue - binary labels (0/1)
 * @param {number[]} yProba - predicted probabilities for positive class
 * @param {number} auc
 */
/**
 * Renders permutation importance as horizontal bar chart with error bars.
 * @param {string} containerId
 * @param {string[]} featureNames
 * @param {number[]} importancesMean
 * @param {number[]} importancesStd
 */
export function renderPermutationImportance(containerId, featureNames, importancesMean, importancesStd) {
    const indices = importancesMean.map((v, i) => i).sort((a, b) => importancesMean[a] - importancesMean[b]);
    const sortedNames = indices.map(i => featureNames[i]);
    const sortedMean = indices.map(i => importancesMean[i]);
    const sortedStd = indices.map(i => importancesStd[i]);

    const data = [{
        type: 'bar',
        x: sortedMean,
        y: sortedNames,
        orientation: 'h',
        marker: { color: sortedMean.map(v => v > 0 ? '#10b981' : '#ef4444') },
        error_x: {
            type: 'data',
            array: sortedStd,
            visible: true,
            color: '#94a3b8'
        }
    }];
    const layout = {
        title: 'Permutation Feature Importance',
        xaxis: { title: 'スコア低下量（大きいほど重要）' },
        margin: { l: 150 },
        height: Math.max(300, sortedNames.length * 30)
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders a Partial Dependence Plot for a single feature.
 * @param {string} containerId
 * @param {string} featureName
 * @param {number[]} xValues - Feature values
 * @param {number[]} pdpValues - Mean predictions at each feature value
 * @param {string} [color='#3b82f6']
 */
export function renderPDP(containerId, featureName, xValues, pdpValues, color = '#3b82f6') {
    const data = [{
        x: xValues,
        y: pdpValues,
        mode: 'lines+markers',
        type: 'scatter',
        name: featureName,
        line: { color, width: 2 },
        marker: { size: 4 }
    }];
    const layout = {
        title: `Partial Dependence: ${featureName}`,
        xaxis: { title: featureName },
        yaxis: { title: '予測値の変化' },
        height: 350
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders a learning curve showing train and validation scores.
 * @param {string} containerId
 * @param {number[]} trainSizes
 * @param {number[]} trainScoresMean
 * @param {number[]} trainScoresStd
 * @param {number[]} testScoresMean
 * @param {number[]} testScoresStd
 * @param {string} [scoreName='R²']
 */
export function renderLearningCurve(containerId, trainSizes, trainScoresMean, trainScoresStd, testScoresMean, testScoresStd, scoreName = 'R²') {
    const data = [
        {
            x: trainSizes,
            y: trainScoresMean,
            mode: 'lines+markers',
            name: '訓練スコア',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 6 }
        },
        {
            x: trainSizes,
            y: trainScoresMean.map((v, i) => v + trainScoresStd[i]),
            mode: 'lines',
            name: '訓練スコア +1σ',
            line: { color: '#3b82f6', width: 0 },
            showlegend: false
        },
        {
            x: trainSizes,
            y: trainScoresMean.map((v, i) => v - trainScoresStd[i]),
            mode: 'lines',
            name: '訓練スコア -1σ',
            line: { color: '#3b82f6', width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(59,130,246,0.15)',
            showlegend: false
        },
        {
            x: trainSizes,
            y: testScoresMean,
            mode: 'lines+markers',
            name: '検証スコア',
            line: { color: '#ef4444', width: 2 },
            marker: { size: 6 }
        },
        {
            x: trainSizes,
            y: testScoresMean.map((v, i) => v + testScoresStd[i]),
            mode: 'lines',
            name: '検証スコア +1σ',
            line: { color: '#ef4444', width: 0 },
            showlegend: false
        },
        {
            x: trainSizes,
            y: testScoresMean.map((v, i) => v - testScoresStd[i]),
            mode: 'lines',
            name: '検証スコア -1σ',
            line: { color: '#ef4444', width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(239,68,68,0.15)',
            showlegend: false
        }
    ];
    const layout = {
        title: 'Learning Curve（学習曲線）',
        xaxis: { title: '訓練サンプル数' },
        yaxis: { title: scoreName },
        height: 400,
        legend: { x: 0.6, y: 0.1 }
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders SHAP summary bar plot (mean |SHAP| per feature).
 * @param {string} containerId
 * @param {string[]} featureNames
 * @param {number[]} meanAbsSHAP
 * @param {number[]} meanSHAP - signed mean SHAP (for color direction)
 */
export function renderSHAPSummary(containerId, featureNames, meanAbsSHAP, meanSHAP) {
    const indices = meanAbsSHAP.map((v, i) => i).sort((a, b) => meanAbsSHAP[a] - meanAbsSHAP[b]);
    const sortedNames = indices.map(i => featureNames[i]);
    const sortedAbs = indices.map(i => meanAbsSHAP[i]);
    const sortedSigned = indices.map(i => meanSHAP[i]);

    const data = [{
        type: 'bar',
        x: sortedAbs,
        y: sortedNames,
        orientation: 'h',
        marker: { color: sortedSigned.map(v => v > 0 ? '#ef4444' : '#3b82f6') },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
    }];
    const layout = {
        title: 'SHAP Feature Importance (mean |SHAP|)',
        xaxis: { title: 'mean |SHAP value|' },
        margin: { l: 150 },
        height: Math.max(300, sortedNames.length * 30)
    };
    renderPlot(containerId, data, layout);
}

/**
 * Renders SHAP beeswarm plot (individual SHAP values colored by feature value).
 * @param {string} containerId
 * @param {string[]} featureNames
 * @param {number[][]} shapValues - (nSamples x nFeatures)
 * @param {number[][]} featureValues - (nSamples x nFeatures)
 */
export function renderSHAPBeeswarm(containerId, featureNames, shapValues, featureValues) {
    const nFeatures = featureNames.length;
    // Sort features by mean |SHAP|
    const meanAbs = Array(nFeatures).fill(0);
    for (const row of shapValues) {
        for (let f = 0; f < nFeatures; f++) meanAbs[f] += Math.abs(row[f]);
    }
    meanAbs.forEach((v, i, a) => a[i] = v / shapValues.length);
    const sortedIdx = meanAbs.map((v, i) => i).sort((a, b) => meanAbs[a] - meanAbs[b]);

    const traces = [];
    for (let rank = 0; rank < sortedIdx.length; rank++) {
        const f = sortedIdx[rank];
        const svs = shapValues.map(row => row[f]);
        const fvs = featureValues.map(row => row[f]);

        // Normalize feature values to [0,1] for coloring
        const fMin = Math.min(...fvs);
        const fMax = Math.max(...fvs);
        const fRange = fMax - fMin || 1;
        const normalized = fvs.map(v => (v - fMin) / fRange);

        // Add jitter for y-axis
        const yJitter = svs.map(() => rank + (Math.random() - 0.5) * 0.3);

        traces.push({
            x: svs,
            y: yJitter,
            mode: 'markers',
            type: 'scatter',
            name: featureNames[f],
            marker: {
                size: 5,
                color: normalized,
                colorscale: [[0, '#3b82f6'], [1, '#ef4444']],
                opacity: 0.7,
                showscale: rank === sortedIdx.length - 1,
                colorbar: rank === sortedIdx.length - 1 ? {
                    title: '特徴量値',
                    titleside: 'right',
                    tickvals: [0, 1],
                    ticktext: ['低', '高']
                } : undefined
            },
            showlegend: false,
            hovertemplate: `${featureNames[f]}<br>SHAP: %{x:.4f}<br>値: %{text}<extra></extra>`,
            text: fvs.map(v => v.toFixed(2))
        });
    }

    const layout = {
        title: 'SHAP Beeswarm Plot',
        xaxis: { title: 'SHAP value', zeroline: true, zerolinecolor: '#94a3b8' },
        yaxis: {
            tickvals: sortedIdx.map((_, i) => i),
            ticktext: sortedIdx.map(i => featureNames[i]),
            automargin: true
        },
        margin: { l: 150 },
        height: Math.max(350, nFeatures * 40),
        hovermode: 'closest'
    };
    renderPlot(containerId, traces, layout);
}

/**
 * Renders SHAP waterfall plot for a single prediction.
 * @param {string} containerId
 * @param {string[]} featureNames
 * @param {number[]} shapValues - SHAP values for one instance
 * @param {number} baseValue - Expected model output
 * @param {number} prediction - Actual prediction for this instance
 */
export function renderSHAPWaterfall(containerId, featureNames, shapValues, baseValue, prediction) {
    // Sort by absolute SHAP value (largest first)
    const indices = shapValues.map((v, i) => i).sort((a, b) => Math.abs(shapValues[b]) - Math.abs(shapValues[a]));

    const labels = ['E[f(x)]', ...indices.map(i => featureNames[i]), 'f(x)'];
    const measures = ['absolute', ...indices.map(() => 'relative'), 'total'];
    const values = [baseValue, ...indices.map(i => shapValues[i]), prediction];
    const colors = ['#94a3b8', ...indices.map(i => shapValues[i] > 0 ? '#ef4444' : '#3b82f6'), '#10b981'];

    const data = [{
        type: 'waterfall',
        orientation: 'v',
        x: labels,
        y: values,
        measure: measures,
        connector: { line: { color: '#cbd5e1', width: 1 } },
        increasing: { marker: { color: '#ef4444' } },
        decreasing: { marker: { color: '#3b82f6' } },
        totals: { marker: { color: '#10b981' } },
        texttemplate: '%{y:.2f}',
        textposition: 'outside',
        hovertemplate: '%{x}<br>%{y:.4f}<extra></extra>'
    }];

    const layout = {
        title: 'SHAP Waterfall (個別予測の説明)',
        yaxis: { title: '予測値' },
        height: 400,
        margin: { t: 50, b: 80 },
        showlegend: false
    };
    renderPlot(containerId, data, layout);
}

export function renderROCCurve(containerId, yTrue, yProba, auc) {
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const points = thresholds.map(t => {
        let tp = 0, fp = 0, fn = 0, tn = 0;
        yTrue.forEach((y, i) => {
            const pred = yProba[i] >= t ? 1 : 0;
            if (y === 1 && pred === 1) tp++;
            if (y === 0 && pred === 1) fp++;
            if (y === 1 && pred === 0) fn++;
            if (y === 0 && pred === 0) tn++;
        });
        const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
        const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
        return { fpr, tpr };
    });

    const data = [
        {
            x: points.map(p => p.fpr),
            y: points.map(p => p.tpr),
            mode: 'lines',
            name: `ROC曲線 (AUC = ${auc.toFixed(3)})`,
            line: { color: '#1e90ff', width: 2 }
        },
        {
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            name: 'ランダム',
            line: { color: '#94a3b8', dash: 'dash', width: 1 }
        }
    ];
    const layout = {
        title: 'ROC曲線',
        xaxis: { title: '偽陽性率 (FPR)', range: [0, 1] },
        yaxis: { title: '真陽性率 (TPR)', range: [0, 1] }
    };
    renderPlot(containerId, data, layout);
}

// ===========================================================================
// CSV Download Utilities
// ===========================================================================

/**
 * Convert headers and rows to CSV string with BOM for Excel.
 * @param {string[]} headers - Column header names
 * @param {Array<Array<string|number>>} rows - 2D array of cell values
 * @returns {string}
 */
export function toCSV(headers, rows) {
    const escape = (val) => {
        let s = val == null ? '' : String(val);
        const trimmed = s.trim();
        const isPlainNumber = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed);
        if (!isPlainNumber && /^[=+\-@\t\r\n＝＋－＠]/.test(s)) {
            s = `\t${s}`;
        }
        return `"${s.replace(/"/g, '""')}"`;
    };
    const lines = [headers.map(escape).join(',')];
    for (const row of rows) {
        lines.push(row.map(escape).join(','));
    }
    return '\uFEFF' + lines.join('\n');
}

/**
 * Trigger a CSV file download in the browser.
 * @param {string} csvContent - CSV-formatted string
 * @param {string} filename - Download filename
 */
export function downloadCSV(csvContent, filename) {
    // Ensure .csv extension
    const safeFilename = filename.endsWith('.csv') ? filename : filename + '.csv';
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = safeFilename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Generate a download filename with dataset name and date.
 * Format: {datasetName}_{suffix}_{YYYYMMDD}.csv
 * @param {string} datasetName - Dataset name (without extension)
 * @param {string} suffix - Type suffix (e.g., '比較結果', '予測結果')
 * @returns {string} Formatted filename
 */
export function makeExportFileName(datasetName, suffix) {
    const now = new Date();
    const date = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
    const name = datasetName || 'data';
    return `${name}_${suffix}_${date}.csv`;
}

/**
 * Create a styled download button HTML string.
 * @param {string} id - Button element id
 * @param {string} label - Button label text
 * @returns {string} HTML string
 */
export function createDownloadButton(id, label) {
    return `<button id="${id}" style="
        background: #059669; color: white; border: none; padding: 0.5rem 1.25rem;
        border-radius: 8px; font-size: 0.85rem; font-weight: 500; cursor: pointer;
        display: inline-flex; align-items: center; gap: 0.5rem; margin-top: 0.75rem;
        transition: all 0.3s ease;
    "><i class="fas fa-download"></i> ${label}</button>`;
}

// ==========================================
// Model Serialization / Deserialization
// ==========================================

/**
 * Download an object as a JSON file.
 * @param {Object} data - The data to serialize
 * @param {string} filename - Download filename
 */
export function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename.endsWith('.json') ? filename : `${filename}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Recursively serialize a decision tree node to a plain object.
 * @param {Object|null} node - TreeNode
 * @returns {Object|null}
 */
function serializeTree(node) {
    if (!node) return null;
    if (node.leaf !== undefined) {
        if (node.leaf) {
            return {
                leaf: true,
                classDist: { ...node.classDist },
                prediction: node.prediction
            };
        }
        return {
            leaf: false,
            featureIdx: node.featureIdx,
            threshold: node.threshold,
            left: serializeTree(node.left),
            right: serializeTree(node.right)
        };
    }
    if (node.value !== undefined && node.left === undefined) {
        // Leaf node
        return { value: node.value, classCounts: node.classCounts || undefined };
    }
    return {
        featureIndex: node.featureIndex,
        threshold: node.threshold,
        left: serializeTree(node.left),
        right: serializeTree(node.right),
        value: node.value,
        classCounts: node.classCounts || undefined
    };
}

/**
 * Serialize encoders Map to a JSON-safe array.
 * @param {Map<number, Object>|null} encodersMap
 * @returns {Array|null}
 */
function serializeEncoders(encodersMap) {
    if (!encodersMap) return null;
    const result = [];
    for (const [colIndex, encoder] of encodersMap) {
        result.push({
            columnIndex: colIndex,
            type: encoder.constructor?.name || 'LabelEncoder',
            classes: encoder._classes ? [...encoder._classes] : encoder.classes ? [...encoder.classes] : []
        });
    }
    return result;
}

/**
 * Extract model-specific parameters for serialization.
 * @param {Object} model - Trained model instance
 * @param {string} modelType - Badge/type identifier
 * @param {'regression'|'classification'} taskType - Analysis task type
 * @returns {Object}
 */
function serializeModelParams(model, modelType, taskType) {
    const params = model.getParams ? model.getParams() : {};

    switch (modelType) {
        case 'Linear':
        case 'Ridge':
        case 'Lasso':
            return {
                coefficients: model.coefficients ? [...model.coefficients] : null,
                intercept: model.intercept,
                alpha: params.alpha,
                nFeatures: model.nFeatures
            };
        case 'Tree':
            return {
                tree: serializeTree(model.tree),
                maxDepth: model.maxDepth,
                minSamplesSplit: model.minSamplesSplit,
                minSamplesLeaf: model.minSamplesLeaf,
                nFeatures: model.nFeatures,
                classes: model.classes ? [...model.classes] : null
            };
        case 'RF':
            return {
                trees: model.trees ? model.trees.map(t => ({
                    tree: serializeTree(t.tree),
                    featureIndices: t.featureIndices ? [...t.featureIndices] : null,
                    maxDepth: t.maxDepth,
                    classes: t.classes ? [...t.classes] : null
                })) : [],
                nEstimators: model.nEstimators,
                maxDepth: model.maxDepth,
                maxFeatures: model.maxFeatures,
                nFeatures: model.nFeatures,
                classes: model.classes ? [...model.classes] : null
            };
        case 'KNN':
            return {
                XTrain: (model.XTrain || model._X) ? (model.XTrain || model._X).map(r => [...r]) : null,
                yTrain: (model.yTrain || model._y) ? [...(model.yTrain || model._y)] : null,
                nNeighbors: model.nNeighbors,
                weights: model.weights,
                nFeatures: model.nFeatures,
                classes: model.classes ? [...model.classes] : null
            };
        case 'GBM':
            if (taskType === 'classification') {
                return {
                    models: model.models ? model.models.map(binaryModel => ({
                        initPred: binaryModel.initPred,
                        trees: binaryModel.trees.map(tree => ({
                            tree: serializeTree(tree.tree),
                            maxDepth: tree.maxDepth,
                            minSamplesSplit: tree.minSamplesSplit,
                            minSamplesLeaf: tree.minSamplesLeaf,
                            nFeatures: tree.nFeatures
                        }))
                    })) : [],
                    classes: model.classes ? [...model.classes] : null,
                    initialPredictions: model.initialPredictions ? [...model.initialPredictions] : null,
                    learningRate: model.learningRate,
                    nEstimators: model.nEstimators,
                    maxDepth: model.maxDepth,
                    subsample: model.subsample,
                    randomState: model.randomState,
                    nFeatures: model.nFeatures
                };
            }
            return {
                trees: model.trees ? model.trees.map(t => ({
                    tree: serializeTree(t.tree),
                    maxDepth: t.maxDepth
                })) : [],
                learningRate: model.learningRate,
                nEstimators: model.nEstimators,
                initialPrediction: model.initialPrediction,
                nFeatures: model.nFeatures,
                nClasses: model.nClasses,
                classPriors: model.classPriors ? [...model.classPriors] : undefined
            };
        case 'LR':
            return {
                weights: model.weights ? model.weights.map(w => [...w]) : null,
                classes: model.classes ? [...model.classes] : null,
                nFeatures: model.nFeatures,
                learningRate: model.learningRate,
                maxIter: model.maxIter,
                tol: model.tol,
                C: model.C
            };
        case 'NB':
            return {
                classPriors: model.classPriors ? { ...model.classPriors } : null,
                classMeans: model.classMeans ? Object.fromEntries(
                    Object.entries(model.classMeans).map(([cls, values]) => [cls, [...values]])
                ) : null,
                classVars: model.classVars ? Object.fromEntries(
                    Object.entries(model.classVars).map(([cls, values]) => [cls, [...values]])
                ) : null,
                classes: model.classes ? [...model.classes] : null,
                nFeatures: model.nFeatures
            };
        case 'SVM':
            return {
                weights: model.weights ? model.weights.map(w => [...w]) : null,
                classes: model.classes ? [...model.classes] : null,
                nFeatures: model.nFeatures,
                C: model.C,
                learningRate: model.learningRate,
                maxIter: model.maxIter,
                randomState: model.randomState
            };
        default:
            return params;
    }
}

/**
 * Serialize a trained model and its preprocessing pipeline to a JSON-safe object.
 * @param {Object} modelObj - { model, cls, name, badge, ... }
 * @param {Object} metadata - { featureNames, scaler, encoders, labelEncoder, targetCol, fileName, taskType, classLabels }
 * @returns {Object} Serializable model export object
 */
export function serializeModel(modelObj, metadata) {
    const now = new Date();
    return {
        version: '1.2',
        appName: 'easyDataScience',
        exportDate: now.toISOString(),
        taskType: metadata.taskType,
        targetCol: metadata.targetCol,
        featureNames: [...metadata.featureNames],
        inputFeatureNames: [...(metadata.inputFeatureNames || metadata.featureNames)],
        classLabels: metadata.classLabels ? [...metadata.classLabels] : null,
        modelInfo: {
            name: modelObj.name,
            badge: modelObj.badge,
            params: serializeModelParams(modelObj.model, modelObj.badge, metadata.taskType)
        },
        preprocessing: {
            scaler: metadata.scaler ? {
                type: metadata.scaler.constructor.name,
                means: metadata.scaler.means ? [...metadata.scaler.means] : null,
                stds: metadata.scaler.stds ? [...metadata.scaler.stds] : null,
                mins: metadata.scaler.mins ? [...metadata.scaler.mins] : null,
                maxs: metadata.scaler.maxs ? [...metadata.scaler.maxs] : null
            } : null,
            encoders: serializeEncoders(metadata.encoders),
            pipeline: metadata.pipelineSpec ? JSON.parse(JSON.stringify(metadata.pipelineSpec)) : null,
            labelEncoder: metadata.labelEncoder ? {
                classes: [...metadata.labelEncoder._classes]
            } : null
        },
        datasetName: metadata.fileName || 'unknown'
    };
}

/**
 * Validate and parse a model JSON file.
 * @param {string|Object} jsonData - Raw JSON string or parsed object
 * @returns {Object} Parsed and validated model data
 * @throws {Error} If validation fails
 */
export function deserializeModel(jsonData) {
    const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error('無効なモデルファイル: JSONオブジェクトではありません');
    }
    if (!['1.1', '1.2'].includes(String(data.version))) {
        throw new Error('無効なモデルファイル: 未対応のバージョンです');
    }
    if (data.appName !== 'easyDataScience') throw new Error('無効なモデルファイル: easyDataScienceで作成されたファイルではありません');
    if (!['regression', 'classification'].includes(data.taskType)) {
        throw new Error('無効なモデルファイル: タスク種別が不正です');
    }
    const validBadges = new Set(['Linear', 'Ridge', 'Lasso', 'Tree', 'RF', 'KNN', 'GBM', 'LR', 'NB', 'SVM']);
    if (!data.modelInfo || typeof data.modelInfo !== 'object' || !validBadges.has(data.modelInfo.badge)) {
        throw new Error('無効なモデルファイル: モデル情報が不正です');
    }
    if (!data.modelInfo.params || typeof data.modelInfo.params !== 'object') {
        throw new Error('無効なモデルファイル: モデルパラメータがありません');
    }
    if (!Array.isArray(data.featureNames) || data.featureNames.length === 0 || data.featureNames.length > 10000 ||
        data.featureNames.some(name => typeof name !== 'string' || name.length === 0 || name.length > 200)) {
        throw new Error('無効なモデルファイル: 特徴量情報が不正です');
    }
    if (data.inputFeatureNames != null && (!Array.isArray(data.inputFeatureNames) ||
        data.inputFeatureNames.length === 0 || data.inputFeatureNames.length > 10000 ||
        data.inputFeatureNames.some(name => typeof name !== 'string' || name.length === 0 || name.length > 200))) {
        throw new Error('無効なモデルファイル: 入力特徴量情報が不正です');
    }
    if (typeof data.targetCol !== 'string' || data.targetCol.length > 200) {
        throw new Error('無効なモデルファイル: 目的変数情報が不正です');
    }
    const pipeline = data.preprocessing?.pipeline;
    if (pipeline != null) {
        if (!Array.isArray(pipeline.inputFeatureNames) || !Array.isArray(pipeline.outputFeatureNames) ||
            !Array.isArray(pipeline.featureSpecs) || !Array.isArray(pipeline.outputFeatures) ||
            pipeline.outputFeatureNames.length !== data.featureNames.length ||
            pipeline.outputFeatures.length !== data.featureNames.length) {
            throw new Error('無効なモデルファイル: 前処理パイプラインが不正です');
        }
        for (const def of pipeline.outputFeatures) {
            if (!def || typeof def.sourceName !== 'string' || !['numeric', 'onehot'].includes(def.type)) {
                throw new Error('無効なモデルファイル: 前処理の特徴量定義が不正です');
            }
            if (def.type === 'numeric' && !Number.isFinite(Number(def.fillValue))) {
                throw new Error('無効なモデルファイル: 数値補完値が不正です');
            }
        }
    }
    validateSerializedModelParameters(data);
    return data;
}

function validateSerializedModelParameters(data) {
    const { badge, params } = data.modelInfo;
    const featureCount = data.featureNames.length;
    const allowedByTask = data.taskType === 'regression'
        ? new Set(['Linear', 'Ridge', 'Lasso', 'Tree', 'RF', 'KNN', 'GBM'])
        : new Set(['Tree', 'RF', 'KNN', 'GBM', 'LR', 'NB', 'SVM']);
    if (!allowedByTask.has(badge)) {
        throw new Error('無効なモデルファイル: タスク種別とモデルが一致しません');
    }

    const finiteVector = (values, expectedLength, label) => {
        if (!Array.isArray(values) || values.length !== expectedLength || values.some(value => !Number.isFinite(value))) {
            throw new Error(`無効なモデルファイル: ${label}の次元または値が不正です`);
        }
    };
    const validateClasses = () => {
        if (!Array.isArray(params.classes) || params.classes.length < 2 || params.classes.length > 1000 ||
            params.classes.some(value => !['string', 'number', 'boolean'].includes(typeof value))) {
            throw new Error('無効なモデルファイル: クラス情報が不正です');
        }
    };

    if (['Linear', 'Ridge', 'Lasso'].includes(badge)) {
        finiteVector(params.coefficients, featureCount, '係数');
        if (!Number.isFinite(params.intercept)) throw new Error('無効なモデルファイル: 切片が不正です');
    }

    if (badge === 'LR' || badge === 'SVM') {
        validateClasses();
        if (!Array.isArray(params.weights) || params.weights.length === 0 || params.weights.length > params.classes.length) {
            throw new Error('無効なモデルファイル: 重み情報が不正です');
        }
        params.weights.forEach((weights, index) => {
            const vector = Array.isArray(weights) ? weights : [...(weights?.w || []), weights?.b];
            finiteVector(vector, featureCount + 1, `重み${index + 1}`);
        });
    }

    if (badge === 'KNN') {
        if (!Array.isArray(params.XTrain) || params.XTrain.length === 0 || params.XTrain.length > 100000 ||
            !Array.isArray(params.yTrain) || params.XTrain.length !== params.yTrain.length) {
            throw new Error('無効なモデルファイル: KNN学習データが不正です');
        }
        params.XTrain.forEach((row, index) => finiteVector(row, featureCount, `KNN学習行${index + 1}`));
        if (data.taskType === 'regression' && params.yTrain.some(value => !Number.isFinite(value))) {
            throw new Error('無効なモデルファイル: KNN目的変数が不正です');
        }
        if (data.taskType === 'classification') validateClasses();
    }

    if (badge === 'NB') {
        validateClasses();
        if (!params.classPriors || !params.classMeans || !params.classVars) {
            throw new Error('無効なモデルファイル: Naive Bayes統計量がありません');
        }
        params.classes.forEach(cls => {
            const key = String(cls);
            if (!Number.isFinite(params.classPriors[key]) || params.classPriors[key] < 0 || params.classPriors[key] > 1) {
                throw new Error('無効なモデルファイル: Naive Bayes事前確率が不正です');
            }
            finiteVector(params.classMeans[key], featureCount, `クラス${key}の平均`);
            finiteVector(params.classVars[key], featureCount, `クラス${key}の分散`);
            if (params.classVars[key].some(value => value < 0)) {
                throw new Error('無効なモデルファイル: Naive Bayes分散が不正です');
            }
        });
    }

    if (badge === 'GBM' && data.taskType === 'classification') {
        validateClasses();
        const expectedModels = params.classes.length === 2 ? 1 : params.classes.length;
        if (!Array.isArray(params.models) || params.models.length !== expectedModels ||
            params.models.some(model => !Number.isFinite(model?.initPred) || !Array.isArray(model?.trees))) {
            throw new Error('無効なモデルファイル: 分類GBMの構造が不正です');
        }
    }

    const scaler = data.preprocessing?.scaler;
    if (scaler) {
        if (scaler.type === 'StandardScaler') {
            finiteVector(scaler.means, featureCount, '標準化平均');
            finiteVector(scaler.stds, featureCount, '標準化標準偏差');
            if (scaler.stds.some(value => value <= 0)) throw new Error('無効なモデルファイル: 標準偏差が不正です');
        } else if (scaler.type === 'MinMaxScaler') {
            finiteVector(scaler.mins, featureCount, '最小値');
            finiteVector(scaler.maxs, featureCount, '最大値');
        } else {
            throw new Error('無効なモデルファイル: 未対応のスケーラーです');
        }
    }
}

/**
 * Generate a descriptive filename for model export.
 * @param {string} datasetName - Original dataset name
 * @param {string} modelBadge - Model badge (e.g., 'RF', 'Linear')
 * @param {string} taskType - 'regression' or 'classification'
 * @returns {string}
 */
export function makeModelFileName(datasetName, modelBadge, taskType) {
    const base = datasetName.replace(/\.\w+$/, '').replace(/[^\w\u3000-\u9fff]/g, '_');
    const task = taskType === 'regression' ? '回帰' : '分類';
    const date = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    return `${base}_${task}_${modelBadge}_${date}.json`;
}
