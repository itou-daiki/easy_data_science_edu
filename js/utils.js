// ==========================================
// UI Helpers for easyDataScience
// ==========================================

/**
 * Toggles the visibility of a collapsible section.
 * @param {HTMLElement} header
 */
export function toggleCollapsible(header) {
    header.classList.toggle('collapsed');
    const content = header.nextElementSibling;
    content.classList.toggle('collapsed');
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
        uploadText.textContent = 'ここにファイルをドラッグ＆ドロップ';
    }
}

/**
 * Shows an error message using a simple alert.
 * @param {string} message
 */
export function showError(message) {
    alert(`エラー: ${message}`);
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
    headers.forEach(h => table += `<th>${h}</th>`);
    table += '</tr></thead><tbody>';
    rowLabels.forEach((r, i) => {
        table += `<tr><th>${r}</th>`;
        data[i].forEach(d => table += `<td>${typeof d === 'number' ? d.toFixed(4) : d}</td>`);
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

    let html = `<div class="table-container"><table class="table">`;
    html += '<thead><tr>';
    html += '<th>#</th>';
    columns.forEach(col => html += `<th>${col}</th>`);
    html += '</tr></thead><tbody>';

    displayData.forEach((row, i) => {
        html += `<tr><td>${i + 1}</td>`;
        columns.forEach(col => {
            const val = row[col];
            html += `<td>${val != null ? val : '<span style="color:#94a3b8;">N/A</span>'}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    if (data.length > maxRows) {
        html += `<p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">先頭 ${maxRows} 行を表示（全 ${data.length} 行）</p>`;
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
        container.innerHTML = '<p style="color: var(--text-secondary);">数値変数がありません。</p>';
        return;
    }

    const stats = numCols.map(col => {
        const values = data.map(row => row[col]).filter(v => v != null && !isNaN(Number(v))).map(Number);
        if (values.length === 0) return { col, count: 0, mean: '-', std: '-', min: '-', q1: '-', median: '-', q3: '-', max: '-', missing: data.length };
        const sorted = [...values].sort((a, b) => a - b);
        const n = values.length;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
        const std = Math.sqrt(variance);
        const q1 = sorted[Math.floor(n * 0.25)];
        const median = sorted[Math.floor(n * 0.5)];
        const q3 = sorted[Math.floor(n * 0.75)];
        const missing = data.length - n;
        return { col, count: n, mean, std, min: sorted[0], q1, median, q3, max: sorted[n - 1], missing };
    });

    let html = '<div class="table-container"><table class="table">';
    html += '<thead><tr><th>変数</th><th>件数</th><th>平均</th><th>標準偏差</th><th>最小</th><th>Q1</th><th>中央値</th><th>Q3</th><th>最大</th><th>欠損</th></tr></thead><tbody>';
    stats.forEach(s => {
        const fmt = v => typeof v === 'number' ? v.toFixed(3) : v;
        html += `<tr><td><strong>${s.col}</strong></td><td>${s.count}</td><td>${fmt(s.mean)}</td><td>${fmt(s.std)}</td><td>${fmt(s.min)}</td><td>${fmt(s.q1)}</td><td>${fmt(s.median)}</td><td>${fmt(s.q3)}</td><td>${fmt(s.max)}</td><td>${s.missing}</td></tr>`;
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
    let html = `<select id="${id}" class="form-select">`;
    html += `<option value="">${placeholder}</option>`;
    options.forEach(opt => html += `<option value="${opt}">${opt}</option>`);
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
        html += `<label class="variable-chip ${isSelected ? 'selected' : ''}" data-name="${name}" data-value="${opt}">
            <input type="checkbox" name="${name}" value="${opt}" ${isSelected ? 'checked' : ''} style="display:none;">
            ${opt}
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
    const formattedValue = formatNumber(value);
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
        width: 450,
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
