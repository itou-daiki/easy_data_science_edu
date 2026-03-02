// ==========================================
// 分類モデル比較 (AutoML) Module
// PyCaret-style: setup → compare_models (CV) → tune_model → predict_model
// ==========================================
import { createSelect, createStepIndicator, formatNumber, renderPlot, renderConfusionMatrix, renderROCCurve, renderFeatureImportance, createMetricCard, renderPermutationImportance, renderPDP, renderLearningCurve, renderSHAPSummary, renderSHAPBeeswarm, renderSHAPWaterfall, toCSV, downloadCSV, createDownloadButton, makeExportFileName, renderDataPreview, renderSummaryStatistics, downloadJSON, serializeModel, makeModelFileName } from '../utils.js';
import { linearSHAP, kernelSHAP, shapSummary } from '../ml/shap.js';
import { prepareFeatures } from '../ml/preprocessing.js';
import { trainTestSplit, crossValidate, gridSearch, permutationImportance, learningCurve } from '../ml/model_selection.js';
import { accuracy, precisionScore, recallScore, f1Score, confusionMatrix, logLoss, rocAucScore } from '../ml/metrics.js';
import { LogisticRegression } from '../ml/classification/logistic.js';
import { DecisionTreeClassifier } from '../ml/classification/decision_tree.js';
import { RandomForestClassifier } from '../ml/classification/random_forest.js';
import { KNNClassifier } from '../ml/classification/knn.js';
import { GaussianNaiveBayes } from '../ml/classification/naive_bayes.js';
import { SVMClassifier } from '../ml/classification/svm.js';
import { GradientBoostingClassifier } from '../ml/classification/gradient_boosting.js';

const MODELS = [
    { name: 'ロジスティック回帰', cls: LogisticRegression, params: { maxIter: 500 }, badge: 'LR' },
    { name: '決定木', cls: DecisionTreeClassifier, params: { maxDepth: 5 }, badge: 'Tree' },
    { name: 'ランダムフォレスト', cls: RandomForestClassifier, params: { nEstimators: 100, maxDepth: 8 }, badge: 'RF' },
    { name: 'K近傍法', cls: KNNClassifier, params: { nNeighbors: 5 }, badge: 'KNN' },
    { name: 'ナイーブベイズ', cls: GaussianNaiveBayes, params: {}, badge: 'NB' },
    { name: 'SVM', cls: SVMClassifier, params: { C: 1.0, maxIter: 500 }, badge: 'SVM' },
    { name: '勾配ブースティング', cls: GradientBoostingClassifier, params: { nEstimators: 100, learningRate: 0.1, maxDepth: 3 }, badge: 'GBM' }
];

const PARAM_GRIDS = {
    'LR': { maxIter: [200, 500, 1000] },
    'Tree': { maxDepth: [3, 5, 8, 10] },
    'RF': { nEstimators: [50, 100], maxDepth: [5, 8] },
    'KNN': { nNeighbors: [3, 5, 7, 9] },
    'SVM': { C: [0.1, 1.0, 10.0], maxIter: [500] },
    'GBM': { nEstimators: [50, 100], learningRate: [0.05, 0.1], maxDepth: [3, 5] },
};

const STEPS = ['Setup', 'Preprocess', 'Compare', 'Create', 'Tune', 'Interpret', 'Blend', 'Stack', 'Finalize', 'Predict'];

// Module-level state for sharing data between steps
let _state = {};

export function render(container, data, characteristics) {
    _state = {};
    const catCols = characteristics.categoricalColumns;
    const numCols = characteristics.numericColumns;
    const allCols = characteristics.allColumns || Object.keys(data[0]);

    // Find suitable target columns (categorical with 2-20 unique values)
    const targetCandidates = allCols.filter(col => {
        const values = data.map(r => r[col]).filter(v => v != null);
        const unique = new Set(values).size;
        return unique >= 2 && unique <= 20;
    });

    container.innerHTML = `
        <h2><i class="fas fa-robot" style="color: #0891b2;"></i> 分類モデル比較 (AutoML)</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            PyCaret のように複数の分類モデルを一括学習・比較し、最適なモデルを見つけます。
        </p>

        ${createStepIndicator(STEPS, 0)}

        <div id="data-overview-section" style="margin-bottom: 1.5rem;">
            <div id="cls-dataframe-container"></div>
            <div id="cls-summary-stats-container"></div>
        </div>

        <div id="setup-section" class="model-config">
            <h3><i class="fas fa-cog"></i> Step 1: セットアップ</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">
                        目的変数（予測したいカテゴリ変数）:
                    </label>
                    ${createSelect('target-select', targetCandidates, '目的変数を選択')}
                    <p id="target-info" style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;"></p>
                </div>
                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">
                        テストデータの割合:
                    </label>
                    <select id="test-size-select" class="form-select">
                        <option value="0.2">20%</option>
                        <option value="0.25">25%</option>
                        <option value="0.3" selected>30%</option>
                    </select>
                </div>
                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">
                        交差検証 Fold数:
                    </label>
                    <select id="cv-fold-select" class="form-select">
                        <option value="3">3-Fold</option>
                        <option value="5" selected>5-Fold</option>
                        <option value="10">10-Fold</option>
                    </select>
                </div>
            </div>
            <div id="feature-selection" style="margin: 1rem 0; display: none;">
                <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">
                    使用する特徴量（数値変数のみ）:
                </label>
                <div id="feature-chips" class="variable-chips"></div>
            </div>
            <button id="btn-compare" class="btn-analysis" style="background: #0891b2;" disabled>
                <i class="fas fa-play"></i> モデル比較を開始
            </button>
        </div>

        <div id="compare-section" style="display: none;">
            <div id="progress-area"></div>
            <div id="comparison-results"></div>
        </div>

        <div id="evaluate-section" style="display: none;">
            <div id="evaluation-content"></div>
        </div>
    `;

    renderDataPreview('cls-dataframe-container', data, 'データプレビュー');
    renderSummaryStatistics('cls-summary-stats-container', data, characteristics, '要約統計量');

    const targetSelect = container.querySelector('#target-select');
    const btnCompare = container.querySelector('#btn-compare');
    const featureSelection = container.querySelector('#feature-selection');
    const featureChips = container.querySelector('#feature-chips');
    const targetInfo = container.querySelector('#target-info');

    targetSelect.addEventListener('change', () => {
        const target = targetSelect.value;
        if (!target) {
            featureSelection.style.display = 'none';
            btnCompare.disabled = true;
            targetInfo.innerHTML = '';
            return;
        }

        const values = data.map(r => r[target]).filter(v => v != null);
        const unique = [...new Set(values)];
        targetInfo.innerHTML = `クラス数: ${unique.length} (${unique.slice(0, 5).join(', ')}${unique.length > 5 ? '...' : ''})`;

        const features = numCols.filter(c => c !== target);
        if (features.length === 0) {
            featureChips.innerHTML = '<p style="color: #ef4444;">数値特徴量がありません。数値変数を含むデータをお使いください。</p>';
            featureSelection.style.display = 'block';
            btnCompare.disabled = true;
            return;
        }

        featureChips.innerHTML = features.map(f =>
            `<label class="variable-chip selected" data-value="${f}">
                <input type="checkbox" value="${f}" checked style="display:none;"> ${f}
            </label>`
        ).join('');

        featureChips.querySelectorAll('.variable-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const cb = chip.querySelector('input');
                cb.checked = !cb.checked;
                chip.classList.toggle('selected', cb.checked);
            });
        });

        featureSelection.style.display = 'block';
        btnCompare.disabled = false;
    });

    btnCompare.addEventListener('click', () => runComparison(container, data, characteristics));
}

async function runComparison(container, data, characteristics) {
    const targetCol = container.querySelector('#target-select').value;
    const testSize = parseFloat(container.querySelector('#test-size-select').value);
    const cvFolds = parseInt(container.querySelector('#cv-fold-select').value);
    const selectedFeatures = Array.from(container.querySelectorAll('#feature-chips input:checked')).map(cb => cb.value);

    if (selectedFeatures.length === 0) {
        alert('特徴量を1つ以上選択してください。');
        return;
    }

    container.querySelector('#setup-section').style.display = 'none';
    container.querySelector('#compare-section').style.display = 'block';

    container.querySelector('.step-indicator').innerHTML = createStepIndicator(STEPS, 1).replace(/<\/?div[^>]*class="step-indicator"[^>]*>/g, '');

    const progressArea = container.querySelector('#progress-area');
    progressArea.innerHTML = `<div style="text-align: center; padding: 2rem;">
        <i class="fas fa-spinner fa-spin fa-2x" style="color: #0891b2;"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
        <div id="model-progress" style="margin-top: 1rem;"></div>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const { X, y, featureNames, labelEncoder, encoders, scaler } = prepareFeatures(data, targetCol, {
            selectedFeatures,
            task: 'classification'
        });

        const classes = [...new Set(y)].sort((a, b) => a - b);
        const classLabels = labelEncoder ? classes.map(c => labelEncoder.inverseTransform([c])[0]) : classes.map(String);

        const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize, randomState: 42 });

        // Save state for tune/predict
        _state = { XTrain, XTest, yTrain, yTest, featureNames, scaler, encoders, labelEncoder, classes, classLabels, cvFolds, targetCol, fileName: characteristics.fileName || 'data' };

        // Compute preprocessing info
        const missingCount = selectedFeatures.reduce((sum, col) => {
            return sum + data.filter(row => row[col] == null || row[col] === '').length;
        }, 0);
        const categoricalCount = encoders ? encoders.size : 0;
        const scalingApplied = scaler !== null;

        // Show preprocessing summary + model training progress
        const progressArea = container.querySelector('#progress-area');
        progressArea.innerHTML = `
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <h3 style="margin: 0 0 1rem 0; font-size: 1.1rem; color: #166534;">
                    <i class="fas fa-magic" style="margin-right: 0.5rem;"></i>Step 2: 前処理 (自動完了)
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>目的変数: <strong>${targetCol}</strong> (${classLabels.length}クラス)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>特徴量: <strong>${featureNames.length}個</strong></span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>欠損値処理: ${missingCount > 0 ? missingCount + '個を平均値で補完' : '欠損値なし'}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>カテゴリ変数: ${categoricalCount > 0 ? categoricalCount + '列をLabel Encoding' : 'エンコード不要'}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>スケーリング: ${scalingApplied ? 'StandardScaler (平均0, 分散1)' : 'なし'}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>データ分割: 訓練${XTrain.length}件 / テスト${XTest.length}件</span>
                    </div>
                </div>
            </div>
            <div style="text-align: center; padding: 2rem;">
                <i class="fas fa-spinner fa-spin fa-2x" style="color: #0891b2;"></i>
                <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
                <div id="model-progress" style="margin-top: 1rem;"></div>
            </div>
        `;

        // Update step to Compare
        container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 2);

        const results = [];
        const modelProgress = container.querySelector('#model-progress');

        for (let i = 0; i < MODELS.length; i++) {
            const modelDef = MODELS[i];
            modelProgress.innerHTML = `<p style="color: var(--text-secondary);">${modelDef.name} を学習中... (${i + 1}/${MODELS.length})</p>
                <div class="progress-bar"><div class="progress-fill" style="width: ${((i + 1) / MODELS.length) * 100}%"></div></div>`;

            await new Promise(r => setTimeout(r, 50));

            try {
                const model = new modelDef.cls(modelDef.params);
                model.fit(XTrain, yTrain);
                const yPred = model.predict(XTest);
                const yProba = model.predictProba ? model.predictProba(XTest) : null;

                // Cross-validation on training data
                const cvScores = crossValidate(model, XTrain, yTrain, { cv: cvFolds, scoring: 'f1' });
                const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
                const cvStd = Math.sqrt(cvScores.reduce((a, v) => a + (v - cvMean) ** 2, 0) / cvScores.length);

                const acc = accuracy(yTest, yPred);
                const prec = precisionScore(yTest, yPred);
                const rec = recallScore(yTest, yPred);
                const f1 = f1Score(yTest, yPred);
                const cmResult = confusionMatrix(yTest, yPred);
                const cm = cmResult.matrix;

                let auc = null;
                let ll = null;
                if (yProba && classes.length === 2) {
                    const positiveProba = yProba.map(p => p[1] || 0);
                    auc = rocAucScore(yTest, positiveProba);
                    ll = logLoss(yTest, positiveProba);
                }

                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    cls: modelDef.cls,
                    model,
                    acc, prec, rec, f1, cm, auc, ll,
                    cvMean, cvStd, cvScores,
                    yPred, yProba,
                    featureImportance: model.getFeatureImportance ? model.getFeatureImportance() : null
                });
            } catch (err) {
                console.error(`${modelDef.name} failed:`, err);
                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    cls: modelDef.cls,
                    model: null,
                    acc: 0, prec: 0, rec: 0, f1: 0, cm: null, auc: null, ll: null,
                    cvMean: -Infinity, cvStd: 0, cvScores: [],
                    yPred: null, yProba: null,
                    error: err.message
                });
            }
        }

        // Sort by CV mean F1 (PyCaret-style)
        results.sort((a, b) => b.cvMean - a.cvMean);
        _state.results = results;

        renderComparisonResults(container, results, yTest, featureNames, classes, classLabels);
    } catch (error) {
        progressArea.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-triangle"></i> エラー: ${error.message}</p>`;
        console.error(error);
    }
}

function renderComparisonResults(container, results, yTest, featureNames, classes, classLabels) {
    const comparisonDiv = container.querySelector('#comparison-results');
    const bestCV = Math.max(...results.filter(r => r.model).map(r => r.cvMean));
    const hasAuc = results.some(r => r.auc != null);

    let html = `
        <h3 style="margin-top: 1rem;"><i class="fas fa-trophy" style="color: #0891b2;"></i> モデル比較結果</h3>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            ${_state.cvFolds}-Fold 交差検証スコア（訓練データ）でソートしています。テストデータ (${yTest.length} サンプル) での評価結果も併記。
        </p>
        <div class="table-container">
            <table class="table model-comparison-table">
                <thead>
                    <tr>
                        <th>順位</th>
                        <th>モデル</th>
                        <th>CV F1 (mean)</th>
                        <th>CV F1 (std)</th>
                        <th>Test F1</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        ${hasAuc ? '<th>AUC</th>' : ''}
                        <th>詳細</th>
                    </tr>
                </thead>
                <tbody>
                    ${results.map((r, i) => `
                        <tr class="${r.cvMean === bestCV && r.model ? 'highlight-row' : ''} ${!r.model ? 'error-row' : ''}">
                            <td>${r.model ? i + 1 : '-'}</td>
                            <td>
                                <span class="badge badge-${r.badge.toLowerCase()}">${r.badge}</span>
                                ${r.name}
                                ${r.cvMean === bestCV && r.model ? ' <i class="fas fa-crown" style="color: #0891b2;"></i>' : ''}
                            </td>
                            <td><strong>${r.model ? formatNumber(r.cvMean) : '-'}</strong></td>
                            <td>${r.model ? formatNumber(r.cvStd) : '-'}</td>
                            <td>${r.model ? formatNumber(r.f1) : '-'}</td>
                            <td>${r.model ? formatNumber(r.acc) : '<span style="color:#ef4444;">エラー</span>'}</td>
                            <td>${r.model ? formatNumber(r.prec) : '-'}</td>
                            <td>${r.model ? formatNumber(r.rec) : '-'}</td>
                            ${hasAuc ? `<td>${r.auc != null ? formatNumber(r.auc) : '-'}</td>` : ''}
                            <td>${r.model ? `<button class="btn-detail" data-index="${i}" style="padding: 0.25rem 0.75rem; font-size: 0.85rem;">詳細</button>` : '-'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        ${createDownloadButton('dl-comparison-csv', '比較結果をCSVダウンロード')}
    `;

    comparisonDiv.innerHTML = html;
    container.querySelector('#progress-area').innerHTML = '';

    comparisonDiv.querySelectorAll('.btn-detail').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.index);
            showModelDetail(container, results[idx], yTest, featureNames, classes, classLabels);
        });
    });

    const dlCompBtn = comparisonDiv.querySelector('#dl-comparison-csv');
    if (dlCompBtn) {
        dlCompBtn.addEventListener('click', () => {
            const headers = ['順位', 'モデル', 'Badge', 'CV F1 (mean)', 'CV F1 (std)', 'Test F1', 'Accuracy', 'Precision', 'Recall'];
            if (hasAuc) headers.push('AUC');
            const rows = results.map((r, i) => {
                const row = [
                    r.model ? i + 1 : '-',
                    r.name,
                    r.badge,
                    r.model ? formatNumber(r.cvMean) : '-',
                    r.model ? formatNumber(r.cvStd) : '-',
                    r.model ? formatNumber(r.f1) : '-',
                    r.model ? formatNumber(r.acc) : '-',
                    r.model ? formatNumber(r.prec) : '-',
                    r.model ? formatNumber(r.rec) : '-'
                ];
                if (hasAuc) row.push(r.auc != null ? formatNumber(r.auc) : '-');
                return row;
            });
            const csv = toCSV(headers, rows);
            downloadCSV(csv, makeExportFileName(_state.fileName, '分類_比較結果'));
        });
    }

    const bestModel = results.find(r => r.model);
    if (bestModel) {
        showModelDetail(container, bestModel, yTest, featureNames, classes, classLabels);
    }
}

function showModelDetail(container, result, yTest, featureNames, classes, classLabels) {
    const evalSection = container.querySelector('#evaluate-section');
    evalSection.style.display = 'block';

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 3);

    const evalContent = container.querySelector('#evaluation-content');
    const hasTuneGrid = PARAM_GRIDS[result.badge] != null;

    evalContent.innerHTML = `
        <h3><i class="fas fa-chart-bar" style="color: #0891b2;"></i> ${result.name} の詳細評価</h3>

        <div class="metrics-grid" style="margin: 1.5rem 0;">
            ${createMetricCard('CV F1 (mean)', result.cvMean, `${_state.cvFolds}-Fold 交差検証平均`)}
            ${createMetricCard('CV F1 (std)', result.cvStd, '交差検証の標準偏差')}
            ${createMetricCard('Accuracy', result.acc, '正解率')}
            ${createMetricCard('Precision', result.prec, '適合率')}
            ${createMetricCard('Recall', result.rec, '再現率')}
            ${createMetricCard('F1 Score', result.f1, 'F1スコア')}
            ${result.auc != null ? createMetricCard('AUC', result.auc, 'ROC曲線下面積') : ''}
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 2rem;">
            <div id="confusion-matrix-plot"></div>
            ${result.auc != null ? '<div id="roc-curve-plot"></div>' : '<div></div>'}
        </div>

        ${result.featureImportance ? '<div id="feature-importance-plot" style="margin-top: 2rem;"></div>' : ''}

        <div style="margin-top: 2rem;">
            <h4>結果の解釈</h4>
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #0891b2; line-height: 1.8;">
                ${interpretResults(result)}
            </div>
        </div>

        <!-- Create Model Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #fef9c3, #fde68a); border-radius: 12px;">
            <h4><i class="fas fa-plus-circle" style="color: #b45309;"></i> create_model - 個別モデル作成</h4>
            <p style="color: #92400e; margin: 0.5rem 0;">
                特定のモデルをパラメータ指定で作成・学習します。Compare の結果を踏まえ、詳細にモデルを構築できます。
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">モデルを選択:</label>
                    <select id="create-model-select" class="form-select">
                        ${MODELS.map((m, i) => {
                            const isTop = _state.results && _state.results[0] && _state.results[0].badge === m.badge;
                            return `<option value="${i}" ${isTop ? 'selected' : ''}>${m.name} (${m.badge})${isTop ? ' ★1位' : ''}</option>`;
                        }).join('')}
                    </select>
                </div>
                <div id="create-model-params">
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">パラメータ:</label>
                    <div id="create-param-fields"></div>
                </div>
            </div>
            <button id="btn-create-model" class="btn-analysis" style="background: #b45309; margin-top: 0.5rem;">
                <i class="fas fa-hammer"></i> create_model を実行
            </button>
            <div id="create-model-results" style="margin-top: 1rem;"></div>
        </div>

        <!-- Tune Model Section -->
        ${hasTuneGrid ? `
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #cffafe, #a5f3fc); border-radius: 12px;">
            <h4><i class="fas fa-sliders-h" style="color: #0891b2;"></i> tune_model - ハイパーパラメータチューニング</h4>
            <p style="color: #164e63; margin: 0.5rem 0;">
                GridSearch CV でパラメータを最適化します。
                探索範囲: ${Object.entries(PARAM_GRIDS[result.badge]).map(([k, v]) => `${k}=[${v.join(', ')}]`).join(', ')}
            </p>
            <button id="btn-tune" class="btn-analysis" style="background: #0891b2; margin-top: 1rem;">
                <i class="fas fa-magic"></i> tune_model を実行
            </button>
            <div id="tune-results" style="margin-top: 1rem;"></div>
        </div>
        ` : ''}

        <!-- Interpret Model Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-radius: 12px;">
            <h4><i class="fas fa-search-plus" style="color: #16a34a;"></i> interpret_model - モデル解釈</h4>
            <p style="color: #166534; margin: 0.5rem 0;">
                Permutation Feature Importance、PDP、Learning Curve、SHAP でモデルを深く理解します。
            </p>
            <button id="btn-interpret" class="btn-analysis" style="background: #16a34a; margin-top: 1rem;">
                <i class="fas fa-microscope"></i> interpret_model を実行
            </button>
            <div id="interpret-results" style="margin-top: 1rem;"></div>
        </div>

        <!-- Blend Models Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #faf5ff, #f3e8ff); border-radius: 12px;">
            <h4><i class="fas fa-layer-group" style="color: #7c3aed;"></i> blend_models - モデルアンサンブル</h4>
            <p style="color: #5b21b6; margin: 0.5rem 0;">
                上位モデルの予測確率を平均して、より安定した分類を実現します。
            </p>
            <div style="margin: 1rem 0;">
                <label style="font-weight: 600; margin-right: 0.5rem;">ブレンドするモデル数:</label>
                <select id="blend-top-n" class="form-select" style="display: inline-block; width: auto;">
                    <option value="3" selected>上位3モデル</option>
                    <option value="5">上位5モデル</option>
                    <option value="7">全モデル (7)</option>
                </select>
            </div>
            <button id="btn-blend" class="btn-analysis" style="background: #7c3aed; margin-top: 0.5rem;">
                <i class="fas fa-blender"></i> blend_models を実行
            </button>
            <div id="blend-results" style="margin-top: 1rem;"></div>
        </div>

        <!-- Stack Models Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #fdf2f8, #fce7f3); border-radius: 12px;">
            <h4><i class="fas fa-layer-group" style="color: #be185d;"></i> stack_models - スタッキングアンサンブル</h4>
            <p style="color: #9d174d; margin: 0.5rem 0;">
                上位モデルの予測をメタ学習器（ロジスティック回帰）の入力として使い、より高精度な予測を目指します。
            </p>
            <div style="margin: 1rem 0;">
                <label style="font-weight: 600; margin-right: 0.5rem;">スタックするモデル数:</label>
                <select id="stack-top-n" class="form-select" style="display: inline-block; width: auto;">
                    <option value="3" selected>上位3モデル</option>
                    <option value="5">上位5モデル</option>
                    <option value="7">全モデル (7)</option>
                </select>
            </div>
            <button id="btn-stack" class="btn-analysis" style="background: #be185d; margin-top: 0.5rem;">
                <i class="fas fa-object-group"></i> stack_models を実行
            </button>
            <div id="stack-results" style="margin-top: 1rem;"></div>
        </div>

        <!-- Finalize Model Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #fef2f2, #fecaca); border-radius: 12px;">
            <h4><i class="fas fa-check-double" style="color: #dc2626;"></i> finalize_model - モデル確定</h4>
            <p style="color: #991b1b; margin: 0.5rem 0;">
                全データ（訓練+テスト）で再学習し、本番用モデルとして確定します。
            </p>
            <button id="btn-finalize" class="btn-analysis" style="background: #dc2626; margin-top: 1rem;">
                <i class="fas fa-flag-checkered"></i> finalize_model を実行
            </button>
            <div id="finalize-results" style="margin-top: 1rem;"></div>
        </div>

        <!-- Predict Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #dbeafe, #bfdbfe); border-radius: 12px;">
            <h4><i class="fas fa-calculator" style="color: #2563eb;"></i> predict_model - 新しいデータで予測</h4>
            <p style="color: #1e40af; margin: 0.5rem 0;">
                各特徴量の値を入力して予測を実行します。
            </p>
            <div id="predict-inputs" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                ${featureNames.map(f => `
                    <div>
                        <label style="font-weight: 600; font-size: 0.85rem; display: block; margin-bottom: 0.25rem;">${f}</label>
                        <input type="number" id="pred-${f}" class="form-select" step="any" placeholder="値を入力" style="width: 100%;">
                    </div>
                `).join('')}
            </div>
            <button id="btn-predict" class="btn-analysis" style="background: #2563eb; margin-top: 0.5rem;">
                <i class="fas fa-play"></i> predict_model を実行
            </button>
            <div id="predict-result" style="margin-top: 1rem;"></div>
        </div>
    `;

    if (result.cm) {
        renderConfusionMatrix('confusion-matrix-plot', result.cm, classLabels);
    }

    if (result.auc != null && result.yProba && classes.length === 2) {
        const positiveProba = result.yProba.map(p => p[1] || 0);
        renderROCCurve('roc-curve-plot', yTest, positiveProba, result.auc);
    }

    if (result.featureImportance && featureNames) {
        renderFeatureImportance('feature-importance-plot', featureNames, result.featureImportance);
    }

    // Create Model: populate param fields on model select change
    const createModelSelect = container.querySelector('#create-model-select');
    updateCreateModelParams(container, parseInt(createModelSelect.value));
    createModelSelect.addEventListener('change', () => {
        updateCreateModelParams(container, parseInt(createModelSelect.value));
    });

    // Create Model button handler
    container.querySelector('#btn-create-model').addEventListener('click', () => {
        runCreateModel(container, featureNames, classes, classLabels);
    });

    // Tune button handler
    if (hasTuneGrid) {
        container.querySelector('#btn-tune').addEventListener('click', () => {
            runTuneModel(container, result, featureNames, classes, classLabels);
        });
    }

    // Interpret button handler
    container.querySelector('#btn-interpret').addEventListener('click', () => {
        runInterpretModel(container, result, featureNames);
    });

    // Blend button handler
    container.querySelector('#btn-blend').addEventListener('click', () => {
        runBlendModels(container, featureNames, classes, classLabels);
    });

    // Stack button handler
    container.querySelector('#btn-stack').addEventListener('click', () => {
        runStackModels(container, featureNames, classes, classLabels);
    });

    // Finalize button handler
    container.querySelector('#btn-finalize').addEventListener('click', () => {
        runFinalizeModel(container, result, featureNames);
    });

    // Predict button handler
    container.querySelector('#btn-predict').addEventListener('click', () => {
        runPredictModel(container, result, featureNames);
    });

    evalSection.scrollIntoView({ behavior: 'smooth' });
}

function updateCreateModelParams(container, modelIndex) {
    const modelDef = MODELS[modelIndex];
    const paramFields = container.querySelector('#create-param-fields');
    const paramEntries = Object.entries(modelDef.params);

    if (paramEntries.length === 0) {
        paramFields.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.85rem;">このモデルにはデフォルトパラメータがありません。</p>';
        return;
    }

    paramFields.innerHTML = paramEntries.map(([key, defaultVal]) => `
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <label style="font-size: 0.85rem; min-width: 100px;">${key}:</label>
            <input type="number" id="create-param-${key}" class="form-select"
                   value="${defaultVal}" step="any"
                   style="width: 120px; font-size: 0.85rem;">
        </div>
    `).join('');
}

async function runCreateModel(container, featureNames, classes, classLabels) {
    const createResults = container.querySelector('#create-model-results');
    const btnCreate = container.querySelector('#btn-create-model');
    btnCreate.disabled = true;

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 3);

    createResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #b45309;"></i>
        <span style="margin-left: 0.5rem;">モデルを作成中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const modelIndex = parseInt(container.querySelector('#create-model-select').value);
        const modelDef = MODELS[modelIndex];

        // Collect custom params from input fields
        const customParams = {};
        for (const [key] of Object.entries(modelDef.params)) {
            const input = container.querySelector(`#create-param-${key}`);
            if (input && input.value !== '') {
                customParams[key] = parseFloat(input.value);
            }
        }

        const mergedParams = { ...modelDef.params, ...customParams };
        const model = new modelDef.cls(mergedParams);
        model.fit(_state.XTrain, _state.yTrain);

        const yPred = model.predict(_state.XTest);
        const yProba = model.predictProba ? model.predictProba(_state.XTest) : null;

        const cvScores = crossValidate(model, _state.XTrain, _state.yTrain, { cv: _state.cvFolds, scoring: 'f1' });
        const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;

        const acc = accuracy(_state.yTest, yPred);
        const prec = precisionScore(_state.yTest, yPred);
        const rec = recallScore(_state.yTest, yPred);
        const f1 = f1Score(_state.yTest, yPred);
        const cm = confusionMatrix(_state.yTest, yPred).matrix;

        // Store created model in state
        _state.createdModel = model;
        _state.createdModelResult = {
            name: modelDef.name,
            badge: modelDef.badge,
            cls: modelDef.cls,
            model,
            acc, prec, rec, f1, cm,
            cvMean,
            yPred, yProba,
            params: mergedParams
        };

        createResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4><i class="fas fa-check-circle" style="color: #10b981;"></i> ${modelDef.name} を作成しました</h4>
                <p style="color: var(--text-secondary); margin: 0.5rem 0;">
                    パラメータ: ${JSON.stringify(mergedParams)}
                </p>
                <div class="metrics-grid" style="margin: 1rem 0;">
                    ${createMetricCard('CV F1 (mean)', cvMean, `${_state.cvFolds}-Fold CV`)}
                    ${createMetricCard('Accuracy', acc, '正解率')}
                    ${createMetricCard('F1 Score', f1, 'F1スコア')}
                    ${createMetricCard('Precision', prec, '適合率')}
                    ${createMetricCard('Recall', rec, '再現率')}
                </div>
                <div id="create-model-cm-plot" style="margin-top: 1.5rem;"></div>
                <p style="color: #166534; font-weight: 600; margin-top: 1rem;">
                    <i class="fas fa-info-circle"></i> 作成したモデルは predict_model で使用できます（finalize / stack / blend が未実行の場合）。
                </p>
            </div>
        `;

        renderConfusionMatrix('create-model-cm-plot', cm, classLabels);

    } catch (error) {
        createResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> モデル作成エラー: ${error.message}</p>`;
        console.error('Create model error:', error);
    }

    btnCreate.disabled = false;
}

async function runTuneModel(container, result, featureNames, classes, classLabels) {
    const tuneResults = container.querySelector('#tune-results');
    const btnTune = container.querySelector('#btn-tune');
    btnTune.disabled = true;

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 4);

    tuneResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #0891b2;"></i>
        <span style="margin-left: 0.5rem;">GridSearch CV 実行中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const paramGrid = PARAM_GRIDS[result.badge];
        const { bestParams, bestScore, results: gsResults } = gridSearch(
            result.cls,
            paramGrid,
            _state.XTrain,
            _state.yTrain,
            { cv: _state.cvFolds, scoring: 'f1' }
        );

        // Train best model on full training data and evaluate on test
        const tunedModel = new result.cls(bestParams);
        tunedModel.fit(_state.XTrain, _state.yTrain);
        const yPredTuned = tunedModel.predict(_state.XTest);
        const tunedF1 = f1Score(_state.yTest, yPredTuned);
        const tunedAcc = accuracy(_state.yTest, yPredTuned);
        const tunedPrec = precisionScore(_state.yTest, yPredTuned);
        const tunedRec = recallScore(_state.yTest, yPredTuned);

        const improved = tunedF1 > result.f1;

        tuneResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4>GridSearch 結果</h4>
                <p><strong>Best Params:</strong> ${JSON.stringify(bestParams)}</p>
                <p><strong>Best CV F1 (mean):</strong> ${formatNumber(bestScore)}</p>

                <h4 style="margin-top: 1rem;">Before vs After</h4>
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr><th>指標</th><th>チューニング前</th><th>チューニング後</th><th>変化</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>CV F1 (mean)</td>
                                <td>${formatNumber(result.cvMean)}</td>
                                <td>${formatNumber(bestScore)}</td>
                                <td style="color: ${bestScore > result.cvMean ? '#10b981' : '#ef4444'};">
                                    ${bestScore > result.cvMean ? '+' : ''}${formatNumber(bestScore - result.cvMean)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test F1</td>
                                <td>${formatNumber(result.f1)}</td>
                                <td>${formatNumber(tunedF1)}</td>
                                <td style="color: ${tunedF1 > result.f1 ? '#10b981' : '#ef4444'};">
                                    ${tunedF1 > result.f1 ? '+' : ''}${formatNumber(tunedF1 - result.f1)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test Accuracy</td>
                                <td>${formatNumber(result.acc)}</td>
                                <td>${formatNumber(tunedAcc)}</td>
                                <td style="color: ${tunedAcc > result.acc ? '#10b981' : '#ef4444'};">
                                    ${tunedAcc > result.acc ? '+' : ''}${formatNumber(tunedAcc - result.acc)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test Precision</td>
                                <td>${formatNumber(result.prec)}</td>
                                <td>${formatNumber(tunedPrec)}</td>
                                <td style="color: ${tunedPrec > result.prec ? '#10b981' : '#ef4444'};">
                                    ${tunedPrec > result.prec ? '+' : ''}${formatNumber(tunedPrec - result.prec)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test Recall</td>
                                <td>${formatNumber(result.rec)}</td>
                                <td>${formatNumber(tunedRec)}</td>
                                <td style="color: ${tunedRec > result.rec ? '#10b981' : '#ef4444'};">
                                    ${tunedRec > result.rec ? '+' : ''}${formatNumber(tunedRec - result.rec)}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                ${improved
                    ? `<p style="color: #10b981; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-check-circle"></i> チューニングによりテスト性能が改善しました！チューニング済みモデルを使用します。
                      </p>`
                    : `<p style="color: #f59e0b; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-info-circle"></i> テスト性能は改善しませんでしたが、CV F1は参考になります。
                      </p>`
                }

                <details style="margin-top: 1rem;">
                    <summary style="cursor: pointer; font-weight: 600;">全パラメータ組み合わせ (${gsResults.length}通り)</summary>
                    <div class="table-container" style="margin-top: 0.5rem;">
                        <table class="table" style="font-size: 0.85rem;">
                            <thead>
                                <tr><th>順位</th><th>パラメータ</th><th>CV F1 (mean)</th></tr>
                            </thead>
                            <tbody>
                                ${gsResults.map((gs, i) => `
                                    <tr class="${i === 0 ? 'highlight-row' : ''}">
                                        <td>${i + 1}</td>
                                        <td>${JSON.stringify(gs.params)}</td>
                                        <td>${formatNumber(gs.meanScore)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </details>
            </div>
        `;

        // Update the model in result if improved
        if (improved) {
            result.model = tunedModel;
            result.yPred = yPredTuned;
            result.f1 = tunedF1;
            result.acc = tunedAcc;
            result.prec = tunedPrec;
            result.rec = tunedRec;
            result.cvMean = bestScore;
            result.featureImportance = tunedModel.getFeatureImportance ? tunedModel.getFeatureImportance() : null;

            // Recompute confusion matrix and proba
            result.cm = confusionMatrix(_state.yTest, yPredTuned).matrix;
            result.yProba = tunedModel.predictProba ? tunedModel.predictProba(_state.XTest) : null;
            if (result.yProba && classes.length === 2) {
                const positiveProba = result.yProba.map(p => p[1] || 0);
                result.auc = rocAucScore(_state.yTest, positiveProba);
            }
        }

    } catch (error) {
        tuneResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> チューニングエラー: ${error.message}</p>`;
        console.error('Tune error:', error);
    }

    btnTune.disabled = false;
}

async function runInterpretModel(container, result, featureNames) {
    const interpretResults = container.querySelector('#interpret-results');
    const btnInterpret = container.querySelector('#btn-interpret');
    btnInterpret.disabled = true;

    // Update step indicator
    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 5);

    interpretResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #16a34a;"></i>
        <span style="margin-left: 0.5rem;">モデル解釈を計算中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        let html = '<div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">';

        // 1. Permutation Feature Importance
        html += '<h4><i class="fas fa-sort-amount-down" style="color: #16a34a;"></i> Permutation Feature Importance</h4>';
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">各特徴量をシャッフルしてスコアへの影響を計測。モデルに依存しない汎用的な重要度指標です。</p>';
        html += '<div id="perm-importance-plot"></div>';

        // 2. PDP
        html += '<h4 style="margin-top: 2rem;"><i class="fas fa-chart-line" style="color: #16a34a;"></i> Partial Dependence Plot (PDP)</h4>';
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">各特徴量が予測確率にどう影響するかを可視化します。</p>';
        html += '<div style="margin-bottom: 1rem;"><label style="font-weight: 600; margin-right: 0.5rem;">特徴量を選択:</label>';
        html += `<select id="pdp-feature-select" class="form-select" style="display: inline-block; width: auto;">
            ${featureNames.map((f, i) => `<option value="${i}">${f}</option>`).join('')}
        </select></div>`;
        html += '<div id="pdp-plot"></div>';

        // 3. Learning Curve
        html += '<h4 style="margin-top: 2rem;"><i class="fas fa-graduation-cap" style="color: #16a34a;"></i> Learning Curve（学習曲線）</h4>';
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">訓練データ量と性能の関係を可視化。過学習・未学習の診断に使います。</p>';
        html += '<div id="learning-curve-plot"></div>';

        // 4. SHAP Analysis
        html += '<h4 style="margin-top: 2rem;"><i class="fas fa-puzzle-piece" style="color: #16a34a;"></i> SHAP (SHapley Additive exPlanations)</h4>';
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">ゲーム理論に基づく各特徴量の貢献度を計算し、モデルの予測を説明します。</p>';
        html += '<div id="shap-summary-plot"></div>';
        html += '<div id="shap-beeswarm-plot" style="margin-top: 1.5rem;"></div>';
        html += '<div id="shap-waterfall-plot" style="margin-top: 1.5rem;"></div>';

        // 5. Interpretation
        html += '<div id="interpret-analysis" style="margin-top: 2rem;"></div>';

        html += '</div>';
        interpretResults.innerHTML = html;

        // Compute permutation importance (using f1 scoring for classification)
        const { importancesMean, importancesStd } = permutationImportance(
            result.model, _state.XTest, _state.yTest,
            { scoring: 'f1', nRepeats: 5 }
        );
        renderPermutationImportance('perm-importance-plot', featureNames, importancesMean, importancesStd);

        // Compute PDP for the first feature initially
        computeAndRenderPDP(result.model, featureNames, 0);

        // PDP feature selector
        container.querySelector('#pdp-feature-select').addEventListener('change', (e) => {
            computeAndRenderPDP(result.model, featureNames, parseInt(e.target.value));
        });

        // Compute learning curve (using f1 scoring)
        const lcResult = learningCurve(
            result.cls, result.model.getParams ? result.model.getParams() : {},
            _state.XTrain, _state.yTrain,
            { cv: Math.min(_state.cvFolds, 3), scoring: 'f1' }
        );
        renderLearningCurve(
            'learning-curve-plot',
            lcResult.trainSizes,
            lcResult.trainScoresMean, lcResult.trainScoresStd,
            lcResult.testScoresMean, lcResult.testScoresStd,
            'F1 Score'
        );

        // Compute SHAP values
        computeAndRenderSHAP(result, featureNames);

        // Generate interpretation analysis
        const analysisHtml = generateInterpretAnalysis(featureNames, importancesMean, lcResult);
        container.querySelector('#interpret-analysis').innerHTML = analysisHtml;

    } catch (error) {
        interpretResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> 解釈エラー: ${error.message}</p>`;
        console.error('Interpret error:', error);
    }

    btnInterpret.disabled = false;
}

function computeAndRenderPDP(model, featureNames, featureIndex) {
    const X = _state.XTest;
    const nGrid = 30;

    // Get range of the feature
    const featureValues = X.map(row => row[featureIndex]);
    const minVal = Math.min(...featureValues);
    const maxVal = Math.max(...featureValues);
    const step = (maxVal - minVal) / (nGrid - 1);
    const gridValues = Array.from({ length: nGrid }, (_, i) => minVal + i * step);

    // For classification, use predictProba if available, otherwise use predict
    let pdpValues;
    if (model.predictProba) {
        pdpValues = gridValues.map(gridVal => {
            const XModified = X.map(row => {
                const newRow = [...row];
                newRow[featureIndex] = gridVal;
                return newRow;
            });
            const proba = model.predictProba(XModified);
            // Use probability of class 1 (or the last class)
            const classIdx = proba[0].length - 1;
            return proba.reduce((sum, p) => sum + (p[classIdx] || 0), 0) / proba.length;
        });
    } else {
        pdpValues = gridValues.map(gridVal => {
            const XModified = X.map(row => {
                const newRow = [...row];
                newRow[featureIndex] = gridVal;
                return newRow;
            });
            const preds = model.predict(XModified);
            return preds.reduce((a, b) => a + b, 0) / preds.length;
        });
    }

    renderPDP('pdp-plot', featureNames[featureIndex], gridValues, pdpValues, '#16a34a');
}

function computeAndRenderSHAP(result, featureNames) {
    const model = result.model;
    const XTest = _state.XTest;
    const XTrain = _state.XTrain;

    // Compute feature means from training data
    const nFeatures = featureNames.length;
    const featureMeans = Array(nFeatures).fill(0);
    for (const row of XTrain) {
        for (let f = 0; f < nFeatures; f++) {
            featureMeans[f] += row[f];
        }
    }
    for (let f = 0; f < nFeatures; f++) {
        featureMeans[f] /= XTrain.length;
    }

    let shapResult;

    // Use linearSHAP for LogisticRegression (has coefficients), kernelSHAP otherwise
    const isLogistic = model instanceof LogisticRegression;

    if (isLogistic && model.weights && model.weights.length > 0) {
        // For binary LogisticRegression, use linearSHAP with the weight vector
        const w = model.weights[0];
        const coefficients = w.slice(0, nFeatures);
        const intercept = w[nFeatures] || 0;
        shapResult = linearSHAP(coefficients, intercept, XTest, featureMeans);
    } else if (model.predictProba) {
        // Use probability of positive class as prediction function
        const predictFn = (X) => {
            const proba = model.predictProba(X);
            return proba.map(p => p[1] || 0);
        };
        // Use a subset of background samples for efficiency
        const bgSample = XTrain.length > 50 ? XTrain.slice(0, 50) : XTrain;
        shapResult = kernelSHAP(predictFn, XTest, bgSample);
    } else {
        // Map class labels to 0/1 for SHAP
        const classes = _state.classes;
        const predictFn = (X) => {
            const preds = model.predict(X);
            return preds.map(p => p === classes[classes.length - 1] ? 1 : 0);
        };
        const bgSample = XTrain.length > 50 ? XTrain.slice(0, 50) : XTrain;
        shapResult = kernelSHAP(predictFn, XTest, bgSample);
    }

    const { shapValues, baseValue } = shapResult;
    const { meanAbsSHAP, meanSHAP } = shapSummary(shapValues);

    // 1. SHAP Summary Bar
    renderSHAPSummary('shap-summary-plot', featureNames, meanAbsSHAP, meanSHAP);

    // 2. SHAP Beeswarm
    renderSHAPBeeswarm('shap-beeswarm-plot', featureNames, shapValues, XTest);

    // 3. SHAP Waterfall for first test sample
    const firstSHAP = shapValues[0];
    const firstPrediction = baseValue + firstSHAP.reduce((a, b) => a + b, 0);
    renderSHAPWaterfall('shap-waterfall-plot', featureNames, firstSHAP, baseValue, firstPrediction);
}

function generateInterpretAnalysis(featureNames, importancesMean, lcResult) {
    // Find most important features
    const indexed = importancesMean.map((v, i) => ({ name: featureNames[i], importance: v }));
    indexed.sort((a, b) => b.importance - a.importance);
    const topFeatures = indexed.filter(f => f.importance > 0).slice(0, 3);

    // Learning curve analysis
    const lastTrainScore = lcResult.trainScoresMean[lcResult.trainScoresMean.length - 1];
    const lastTestScore = lcResult.testScoresMean[lcResult.testScoresMean.length - 1];
    const gap = lastTrainScore - lastTestScore;

    let html = `
        <h4><i class="fas fa-lightbulb" style="color: #0891b2;"></i> 解釈の総合分析</h4>
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #16a34a; line-height: 1.8;">
            <p><strong>重要な特徴量:</strong></p>
            <ul>
    `;

    if (topFeatures.length > 0) {
        topFeatures.forEach((f, i) => {
            html += `<li><strong>${f.name}</strong> (重要度: ${f.importance.toFixed(4)}) - `;
            if (i === 0) html += 'このモデルで最も分類に寄与する特徴量です。';
            else html += '分類に重要な特徴量です。';
            html += '</li>';
        });
    } else {
        html += '<li>全ての特徴量の重要度が低い、またはモデルが十分に学習できていません。</li>';
    }

    html += '</ul><p style="margin-top: 1rem;"><strong>学習曲線の分析:</strong></p><ul>';

    if (gap > 0.15) {
        html += `<li style="color: #ef4444;"><strong>過学習の兆候</strong>: 訓練スコア (${lastTrainScore.toFixed(3)}) と検証スコア (${lastTestScore.toFixed(3)}) の差が大きい (${gap.toFixed(3)})。正則化の強化やデータ追加を検討してください。</li>`;
    } else if (gap > 0.05) {
        html += `<li style="color: #f59e0b;">訓練スコアと検証スコアにやや差があります (gap=${gap.toFixed(3)})。軽度の過学習の可能性。</li>`;
    } else {
        html += `<li style="color: #10b981;">訓練スコアと検証スコアが近く (gap=${gap.toFixed(3)})、<strong>良好な汎化性能</strong>を示しています。</li>`;
    }

    if (lastTestScore < 0.5) {
        html += '<li style="color: #ef4444;"><strong>未学習の兆候</strong>: 検証スコアが低いため、より複雑なモデルや特徴量エンジニアリングを検討してください。</li>';
    }

    const earlyTestScore = lcResult.testScoresMean[0];
    if (lastTestScore - earlyTestScore > 0.05) {
        html += '<li>データ量の増加でスコアが向上しているため、<strong>追加データ収集が効果的</strong>と考えられます。</li>';
    } else {
        html += '<li>データ量増加による改善が小さいため、特徴量の改善やモデルの変更がより効果的です。</li>';
    }

    html += '</ul></div>';
    return html;
}

async function runBlendModels(container, featureNames, classes, classLabels) {
    const blendResults = container.querySelector('#blend-results');
    const btnBlend = container.querySelector('#btn-blend');
    btnBlend.disabled = true;

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 6);

    blendResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #7c3aed;"></i>
        <span style="margin-left: 0.5rem;">モデルをブレンド中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const topN = parseInt(container.querySelector('#blend-top-n').value);
        const validResults = _state.results.filter(r => r.model);
        const topModels = validResults.slice(0, topN);

        // For classification, use majority voting
        const blendedPred = _state.yTest.map((_, i) => {
            const votes = {};
            topModels.forEach(m => {
                const pred = m.yPred[i];
                votes[pred] = (votes[pred] || 0) + 1;
            });
            // Return class with most votes
            let maxVotes = 0;
            let bestClass = topModels[0].yPred[i];
            for (const [cls, count] of Object.entries(votes)) {
                if (count > maxVotes) {
                    maxVotes = count;
                    bestClass = Number(cls);
                }
            }
            return bestClass;
        });

        const blendedF1 = f1Score(_state.yTest, blendedPred);
        const blendedAcc = accuracy(_state.yTest, blendedPred);
        const blendedPrec = precisionScore(_state.yTest, blendedPred);
        const blendedRec = recallScore(_state.yTest, blendedPred);
        const bestSingleF1 = topModels[0].f1;

        // Create a blended model proxy for predict
        const blendedModel = {
            models: topModels.map(m => m.model),
            predict(X) {
                return X.map((_, i) => {
                    const votes = {};
                    this.models.forEach(m => {
                        const preds = m.predict(X);
                        const pred = preds[i];
                        votes[pred] = (votes[pred] || 0) + 1;
                    });
                    let maxVotes = 0;
                    let bestClass = 0;
                    for (const [cls, count] of Object.entries(votes)) {
                        if (count > maxVotes) {
                            maxVotes = count;
                            bestClass = Number(cls);
                        }
                    }
                    return bestClass;
                });
            },
            predictProba(X) {
                const modelsWithProba = this.models.filter(m => m.predictProba);
                if (modelsWithProba.length === 0) return null;
                const probas = modelsWithProba.map(m => m.predictProba(X));
                return X.map((_, i) => {
                    const nClasses = probas[0][i].length;
                    const avgProba = Array(nClasses).fill(0);
                    probas.forEach(p => {
                        for (let c = 0; c < nClasses; c++) {
                            avgProba[c] += (p[i][c] || 0);
                        }
                    });
                    return avgProba.map(v => v / modelsWithProba.length);
                });
            },
            getParams() { return { type: 'blend', nModels: this.models.length }; }
        };

        _state.blendedModel = blendedModel;

        blendResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4>ブレンド結果 (上位 ${topN} モデルの多数決投票)</h4>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    使用モデル: ${topModels.map(m => m.badge).join(', ')}
                </p>
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr><th>指標</th><th>ベストモデル (${topModels[0].badge})</th><th>ブレンドモデル</th><th>変化</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Test F1</td>
                                <td>${formatNumber(bestSingleF1)}</td>
                                <td>${formatNumber(blendedF1)}</td>
                                <td style="color: ${blendedF1 > bestSingleF1 ? '#10b981' : '#ef4444'};">
                                    ${blendedF1 > bestSingleF1 ? '+' : ''}${formatNumber(blendedF1 - bestSingleF1)}
                                </td>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>${formatNumber(topModels[0].acc)}</td>
                                <td>${formatNumber(blendedAcc)}</td>
                                <td style="color: ${blendedAcc > topModels[0].acc ? '#10b981' : '#ef4444'};">
                                    ${blendedAcc > topModels[0].acc ? '+' : ''}${formatNumber(blendedAcc - topModels[0].acc)}
                                </td>
                            </tr>
                            <tr>
                                <td>Precision</td>
                                <td>${formatNumber(topModels[0].prec)}</td>
                                <td>${formatNumber(blendedPrec)}</td>
                                <td style="color: ${blendedPrec > topModels[0].prec ? '#10b981' : '#ef4444'};">
                                    ${blendedPrec > topModels[0].prec ? '+' : ''}${formatNumber(blendedPrec - topModels[0].prec)}
                                </td>
                            </tr>
                            <tr>
                                <td>Recall</td>
                                <td>${formatNumber(topModels[0].rec)}</td>
                                <td>${formatNumber(blendedRec)}</td>
                                <td style="color: ${blendedRec > topModels[0].rec ? '#10b981' : '#ef4444'};">
                                    ${blendedRec > topModels[0].rec ? '+' : ''}${formatNumber(blendedRec - topModels[0].rec)}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                ${blendedF1 > bestSingleF1
                    ? `<p style="color: #10b981; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-check-circle"></i> ブレンドモデルがベストモデルを上回りました！predict_model でブレンドモデルを使用できます。
                      </p>`
                    : `<p style="color: #f59e0b; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-info-circle"></i> ブレンドモデルはベストモデルを上回りませんでした。単体モデルの方が適している可能性があります。
                      </p>`
                }
            </div>
        `;

        // Add confusion matrix for blended model
        const blendedCM = confusionMatrix(_state.yTest, blendedPred);
        blendResults.querySelector('div').insertAdjacentHTML('beforeend', '<div id="blend-cm-plot" style="margin-top: 1.5rem;"></div>');
        renderConfusionMatrix('blend-cm-plot', blendedCM.matrix, classLabels);

    } catch (error) {
        blendResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> ブレンドエラー: ${error.message}</p>`;
        console.error('Blend error:', error);
    }

    btnBlend.disabled = false;
}

async function runStackModels(container, featureNames, classes, classLabels) {
    const stackResults = container.querySelector('#stack-results');
    const btnStack = container.querySelector('#btn-stack');
    btnStack.disabled = true;

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 7);

    stackResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #be185d;"></i>
        <span style="margin-left: 0.5rem;">モデルをスタッキング中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const topN = parseInt(container.querySelector('#stack-top-n').value);
        const validResults = _state.results.filter(r => r.model);
        const topModels = validResults.slice(0, topN);

        // Build stacked features for training data
        const stackedTrainFeatures = _state.XTrain.map((_, sampleIdx) => {
            const features = [];
            for (const mr of topModels) {
                const sampleInput = [_state.XTrain[sampleIdx]];
                if (mr.model.predictProba) {
                    const proba = mr.model.predictProba(sampleInput);
                    features.push(...proba[0]);
                } else {
                    // One-hot encode predictions
                    const pred = mr.model.predict(sampleInput)[0];
                    const oneHot = classes.map(c => c === pred ? 1 : 0);
                    features.push(...oneHot);
                }
            }
            return features;
        });

        // Build stacked features for test data
        const stackedTestFeatures = _state.XTest.map((_, sampleIdx) => {
            const features = [];
            for (const mr of topModels) {
                const sampleInput = [_state.XTest[sampleIdx]];
                if (mr.model.predictProba) {
                    const proba = mr.model.predictProba(sampleInput);
                    features.push(...proba[0]);
                } else {
                    const pred = mr.model.predict(sampleInput)[0];
                    const oneHot = classes.map(c => c === pred ? 1 : 0);
                    features.push(...oneHot);
                }
            }
            return features;
        });

        // Train meta-learner (LogisticRegression) on stacked features
        const metaLearner = new LogisticRegression({ maxIter: 1000 });
        metaLearner.fit(stackedTrainFeatures, _state.yTrain);

        // Evaluate on test data
        const stackedPred = metaLearner.predict(stackedTestFeatures);
        const stackedF1 = f1Score(_state.yTest, stackedPred);
        const stackedAcc = accuracy(_state.yTest, stackedPred);
        const stackedPrec = precisionScore(_state.yTest, stackedPred);
        const stackedRec = recallScore(_state.yTest, stackedPred);
        const bestSingleF1 = topModels[0].f1;

        // Create stacked model proxy for predict
        const stackedModel = {
            baseModels: topModels.map(m => m.model),
            metaLearner,
            _classes: classes,
            predict(X) {
                const stackedFeatures = X.map(row => {
                    const features = [];
                    for (const bm of this.baseModels) {
                        if (bm.predictProba) {
                            const proba = bm.predictProba([row]);
                            features.push(...proba[0]);
                        } else {
                            const pred = bm.predict([row])[0];
                            const oneHot = this._classes.map(c => c === pred ? 1 : 0);
                            features.push(...oneHot);
                        }
                    }
                    return features;
                });
                return this.metaLearner.predict(stackedFeatures);
            },
            predictProba(X) {
                const stackedFeatures = X.map(row => {
                    const features = [];
                    for (const bm of this.baseModels) {
                        if (bm.predictProba) {
                            const proba = bm.predictProba([row]);
                            features.push(...proba[0]);
                        } else {
                            const pred = bm.predict([row])[0];
                            const oneHot = this._classes.map(c => c === pred ? 1 : 0);
                            features.push(...oneHot);
                        }
                    }
                    return features;
                });
                return this.metaLearner.predictProba(stackedFeatures);
            },
            getParams() { return { type: 'stack', nModels: this.baseModels.length, meta: 'LogisticRegression' }; }
        };

        _state.stackedModel = stackedModel;

        stackResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4>スタッキング結果 (上位 ${topN} モデル + LogisticRegression メタ学習器)</h4>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    ベースモデル: ${topModels.map(m => m.badge).join(', ')} / メタ学習器: LogisticRegression
                </p>
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr><th>指標</th><th>ベストモデル (${topModels[0].badge})</th><th>スタッキングモデル</th><th>変化</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Test F1</td>
                                <td>${formatNumber(bestSingleF1)}</td>
                                <td>${formatNumber(stackedF1)}</td>
                                <td style="color: ${stackedF1 > bestSingleF1 ? '#10b981' : '#ef4444'};">
                                    ${stackedF1 > bestSingleF1 ? '+' : ''}${formatNumber(stackedF1 - bestSingleF1)}
                                </td>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>${formatNumber(topModels[0].acc)}</td>
                                <td>${formatNumber(stackedAcc)}</td>
                                <td style="color: ${stackedAcc > topModels[0].acc ? '#10b981' : '#ef4444'};">
                                    ${stackedAcc > topModels[0].acc ? '+' : ''}${formatNumber(stackedAcc - topModels[0].acc)}
                                </td>
                            </tr>
                            <tr>
                                <td>Precision</td>
                                <td>${formatNumber(topModels[0].prec)}</td>
                                <td>${formatNumber(stackedPrec)}</td>
                                <td style="color: ${stackedPrec > topModels[0].prec ? '#10b981' : '#ef4444'};">
                                    ${stackedPrec > topModels[0].prec ? '+' : ''}${formatNumber(stackedPrec - topModels[0].prec)}
                                </td>
                            </tr>
                            <tr>
                                <td>Recall</td>
                                <td>${formatNumber(topModels[0].rec)}</td>
                                <td>${formatNumber(stackedRec)}</td>
                                <td style="color: ${stackedRec > topModels[0].rec ? '#10b981' : '#ef4444'};">
                                    ${stackedRec > topModels[0].rec ? '+' : ''}${formatNumber(stackedRec - topModels[0].rec)}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                ${stackedF1 > bestSingleF1
                    ? `<p style="color: #10b981; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-check-circle"></i> スタッキングモデルがベストモデルを上回りました！predict_model でスタッキングモデルを使用できます。
                      </p>`
                    : `<p style="color: #f59e0b; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-info-circle"></i> スタッキングモデルはベストモデルを上回りませんでした。単体モデルやブレンドモデルの方が適している可能性があります。
                      </p>`
                }
            </div>
        `;

        // Add confusion matrix for stacked model
        const stackedCM = confusionMatrix(_state.yTest, stackedPred);
        stackResults.querySelector('div').insertAdjacentHTML('beforeend', '<div id="stack-cm-plot" style="margin-top: 1.5rem;"></div>');
        renderConfusionMatrix('stack-cm-plot', stackedCM.matrix, classLabels);

    } catch (error) {
        stackResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> スタッキングエラー: ${error.message}</p>`;
        console.error('Stack error:', error);
    }

    btnStack.disabled = false;
}

async function runFinalizeModel(container, result, featureNames) {
    const finalizeResults = container.querySelector('#finalize-results');
    const btnFinalize = container.querySelector('#btn-finalize');
    btnFinalize.disabled = true;

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 8);

    finalizeResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #dc2626;"></i>
        <span style="margin-left: 0.5rem;">全データで再学習中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const XFull = [..._state.XTrain, ..._state.XTest];
        const yFull = [..._state.yTrain, ..._state.yTest];

        const params = result.model.getParams ? result.model.getParams() : {};
        const finalModel = new result.cls(params);
        finalModel.fit(XFull, yFull);

        _state.finalizedModel = finalModel;
        _state.isFinalized = true;

        const cvScores = crossValidate(finalModel, XFull, yFull, { cv: _state.cvFolds, scoring: 'f1' });
        const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
        const cvStd = Math.sqrt(cvScores.reduce((a, v) => a + (v - cvMean) ** 2, 0) / cvScores.length);

        finalizeResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4><i class="fas fa-check-circle" style="color: #10b981;"></i> モデル確定完了</h4>
                <p style="margin: 1rem 0;">
                    <strong>${result.name}</strong> を全データ (${XFull.length} サンプル) で再学習しました。
                </p>
                <div class="metrics-grid" style="margin: 1rem 0;">
                    ${createMetricCard('学習サンプル数', XFull.length, '訓練+テストの全データ')}
                    ${createMetricCard('CV F1 (mean)', cvMean, `${_state.cvFolds}-Fold 全データCV`)}
                    ${createMetricCard('CV F1 (std)', cvStd, '交差検証の標準偏差')}
                </div>
                <div style="background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                    <p style="color: #166534;">
                        <i class="fas fa-info-circle"></i>
                        これは本番用モデルです。predict_model では確定済みモデルで予測を行います。
                        テストデータがなくなるため、テスト評価は行えませんが、CVスコアが参考になります。
                    </p>
                </div>
                ${createDownloadButton('dl-model-json', 'モデルをJSONダウンロード')}
            </div>
        `;

        // Model download handler
        const dlModelBtn = finalizeResults.querySelector('#dl-model-json');
        if (dlModelBtn) {
            dlModelBtn.addEventListener('click', () => {
                const exportData = serializeModel(
                    { model: finalModel, name: result.name, badge: result.badge },
                    { featureNames: _state.featureNames, scaler: _state.scaler, encoders: _state.encoders, labelEncoder: _state.labelEncoder, classLabels: _state.classLabels, targetCol: _state.targetCol, fileName: _state.fileName, taskType: 'classification' }
                );
                downloadJSON(exportData, makeModelFileName(_state.fileName, result.badge, 'classification'));
            });
        }

    } catch (error) {
        finalizeResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> 確定エラー: ${error.message}</p>`;
        console.error('Finalize error:', error);
    }

    btnFinalize.disabled = false;
}

function runPredictModel(container, result, featureNames) {
    const predictResult = container.querySelector('#predict-result');

    // Collect input values
    const inputValues = [];
    let hasEmpty = false;
    for (const f of featureNames) {
        const input = container.querySelector(`#pred-${f}`);
        if (!input || input.value === '') {
            hasEmpty = true;
            break;
        }
        inputValues.push(parseFloat(input.value));
    }

    if (hasEmpty) {
        predictResult.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> すべての特徴量に値を入力してください。</p>`;
        return;
    }

    if (inputValues.some(v => isNaN(v))) {
        predictResult.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> 数値を正しく入力してください。</p>`;
        return;
    }

    try {
        // Apply scaler if used during training
        let processedInput = [inputValues];
        if (_state.scaler) {
            processedInput = _state.scaler.transform(processedInput);
        }

        // Use finalized > stacked > blended > created > best model
        const activeModel = _state.finalizedModel || _state.stackedModel || _state.blendedModel
                         || (_state.createdModel ? _state.createdModel : result.model);
        const modelLabel = _state.finalizedModel ? `${result.name} (確定済み)`
                         : _state.stackedModel ? 'スタッキングモデル'
                         : _state.blendedModel ? 'ブレンドモデル'
                         : _state.createdModel ? `${_state.createdModelResult.name} (作成済み)`
                         : result.name;
        const prediction = activeModel.predict(processedInput);
        const proba = activeModel.predictProba ? activeModel.predictProba(processedInput) : null;

        // Update step indicator
        container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 9);

        // Convert numeric prediction back to label
        const predictedLabel = _state.labelEncoder
            ? _state.labelEncoder.inverseTransform(prediction)[0]
            : prediction[0];

        let probaHtml = '';
        if (proba && proba[0]) {
            probaHtml = `
                <div style="margin-top: 1rem;">
                    <p style="font-weight: 600; margin-bottom: 0.5rem;">クラス別確率:</p>
                    ${_state.classLabels.map((label, i) => {
                        const p = proba[0][i] || 0;
                        return `<div style="display: flex; align-items: center; margin: 0.25rem 0;">
                            <span style="width: 80px; font-size: 0.85rem;">${label}</span>
                            <div style="flex: 1; background: #e2e8f0; border-radius: 4px; height: 20px; margin: 0 0.5rem;">
                                <div style="width: ${(p * 100).toFixed(1)}%; background: #0891b2; border-radius: 4px; height: 100%;"></div>
                            </div>
                            <span style="font-size: 0.85rem; width: 50px;">${(p * 100).toFixed(1)}%</span>
                        </div>`;
                    }).join('')}
                </div>
            `;
        }

        predictResult.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: center;">
                <p style="font-size: 0.9rem; color: var(--text-secondary);">予測結果 (${modelLabel})</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #0891b2; margin: 0.5rem 0;">
                    ${predictedLabel}
                </p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">
                    目的変数: ${_state.targetCol}
                </p>
                ${probaHtml}
            </div>
            ${createDownloadButton('dl-predict-csv', '予測結果をCSVダウンロード')}
        `;

        const dlPredBtn = predictResult.querySelector('#dl-predict-csv');
        if (dlPredBtn) {
            dlPredBtn.addEventListener('click', () => {
                const headers = ['項目', '値'];
                const rows = [
                    ...featureNames.map((f, i) => [f, inputValues[i]]),
                    ['予測クラス', predictedLabel]
                ];
                if (proba && proba[0]) {
                    _state.classLabels.forEach((label, i) => {
                        const p = proba[0][i] || 0;
                        rows.push([`確率: ${label}`, `${(p * 100).toFixed(1)}%`]);
                    });
                }
                const csv = toCSV(headers, rows);
                downloadCSV(csv, makeExportFileName(_state.fileName, '分類_予測結果'));
            });
        }
    } catch (error) {
        predictResult.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> 予測エラー: ${error.message}</p>`;
        console.error('Predict error:', error);
    }
}

function interpretResults(result) {
    const f1 = result.f1;
    let interpretation = `<p><strong>${result.name}</strong> の評価結果:</p><ul>`;

    if (f1 >= 0.9) {
        interpretation += `<li>F1 = ${formatNumber(f1)} : <strong style="color: #10b981;">非常に高い分類精度</strong>です。</li>`;
    } else if (f1 >= 0.7) {
        interpretation += `<li>F1 = ${formatNumber(f1)} : <strong style="color: #3b82f6;">良好な分類精度</strong>です。</li>`;
    } else if (f1 >= 0.5) {
        interpretation += `<li>F1 = ${formatNumber(f1)} : <strong style="color: #f59e0b;">中程度の分類精度</strong>です。特徴量の改善を検討してください。</li>`;
    } else {
        interpretation += `<li>F1 = ${formatNumber(f1)} : <strong style="color: #ef4444;">分類精度が低い</strong>です。データやモデルの見直しが必要です。</li>`;
    }

    // CV interpretation
    interpretation += `<li>CV F1 = ${formatNumber(result.cvMean)} ± ${formatNumber(result.cvStd)} : `;
    if (Math.abs(result.cvMean - result.f1) < 0.1) {
        interpretation += `交差検証とテストの差が小さく、<strong style="color: #10b981;">安定したモデル</strong>です。</li>`;
    } else if (result.cvMean > result.f1) {
        interpretation += `CVがテストF1より高く、<strong style="color: #f59e0b;">テストデータにやや弱い</strong>可能性があります。</li>`;
    } else {
        interpretation += `CVよりテストF1が高い結果です。</li>`;
    }

    interpretation += `<li>Accuracy = ${formatNumber(result.acc)} : 全体の ${(result.acc * 100).toFixed(1)}% を正しく分類できました。</li>`;
    interpretation += `<li>Precision = ${formatNumber(result.prec)} : 正と予測したものの ${(result.prec * 100).toFixed(1)}% が実際に正でした。</li>`;
    interpretation += `<li>Recall = ${formatNumber(result.rec)} : 実際に正のものの ${(result.rec * 100).toFixed(1)}% を検出できました。</li>`;

    if (result.auc != null) {
        interpretation += `<li>AUC = ${formatNumber(result.auc)} : ROC曲線下面積で、1.0に近いほどランダムより優れた分類です。</li>`;
    }

    interpretation += `</ul>`;
    return interpretation;
}
