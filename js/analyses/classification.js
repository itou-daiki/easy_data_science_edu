// ==========================================
// 分類モデル比較 (AutoML) Module
// PyCaret-style: setup → compare_models (CV) → tune_model → predict_model
// ==========================================
import { createSelect, createStepIndicator, formatNumber, renderPlot, renderConfusionMatrix, renderROCCurve, renderFeatureImportance, createMetricCard } from '../utils.js';
import { prepareFeatures } from '../ml/preprocessing.js';
import { trainTestSplit, crossValidate, gridSearch } from '../ml/model_selection.js';
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

        ${createStepIndicator(['Setup', 'Compare', 'Tune', 'Predict'], 0)}

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

    container.querySelector('.step-indicator').innerHTML = createStepIndicator(['Setup', 'Compare', 'Tune', 'Predict'], 1).replace(/<\/?div[^>]*class="step-indicator"[^>]*>/g, '');

    const progressArea = container.querySelector('#progress-area');
    progressArea.innerHTML = `<div style="text-align: center; padding: 2rem;">
        <i class="fas fa-spinner fa-spin fa-2x" style="color: #0891b2;"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
        <div id="model-progress" style="margin-top: 1rem;"></div>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const { X, y, featureNames, labelEncoder, scaler } = prepareFeatures(data, targetCol, {
            selectedFeatures,
            task: 'classification'
        });

        const classes = [...new Set(y)].sort((a, b) => a - b);
        const classLabels = labelEncoder ? classes.map(c => labelEncoder.inverseTransform([c])[0]) : classes.map(String);

        const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize, randomState: 42 });

        // Save state for tune/predict
        _state = { XTrain, XTest, yTrain, yTest, featureNames, scaler, labelEncoder, classes, classLabels, cvFolds, targetCol };

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
    `;

    comparisonDiv.innerHTML = html;
    container.querySelector('#progress-area').innerHTML = '';

    comparisonDiv.querySelectorAll('.btn-detail').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.index);
            showModelDetail(container, results[idx], yTest, featureNames, classes, classLabels);
        });
    });

    const bestModel = results.find(r => r.model);
    if (bestModel) {
        showModelDetail(container, bestModel, yTest, featureNames, classes, classLabels);
    }
}

function showModelDetail(container, result, yTest, featureNames, classes, classLabels) {
    const evalSection = container.querySelector('#evaluate-section');
    evalSection.style.display = 'block';

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(['Setup', 'Compare', 'Tune', 'Predict'], 2);

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

    // Tune button handler
    if (hasTuneGrid) {
        container.querySelector('#btn-tune').addEventListener('click', () => {
            runTuneModel(container, result, featureNames, classes, classLabels);
        });
    }

    // Predict button handler
    container.querySelector('#btn-predict').addEventListener('click', () => {
        runPredictModel(container, result, featureNames);
    });

    evalSection.scrollIntoView({ behavior: 'smooth' });
}

async function runTuneModel(container, result, featureNames, classes, classLabels) {
    const tuneResults = container.querySelector('#tune-results');
    const btnTune = container.querySelector('#btn-tune');
    btnTune.disabled = true;

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

        const prediction = result.model.predict(processedInput);
        const proba = result.model.predictProba ? result.model.predictProba(processedInput) : null;

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
                <p style="font-size: 0.9rem; color: var(--text-secondary);">予測結果 (${result.name})</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #0891b2; margin: 0.5rem 0;">
                    ${predictedLabel}
                </p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">
                    目的変数: ${_state.targetCol}
                </p>
                ${probaHtml}
            </div>
        `;
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
