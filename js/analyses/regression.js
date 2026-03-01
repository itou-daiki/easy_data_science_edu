// ==========================================
// 回帰モデル比較 (AutoML) Module
// PyCaret-style: setup → compare_models (CV) → tune_model → predict_model
// ==========================================
import { createSelect, createStepIndicator, formatNumber, renderPlot, renderActualVsPredicted, renderResidualPlot, renderFeatureImportance, createMetricCard, renderPermutationImportance, renderPDP, renderLearningCurve } from '../utils.js';
import { prepareFeatures } from '../ml/preprocessing.js';
import { trainTestSplit, crossValidate, gridSearch, permutationImportance, learningCurve } from '../ml/model_selection.js';
import { meanAbsoluteError, meanSquaredError, rootMeanSquaredError, rSquared, adjustedRSquared } from '../ml/metrics.js';
import { LinearRegression } from '../ml/regression/linear.js';
import { RidgeRegression } from '../ml/regression/ridge.js';
import { LassoRegression } from '../ml/regression/lasso.js';
import { DecisionTreeRegressor } from '../ml/regression/decision_tree.js';
import { RandomForestRegressor } from '../ml/regression/random_forest.js';
import { KNNRegressor } from '../ml/regression/knn.js';
import { GradientBoostingRegressor } from '../ml/regression/gradient_boosting.js';

const MODELS = [
    { name: '線形回帰', cls: LinearRegression, params: {}, badge: 'Linear' },
    { name: 'Ridge回帰', cls: RidgeRegression, params: { alpha: 1.0 }, badge: 'Ridge' },
    { name: 'Lasso回帰', cls: LassoRegression, params: { alpha: 1.0 }, badge: 'Lasso' },
    { name: '決定木', cls: DecisionTreeRegressor, params: { maxDepth: 5 }, badge: 'Tree' },
    { name: 'ランダムフォレスト', cls: RandomForestRegressor, params: { nEstimators: 100, maxDepth: 8, maxFeatures: null }, badge: 'RF' },
    { name: 'K近傍法', cls: KNNRegressor, params: { nNeighbors: 5 }, badge: 'KNN' },
    { name: '勾配ブースティング', cls: GradientBoostingRegressor, params: { nEstimators: 100, learningRate: 0.1, maxDepth: 3 }, badge: 'GBM' }
];

const PARAM_GRIDS = {
    'Ridge': { alpha: [0.01, 0.1, 1.0, 10.0] },
    'Lasso': { alpha: [0.001, 0.01, 0.1, 1.0] },
    'Tree': { maxDepth: [3, 5, 8, 10] },
    'RF': { nEstimators: [50, 100], maxDepth: [5, 8] },
    'KNN': { nNeighbors: [3, 5, 7, 9] },
    'GBM': { nEstimators: [50, 100], learningRate: [0.05, 0.1], maxDepth: [3, 5] },
};

// Module-level state for sharing data between steps
let _state = {};

export function render(container, data, characteristics) {
    _state = {};
    const numCols = characteristics.numericColumns;

    container.innerHTML = `
        <h2><i class="fas fa-robot" style="color: #d97706;"></i> 回帰モデル比較 (AutoML)</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            PyCaret のように複数の回帰モデルを一括学習・比較し、最適なモデルを見つけます。
        </p>

        ${createStepIndicator(['Setup', 'Compare', 'Tune', 'Interpret', 'Predict'], 0)}

        <div id="setup-section" class="model-config">
            <h3><i class="fas fa-cog"></i> Step 1: セットアップ</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">
                        目的変数（予測したい数値変数）:
                    </label>
                    ${createSelect('target-select', numCols, '目的変数を選択')}
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
                    使用する特徴量（チェックを外すと除外）:
                </label>
                <div id="feature-chips" class="variable-chips"></div>
            </div>
            <button id="btn-compare" class="btn-analysis" style="background: #d97706;" disabled>
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

    targetSelect.addEventListener('change', () => {
        const target = targetSelect.value;
        if (!target) {
            featureSelection.style.display = 'none';
            btnCompare.disabled = true;
            return;
        }

        const features = numCols.filter(c => c !== target);
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

    container.querySelector('.step-indicator').innerHTML = createStepIndicator(['Setup', 'Compare', 'Tune', 'Interpret', 'Predict'], 1).replace(/<\/?div[^>]*class="step-indicator"[^>]*>/g, '');

    const progressArea = container.querySelector('#progress-area');
    progressArea.innerHTML = `<div style="text-align: center; padding: 2rem;">
        <i class="fas fa-spinner fa-spin fa-2x" style="color: #d97706;"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
        <div id="model-progress" style="margin-top: 1rem;"></div>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const { X, y, featureNames, scaler } = prepareFeatures(data, targetCol, {
            selectedFeatures,
            task: 'regression'
        });

        const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize, randomState: 42 });

        // Save state for tune/predict
        _state = { XTrain, XTest, yTrain, yTest, featureNames, scaler, cvFolds, targetCol };

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

                // Cross-validation on training data
                const cvScores = crossValidate(model, XTrain, yTrain, { cv: cvFolds, scoring: 'r2' });
                const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
                const cvStd = Math.sqrt(cvScores.reduce((a, v) => a + (v - cvMean) ** 2, 0) / cvScores.length);

                const mae = meanAbsoluteError(yTest, yPred);
                const mse = meanSquaredError(yTest, yPred);
                const rmse = rootMeanSquaredError(yTest, yPred);
                const r2 = rSquared(yTest, yPred);
                const adjR2 = adjustedRSquared(yTest, yPred, featureNames.length);

                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    cls: modelDef.cls,
                    model,
                    mae, mse, rmse, r2, adjR2,
                    cvMean, cvStd, cvScores,
                    yPred,
                    featureImportance: model.getFeatureImportance ? model.getFeatureImportance() : null
                });
            } catch (err) {
                console.error(`${modelDef.name} failed:`, err);
                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    cls: modelDef.cls,
                    model: null,
                    mae: Infinity, mse: Infinity, rmse: Infinity, r2: -Infinity, adjR2: -Infinity,
                    cvMean: -Infinity, cvStd: 0, cvScores: [],
                    yPred: null,
                    error: err.message
                });
            }
        }

        // Sort by CV mean R² (PyCaret-style)
        results.sort((a, b) => b.cvMean - a.cvMean);
        _state.results = results;

        renderComparisonResults(container, results, yTest, featureNames);
    } catch (error) {
        progressArea.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-triangle"></i> エラー: ${error.message}</p>`;
        console.error(error);
    }
}

function renderComparisonResults(container, results, yTest, featureNames) {
    const comparisonDiv = container.querySelector('#comparison-results');

    const bestCV = Math.max(...results.filter(r => r.model).map(r => r.cvMean));

    let html = `
        <h3 style="margin-top: 1rem;"><i class="fas fa-trophy" style="color: #d97706;"></i> モデル比較結果</h3>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            ${_state.cvFolds}-Fold 交差検証スコア（訓練データ）でソートしています。テストデータ (${yTest.length} サンプル) での評価結果も併記。
        </p>
        <div class="table-container">
            <table class="table model-comparison-table">
                <thead>
                    <tr>
                        <th>順位</th>
                        <th>モデル</th>
                        <th>CV R² (mean)</th>
                        <th>CV R² (std)</th>
                        <th>Test R²</th>
                        <th>MAE</th>
                        <th>RMSE</th>
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
                                ${r.cvMean === bestCV && r.model ? ' <i class="fas fa-crown" style="color: #d97706;"></i>' : ''}
                            </td>
                            <td><strong>${r.model ? formatNumber(r.cvMean) : '-'}</strong></td>
                            <td>${r.model ? formatNumber(r.cvStd) : '-'}</td>
                            <td>${r.model ? formatNumber(r.r2) : '-'}</td>
                            <td>${r.model ? formatNumber(r.mae) : '<span style="color:#ef4444;">エラー</span>'}</td>
                            <td>${r.model ? formatNumber(r.rmse) : '-'}</td>
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
            showModelDetail(container, results[idx], yTest, featureNames);
        });
    });

    // Auto-show best model detail
    const bestModel = results.find(r => r.model);
    if (bestModel) {
        showModelDetail(container, bestModel, yTest, featureNames);
    }
}

function showModelDetail(container, result, yTest, featureNames) {
    const evalSection = container.querySelector('#evaluate-section');
    evalSection.style.display = 'block';

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(['Setup', 'Compare', 'Tune', 'Interpret', 'Predict'], 2);

    const evalContent = container.querySelector('#evaluation-content');
    const hasTuneGrid = PARAM_GRIDS[result.badge] != null;

    evalContent.innerHTML = `
        <h3><i class="fas fa-chart-bar" style="color: #d97706;"></i> ${result.name} の詳細評価</h3>

        <div class="metrics-grid" style="margin: 1.5rem 0;">
            ${createMetricCard('CV R² (mean)', result.cvMean, `${_state.cvFolds}-Fold 交差検証平均`)}
            ${createMetricCard('CV R² (std)', result.cvStd, '交差検証の標準偏差')}
            ${createMetricCard('Test R²', result.r2, 'テストデータ決定係数')}
            ${createMetricCard('MAE', result.mae, '平均絶対誤差')}
            ${createMetricCard('RMSE', result.rmse, '二乗平均平方根誤差')}
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 2rem;">
            <div><div id="actual-vs-pred-plot"></div></div>
            <div><div id="residual-plot"></div></div>
        </div>

        ${result.featureImportance ? '<div id="feature-importance-plot" style="margin-top: 2rem;"></div>' : ''}

        <div style="margin-top: 2rem;">
            <h4>結果の解釈</h4>
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #d97706; line-height: 1.8;">
                ${interpretResults(result)}
            </div>
        </div>

        <!-- Tune Model Section -->
        ${hasTuneGrid ? `
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 12px;">
            <h4><i class="fas fa-sliders-h" style="color: #d97706;"></i> tune_model - ハイパーパラメータチューニング</h4>
            <p style="color: #92400e; margin: 0.5rem 0;">
                GridSearch CV でパラメータを最適化します。
                探索範囲: ${Object.entries(PARAM_GRIDS[result.badge]).map(([k, v]) => `${k}=[${v.join(', ')}]`).join(', ')}
            </p>
            <button id="btn-tune" class="btn-analysis" style="background: #d97706; margin-top: 1rem;">
                <i class="fas fa-magic"></i> tune_model を実行
            </button>
            <div id="tune-results" style="margin-top: 1rem;"></div>
        </div>
        ` : ''}

        <!-- Interpret Model Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-radius: 12px;">
            <h4><i class="fas fa-search-plus" style="color: #16a34a;"></i> interpret_model - モデル解釈</h4>
            <p style="color: #166534; margin: 0.5rem 0;">
                Permutation Feature Importance、Partial Dependence Plot (PDP)、Learning Curve でモデルを深く理解します。
            </p>
            <button id="btn-interpret" class="btn-analysis" style="background: #16a34a; margin-top: 1rem;">
                <i class="fas fa-microscope"></i> interpret_model を実行
            </button>
            <div id="interpret-results" style="margin-top: 1rem;"></div>
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

    renderActualVsPredicted('actual-vs-pred-plot', yTest, result.yPred);
    renderResidualPlot('residual-plot', yTest, result.yPred);

    if (result.featureImportance && featureNames) {
        renderFeatureImportance('feature-importance-plot', featureNames, result.featureImportance);
    }

    // Tune button handler
    if (hasTuneGrid) {
        container.querySelector('#btn-tune').addEventListener('click', () => {
            runTuneModel(container, result, featureNames);
        });
    }

    // Interpret button handler
    container.querySelector('#btn-interpret').addEventListener('click', () => {
        runInterpretModel(container, result, featureNames);
    });

    // Predict button handler
    container.querySelector('#btn-predict').addEventListener('click', () => {
        runPredictModel(container, result, featureNames);
    });

    evalSection.scrollIntoView({ behavior: 'smooth' });
}

async function runTuneModel(container, result, featureNames) {
    const tuneResults = container.querySelector('#tune-results');
    const btnTune = container.querySelector('#btn-tune');
    btnTune.disabled = true;

    tuneResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #d97706;"></i>
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
            { cv: _state.cvFolds, scoring: 'r2' }
        );

        // Train best model on full training data and evaluate on test
        const tunedModel = new result.cls(bestParams);
        tunedModel.fit(_state.XTrain, _state.yTrain);
        const yPredTuned = tunedModel.predict(_state.XTest);
        const tunedR2 = rSquared(_state.yTest, yPredTuned);
        const tunedMAE = meanAbsoluteError(_state.yTest, yPredTuned);
        const tunedRMSE = rootMeanSquaredError(_state.yTest, yPredTuned);

        const improved = tunedR2 > result.r2;

        tuneResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4>GridSearch 結果</h4>
                <p><strong>Best Params:</strong> ${JSON.stringify(bestParams)}</p>
                <p><strong>Best CV R² (mean):</strong> ${formatNumber(bestScore)}</p>

                <h4 style="margin-top: 1rem;">Before vs After</h4>
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr><th>指標</th><th>チューニング前</th><th>チューニング後</th><th>変化</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>CV R² (mean)</td>
                                <td>${formatNumber(result.cvMean)}</td>
                                <td>${formatNumber(bestScore)}</td>
                                <td style="color: ${bestScore > result.cvMean ? '#10b981' : '#ef4444'};">
                                    ${bestScore > result.cvMean ? '+' : ''}${formatNumber(bestScore - result.cvMean)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test R²</td>
                                <td>${formatNumber(result.r2)}</td>
                                <td>${formatNumber(tunedR2)}</td>
                                <td style="color: ${tunedR2 > result.r2 ? '#10b981' : '#ef4444'};">
                                    ${tunedR2 > result.r2 ? '+' : ''}${formatNumber(tunedR2 - result.r2)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test MAE</td>
                                <td>${formatNumber(result.mae)}</td>
                                <td>${formatNumber(tunedMAE)}</td>
                                <td style="color: ${tunedMAE < result.mae ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(tunedMAE - result.mae)}
                                </td>
                            </tr>
                            <tr>
                                <td>Test RMSE</td>
                                <td>${formatNumber(result.rmse)}</td>
                                <td>${formatNumber(tunedRMSE)}</td>
                                <td style="color: ${tunedRMSE < result.rmse ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(tunedRMSE - result.rmse)}
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
                        <i class="fas fa-info-circle"></i> テスト性能は改善しませんでしたが、CV R²は参考になります。
                      </p>`
                }

                <details style="margin-top: 1rem;">
                    <summary style="cursor: pointer; font-weight: 600;">全パラメータ組み合わせ (${gsResults.length}通り)</summary>
                    <div class="table-container" style="margin-top: 0.5rem;">
                        <table class="table" style="font-size: 0.85rem;">
                            <thead>
                                <tr><th>順位</th><th>パラメータ</th><th>CV R² (mean)</th></tr>
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
            result.r2 = tunedR2;
            result.mae = tunedMAE;
            result.rmse = tunedRMSE;
            result.cvMean = bestScore;
            result.featureImportance = tunedModel.getFeatureImportance ? tunedModel.getFeatureImportance() : null;
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
    container.querySelector('.step-indicator').outerHTML = createStepIndicator(['Setup', 'Compare', 'Tune', 'Interpret', 'Predict'], 3);

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
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">各特徴量が予測値にどう影響するかを可視化します。</p>';
        html += '<div style="margin-bottom: 1rem;"><label style="font-weight: 600; margin-right: 0.5rem;">特徴量を選択:</label>';
        html += `<select id="pdp-feature-select" class="form-select" style="display: inline-block; width: auto;">
            ${featureNames.map((f, i) => `<option value="${i}">${f}</option>`).join('')}
        </select></div>`;
        html += '<div id="pdp-plot"></div>';

        // 3. Learning Curve
        html += '<h4 style="margin-top: 2rem;"><i class="fas fa-graduation-cap" style="color: #16a34a;"></i> Learning Curve（学習曲線）</h4>';
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">訓練データ量と性能の関係を可視化。過学習・未学習の診断に使います。</p>';
        html += '<div id="learning-curve-plot"></div>';

        // 4. Interpretation
        html += '<div id="interpret-analysis" style="margin-top: 2rem;"></div>';

        html += '</div>';
        interpretResults.innerHTML = html;

        // Compute permutation importance
        const { importancesMean, importancesStd } = permutationImportance(
            result.model, _state.XTest, _state.yTest,
            { scoring: 'r2', nRepeats: 5 }
        );
        renderPermutationImportance('perm-importance-plot', featureNames, importancesMean, importancesStd);

        // Compute PDP for the first feature initially
        computeAndRenderPDP(result.model, featureNames, 0);

        // PDP feature selector
        container.querySelector('#pdp-feature-select').addEventListener('change', (e) => {
            computeAndRenderPDP(result.model, featureNames, parseInt(e.target.value));
        });

        // Compute learning curve
        const lcResult = learningCurve(
            result.cls, result.model.getParams ? result.model.getParams() : {},
            _state.XTrain, _state.yTrain,
            { cv: Math.min(_state.cvFolds, 3), scoring: 'r2' }
        );
        renderLearningCurve(
            'learning-curve-plot',
            lcResult.trainSizes,
            lcResult.trainScoresMean, lcResult.trainScoresStd,
            lcResult.testScoresMean, lcResult.testScoresStd,
            'R²'
        );

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

    // Compute partial dependence
    const pdpValues = gridValues.map(gridVal => {
        const XModified = X.map(row => {
            const newRow = [...row];
            newRow[featureIndex] = gridVal;
            return newRow;
        });
        const preds = model.predict(XModified);
        return preds.reduce((a, b) => a + b, 0) / preds.length;
    });

    renderPDP('pdp-plot', featureNames[featureIndex], gridValues, pdpValues, '#16a34a');
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
        <h4><i class="fas fa-lightbulb" style="color: #d97706;"></i> 解釈の総合分析</h4>
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #16a34a; line-height: 1.8;">
            <p><strong>重要な特徴量:</strong></p>
            <ul>
    `;

    if (topFeatures.length > 0) {
        topFeatures.forEach((f, i) => {
            html += `<li><strong>${f.name}</strong> (重要度: ${f.importance.toFixed(4)}) - `;
            if (i === 0) html += 'このモデルで最も予測に寄与する特徴量です。';
            else html += '予測に重要な特徴量です。';
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

        predictResult.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: center;">
                <p style="font-size: 0.9rem; color: var(--text-secondary);">予測結果 (${result.name})</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #2563eb; margin: 0.5rem 0;">
                    ${formatNumber(prediction[0], 4)}
                </p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">
                    目的変数: ${_state.targetCol}
                </p>
            </div>
        `;
    } catch (error) {
        predictResult.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> 予測エラー: ${error.message}</p>`;
        console.error('Predict error:', error);
    }
}

function interpretResults(result) {
    const r2 = result.r2;
    let interpretation = `<p><strong>${result.name}</strong> の評価結果:</p><ul>`;

    if (r2 >= 0.9) {
        interpretation += `<li>R² = ${formatNumber(r2)} : <strong style="color: #10b981;">非常に高い予測精度</strong>です。モデルがデータの${(r2 * 100).toFixed(1)}%の分散を説明しています。</li>`;
    } else if (r2 >= 0.7) {
        interpretation += `<li>R² = ${formatNumber(r2)} : <strong style="color: #3b82f6;">良好な予測精度</strong>です。実用的に十分な精度と言えます。</li>`;
    } else if (r2 >= 0.5) {
        interpretation += `<li>R² = ${formatNumber(r2)} : <strong style="color: #f59e0b;">中程度の予測精度</strong>です。特徴量の追加や前処理の改善を検討してください。</li>`;
    } else {
        interpretation += `<li>R² = ${formatNumber(r2)} : <strong style="color: #ef4444;">予測精度が低い</strong>です。データの品質や特徴量の選択を見直してください。</li>`;
    }

    // CV interpretation
    interpretation += `<li>CV R² = ${formatNumber(result.cvMean)} ± ${formatNumber(result.cvStd)} : `;
    if (Math.abs(result.cvMean - result.r2) < 0.1) {
        interpretation += `交差検証とテストの差が小さく、<strong style="color: #10b981;">安定したモデル</strong>です。</li>`;
    } else if (result.cvMean > result.r2) {
        interpretation += `CVがテストR²より高く、<strong style="color: #f59e0b;">テストデータにやや弱い</strong>可能性があります。</li>`;
    } else {
        interpretation += `CVよりテストR²が高い結果です。</li>`;
    }

    interpretation += `<li>MAE = ${formatNumber(result.mae)} : 予測値と実測値の平均的なずれは ${formatNumber(result.mae)} です。</li>`;
    interpretation += `<li>RMSE = ${formatNumber(result.rmse)} : 大きな誤差をより重く評価した指標で ${formatNumber(result.rmse)} です。</li>`;
    interpretation += `</ul>`;

    return interpretation;
}
