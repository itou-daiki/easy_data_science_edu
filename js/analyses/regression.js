// ==========================================
// 回帰モデル比較 (AutoML) Module
// PyCaret-style: setup → compare_models (CV) → tune_model → predict_model
// ==========================================
import { createSelect, createStepIndicator, formatNumber, renderPlot, renderActualVsPredicted, renderResidualPlot, renderFeatureImportance, createMetricCard, renderPermutationImportance, renderPDP, renderLearningCurve, renderSHAPSummary, renderSHAPBeeswarm, renderSHAPWaterfall, toCSV, downloadCSV, createDownloadButton, makeExportFileName, renderDataPreview, renderSummaryStatistics, downloadJSON, serializeModel, makeModelFileName } from '../utils.js';
import { linearSHAP, kernelSHAP, shapSummary } from '../ml/shap.js';
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

const STEPS = ['Setup', 'Preprocess', 'Compare', 'Create', 'Tune', 'Interpret', 'Blend', 'Stack', 'Finalize', 'Predict'];

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

        ${createStepIndicator(STEPS, 0)}

        <div id="data-overview-section" style="margin-bottom: 1.5rem;">
            <div id="reg-dataframe-container"></div>
            <div id="reg-summary-stats-container"></div>
        </div>

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

    renderDataPreview('reg-dataframe-container', data, 'データプレビュー');
    renderSummaryStatistics('reg-summary-stats-container', data, characteristics, '要約統計量');

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

    // Show Preprocess step first
    container.querySelector('.step-indicator').innerHTML = createStepIndicator(STEPS, 1).replace(/<\/?div[^>]*class="step-indicator"[^>]*>/g, '');

    const progressArea = container.querySelector('#progress-area');
    progressArea.innerHTML = `<div style="text-align: center; padding: 2rem;">
        <i class="fas fa-spinner fa-spin fa-2x" style="color: #d97706;"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
        <div id="model-progress" style="margin-top: 1rem;"></div>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const { X, y, featureNames, encoders, scaler } = prepareFeatures(data, targetCol, {
            selectedFeatures,
            task: 'regression'
        });

        const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize, randomState: 42 });

        // Save state for tune/predict
        _state = { XTrain, XTest, yTrain, yTest, featureNames, scaler, encoders, cvFolds, targetCol, fileName: characteristics.fileName || 'data' };

        // Compute preprocessing info
        const missingCount = selectedFeatures.reduce((sum, col) => {
            return sum + data.filter(row => row[col] == null || row[col] === '').length;
        }, 0);
        const categoricalCount = encoders ? encoders.size : 0;
        const scalingApplied = scaler !== null;

        // Show preprocessing summary
        progressArea.innerHTML = `
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <h3 style="margin: 0 0 1rem 0; font-size: 1.1rem; color: #166534;">
                    <i class="fas fa-magic" style="margin-right: 0.5rem;"></i>Step 2: 前処理 (自動完了)
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #15803d;">
                        <i class="fas fa-check-circle"></i>
                        <span>目的変数: <strong>${targetCol}</strong></span>
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
                <i class="fas fa-spinner fa-spin fa-2x" style="color: #d97706;"></i>
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
        ${createDownloadButton('dl-comparison-csv', '比較結果をCSVダウンロード')}
    `;

    comparisonDiv.innerHTML = html;
    container.querySelector('#progress-area').innerHTML = '';

    comparisonDiv.querySelectorAll('.btn-detail').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.index);
            showModelDetail(container, results[idx], yTest, featureNames);
        });
    });

    const dlCompBtn = comparisonDiv.querySelector('#dl-comparison-csv');
    if (dlCompBtn) {
        dlCompBtn.addEventListener('click', () => {
            const headers = ['順位', 'モデル', 'Badge', 'CV R² (mean)', 'CV R² (std)', 'Test R²', 'MAE', 'RMSE'];
            const rows = _state.results.map((r, i) => [
                r.model ? i + 1 : '-',
                r.name,
                r.badge,
                r.model ? formatNumber(r.cvMean) : '-',
                r.model ? formatNumber(r.cvStd) : '-',
                r.model ? formatNumber(r.r2) : '-',
                r.model ? formatNumber(r.mae) : '-',
                r.model ? formatNumber(r.rmse) : '-'
            ]);
            downloadCSV(toCSV(headers, rows), makeExportFileName(_state.fileName, '回帰_比較結果'));
        });
    }

    // Auto-show best model detail
    const bestModel = results.find(r => r.model);
    if (bestModel) {
        showModelDetail(container, bestModel, yTest, featureNames);
    }
}

function showModelDetail(container, result, yTest, featureNames) {
    const evalSection = container.querySelector('#evaluate-section');
    evalSection.style.display = 'block';

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 3);

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

        <!-- Create Model Section -->
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-radius: 12px;">
            <h4><i class="fas fa-plus-circle" style="color: #059669;"></i> create_model - モデル個別作成</h4>
            <p style="color: #065f46; margin: 0.5rem 0;">
                特定のアルゴリズムとパラメータを指定してモデルを作成します。
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">アルゴリズム:</label>
                    <select id="create-model-select" class="form-select">
                        ${MODELS.map((m, i) => {
                            const isTop = _state.results && _state.results[0] && _state.results[0].badge === m.badge;
                            return `<option value="${i}" ${isTop ? 'selected' : ''}>${m.name} (${m.badge})${isTop ? ' ★1位' : ''}</option>`;
                        }).join('')}
                    </select>
                </div>
                <div id="create-params-area">
                    <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">パラメータ:</label>
                    <div id="create-params-inputs"></div>
                </div>
            </div>
            <button id="btn-create-model" class="btn-analysis" style="background: #059669; margin-top: 0.5rem;">
                <i class="fas fa-hammer"></i> create_model を実行
            </button>
            <div id="create-model-results" style="margin-top: 1rem;"></div>
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
                上位モデルの予測値を平均して、より安定した予測を実現します。
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
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #fdf4ff, #fae8ff); border-radius: 12px;">
            <h4><i class="fas fa-layer-group" style="color: #a855f7;"></i> stack_models - スタッキングアンサンブル</h4>
            <p style="color: #6b21a8; margin: 0.5rem 0;">
                上位モデルの予測値を特徴量として、メタモデル（線形回帰）で最終予測を行います。ブレンド（平均）より高度なアンサンブル手法です。
            </p>
            <div style="margin: 1rem 0;">
                <label style="font-weight: 600; margin-right: 0.5rem;">ベースモデル数:</label>
                <select id="stack-top-n" class="form-select" style="display: inline-block; width: auto;">
                    <option value="3" selected>上位3モデル</option>
                    <option value="5">上位5モデル</option>
                    <option value="7">全モデル (7)</option>
                </select>
            </div>
            <button id="btn-stack" class="btn-analysis" style="background: #a855f7; margin-top: 0.5rem;">
                <i class="fas fa-cubes"></i> stack_models を実行
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

    renderActualVsPredicted('actual-vs-pred-plot', yTest, result.yPred);
    renderResidualPlot('residual-plot', yTest, result.yPred);

    if (result.featureImportance && featureNames) {
        renderFeatureImportance('feature-importance-plot', featureNames, result.featureImportance);
    }

    // Create model - initialize param inputs and handlers
    updateCreateModelParams(container, 0);
    container.querySelector('#create-model-select').addEventListener('change', (e) => {
        updateCreateModelParams(container, parseInt(e.target.value));
    });
    container.querySelector('#btn-create-model').addEventListener('click', () => {
        runCreateModel(container, featureNames);
    });

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

    // Blend button handler
    container.querySelector('#btn-blend').addEventListener('click', () => {
        runBlendModels(container, featureNames);
    });

    // Stack button handler
    container.querySelector('#btn-stack').addEventListener('click', () => {
        runStackModels(container, featureNames);
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

async function runTuneModel(container, result, featureNames) {
    const tuneResults = container.querySelector('#tune-results');
    const btnTune = container.querySelector('#btn-tune');
    btnTune.disabled = true;

    tuneResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #d97706;"></i>
        <span style="margin-left: 0.5rem;">GridSearch CV 実行中...</span>
    </div>`;

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 4);
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

function updateCreateModelParams(container, modelIndex) {
    const modelDef = MODELS[modelIndex];
    const paramsContainer = container.querySelector('#create-params-inputs');
    const entries = Object.entries(modelDef.params);

    if (entries.length === 0) {
        paramsContainer.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.85rem;">このモデルにはパラメータがありません。</p>';
        return;
    }

    paramsContainer.innerHTML = entries.map(([key, defaultVal]) => `
        <div style="margin-bottom: 0.5rem;">
            <label style="font-size: 0.85rem; display: inline-block; width: 120px;">${key}:</label>
            <input type="number" id="create-param-${key}" class="form-select"
                   value="${defaultVal !== null ? defaultVal : ''}"
                   step="any" placeholder="${defaultVal !== null ? defaultVal : 'auto'}"
                   style="display: inline-block; width: 120px;">
        </div>
    `).join('');
}

async function runCreateModel(container, featureNames) {
    const createResults = container.querySelector('#create-model-results');
    const btnCreate = container.querySelector('#btn-create-model');
    btnCreate.disabled = true;

    // Update step indicator
    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 3);

    createResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #059669;"></i>
        <span style="margin-left: 0.5rem;">モデルを作成中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const modelIndex = parseInt(container.querySelector('#create-model-select').value);
        const modelDef = MODELS[modelIndex];

        // Collect params from inputs
        const params = {};
        for (const [key, defaultVal] of Object.entries(modelDef.params)) {
            const input = container.querySelector(`#create-param-${key}`);
            if (input && input.value !== '') {
                params[key] = parseFloat(input.value);
            } else if (defaultVal !== null) {
                params[key] = defaultVal;
            }
        }

        // Train the model
        const model = new modelDef.cls(params);
        model.fit(_state.XTrain, _state.yTrain);
        const yPred = model.predict(_state.XTest);

        // Cross-validation
        const cvScores = crossValidate(model, _state.XTrain, _state.yTrain, {
            cv: _state.cvFolds, scoring: 'r2'
        });
        const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
        const cvStd = Math.sqrt(cvScores.reduce((a, v) => a + (v - cvMean) ** 2, 0) / cvScores.length);

        const r2 = rSquared(_state.yTest, yPred);
        const mae = meanAbsoluteError(_state.yTest, yPred);
        const rmse = rootMeanSquaredError(_state.yTest, yPred);

        // Store in state
        _state.createdModel = model;
        _state.createdModelResult = {
            name: modelDef.name,
            badge: modelDef.badge,
            cls: modelDef.cls,
            model, r2, mae, rmse, cvMean, cvStd, yPred, params
        };

        const plotId = 'create-model-avp-plot';
        createResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4><i class="fas fa-check-circle" style="color: #059669;"></i> ${modelDef.name} を作成しました</h4>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    パラメータ: ${JSON.stringify(params)}
                </p>
                <div class="metrics-grid" style="margin: 1rem 0;">
                    ${createMetricCard('CV R² (mean)', cvMean, `${_state.cvFolds}-Fold 交差検証平均`)}
                    ${createMetricCard('Test R²', r2, 'テストデータ決定係数')}
                    ${createMetricCard('MAE', mae, '平均絶対誤差')}
                    ${createMetricCard('RMSE', rmse, '二乗平均平方根誤差')}
                </div>

                ${_state.results && _state.results[0] ? `
                <div class="table-container" style="margin-top: 1rem;">
                    <table class="table">
                        <thead>
                            <tr><th>指標</th><th>ベストモデル (${_state.results[0].badge})</th><th>作成モデル (${modelDef.badge})</th><th>変化</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Test R²</td>
                                <td>${formatNumber(_state.results[0].r2)}</td>
                                <td>${formatNumber(r2)}</td>
                                <td style="color: ${r2 > _state.results[0].r2 ? '#10b981' : '#ef4444'};">
                                    ${r2 > _state.results[0].r2 ? '+' : ''}${formatNumber(r2 - _state.results[0].r2)}
                                </td>
                            </tr>
                            <tr>
                                <td>MAE</td>
                                <td>${formatNumber(_state.results[0].mae)}</td>
                                <td>${formatNumber(mae)}</td>
                                <td style="color: ${mae < _state.results[0].mae ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(mae - _state.results[0].mae)}
                                </td>
                            </tr>
                            <tr>
                                <td>RMSE</td>
                                <td>${formatNumber(_state.results[0].rmse)}</td>
                                <td>${formatNumber(rmse)}</td>
                                <td style="color: ${rmse < _state.results[0].rmse ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(rmse - _state.results[0].rmse)}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                ` : ''}

                <p style="color: #059669; font-weight: 600; margin-top: 1rem;">
                    <i class="fas fa-info-circle"></i> 作成したモデルは predict_model で使用できます。
                </p>
                <div id="${plotId}" style="margin-top: 1.5rem;"></div>
            </div>
        `;

        renderActualVsPredicted(plotId, _state.yTest, yPred);

    } catch (error) {
        createResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> モデル作成エラー: ${error.message}</p>`;
        console.error('Create model error:', error);
    }

    btnCreate.disabled = false;
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

        // 4. SHAP Values
        html += '<h4 style="margin-top: 2rem;"><i class="fas fa-chart-pie" style="color: #16a34a;"></i> SHAP (SHapley Additive exPlanations)</h4>';
        html += '<p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">ゲーム理論に基づく特徴量の貢献度。各特徴量が個々の予測にどの程度影響しているかを定量化します。</p>';
        html += '<div id="shap-summary-plot"></div>';
        html += '<div id="shap-beeswarm-plot" style="margin-top: 1.5rem;"></div>';
        html += '<div id="shap-waterfall-plot" style="margin-top: 1.5rem;"></div>';

        // 5. Interpretation
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

        // Compute SHAP values
        try {
            const isLinearModel = result.model.coefficients != null && result.model.intercept != null;
            let shapResult;

            if (isLinearModel) {
                // Exact SHAP for linear models
                const featureMeans = featureNames.map((_, fi) =>
                    _state.XTrain.reduce((sum, row) => sum + row[fi], 0) / _state.XTrain.length
                );
                shapResult = linearSHAP(
                    result.model.coefficients,
                    result.model.intercept,
                    _state.XTest,
                    featureMeans
                );
            } else {
                // Kernel SHAP for non-linear models
                const predictFn = (X) => result.model.predict(X);
                shapResult = kernelSHAP(predictFn, _state.XTest, _state.XTrain, {
                    maxBackground: 50
                });
            }

            const { meanAbsSHAP, meanSHAP } = shapSummary(shapResult.shapValues);

            // Render SHAP Summary Bar
            renderSHAPSummary('shap-summary-plot', featureNames, meanAbsSHAP, meanSHAP);

            // Render SHAP Beeswarm
            renderSHAPBeeswarm('shap-beeswarm-plot', featureNames, shapResult.shapValues, _state.XTest);

            // Render SHAP Waterfall for the first test sample
            if (_state.XTest.length > 0) {
                const firstPred = result.model.predict([_state.XTest[0]])[0];
                renderSHAPWaterfall(
                    'shap-waterfall-plot',
                    featureNames,
                    shapResult.shapValues[0],
                    shapResult.baseValue,
                    firstPred
                );
            }
        } catch (shapError) {
            console.warn('SHAP computation failed:', shapError);
            const shapContainer = container.querySelector('#shap-summary-plot');
            if (shapContainer) {
                shapContainer.innerHTML = `<p style="color: #f59e0b; font-size: 0.9rem;">
                    <i class="fas fa-exclamation-triangle"></i> SHAP値の計算に失敗しました: ${shapError.message}
                </p>`;
            }
        }

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

async function runBlendModels(container, featureNames) {
    const blendResults = container.querySelector('#blend-results');
    const btnBlend = container.querySelector('#btn-blend');
    btnBlend.disabled = true;

    // Update step indicator
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

        // Blend predictions (average)
        const blendedPred = _state.yTest.map((_, i) => {
            const sum = topModels.reduce((acc, m) => acc + m.yPred[i], 0);
            return sum / topModels.length;
        });

        const blendedR2 = rSquared(_state.yTest, blendedPred);
        const blendedMAE = meanAbsoluteError(_state.yTest, blendedPred);
        const blendedRMSE = rootMeanSquaredError(_state.yTest, blendedPred);
        const bestSingleR2 = topModels[0].r2;

        // Create a blended model proxy for predict
        const blendedModel = {
            models: topModels.map(m => m.model),
            predict(X) {
                const predictions = this.models.map(m => m.predict(X));
                return X.map((_, i) => {
                    const sum = predictions.reduce((acc, p) => acc + p[i], 0);
                    return sum / predictions.length;
                });
            },
            getParams() { return { type: 'blend', nModels: this.models.length }; }
        };

        // Store blended model in state for predict
        _state.blendedModel = blendedModel;

        blendResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4>ブレンド結果 (上位 ${topN} モデルの平均)</h4>
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
                                <td>Test R²</td>
                                <td>${formatNumber(bestSingleR2)}</td>
                                <td>${formatNumber(blendedR2)}</td>
                                <td style="color: ${blendedR2 > bestSingleR2 ? '#10b981' : '#ef4444'};">
                                    ${blendedR2 > bestSingleR2 ? '+' : ''}${formatNumber(blendedR2 - bestSingleR2)}
                                </td>
                            </tr>
                            <tr>
                                <td>MAE</td>
                                <td>${formatNumber(topModels[0].mae)}</td>
                                <td>${formatNumber(blendedMAE)}</td>
                                <td style="color: ${blendedMAE < topModels[0].mae ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(blendedMAE - topModels[0].mae)}
                                </td>
                            </tr>
                            <tr>
                                <td>RMSE</td>
                                <td>${formatNumber(topModels[0].rmse)}</td>
                                <td>${formatNumber(blendedRMSE)}</td>
                                <td style="color: ${blendedRMSE < topModels[0].rmse ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(blendedRMSE - topModels[0].rmse)}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                ${blendedR2 > bestSingleR2
                    ? `<p style="color: #10b981; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-check-circle"></i> ブレンドモデルがベストモデルを上回りました！predict_model でブレンドモデルを使用できます。
                      </p>`
                    : `<p style="color: #f59e0b; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-info-circle"></i> ブレンドモデルはベストモデルを上回りませんでした。単体モデルの方が適している可能性があります。
                      </p>`
                }
            </div>
        `;

        // Add actual vs predicted plot for blended model
        const plotId = 'blend-avp-plot';
        blendResults.querySelector('div').insertAdjacentHTML('beforeend', `<div id="${plotId}" style="margin-top: 1.5rem;"></div>`);
        renderActualVsPredicted(plotId, _state.yTest, blendedPred);

    } catch (error) {
        blendResults.innerHTML = `<p style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> ブレンドエラー: ${error.message}</p>`;
        console.error('Blend error:', error);
    }

    btnBlend.disabled = false;
}

async function runStackModels(container, featureNames) {
    const stackResults = container.querySelector('#stack-results');
    const btnStack = container.querySelector('#btn-stack');
    btnStack.disabled = true;

    // Update step indicator
    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 7);

    stackResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #a855f7;"></i>
        <span style="margin-left: 0.5rem;">スタッキングモデルを構築中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const topN = parseInt(container.querySelector('#stack-top-n').value);
        const validResults = _state.results.filter(r => r.model);
        const baseModels = validResults.slice(0, topN);

        // Generate meta-features from base model predictions on training data
        const metaTrainFeatures = _state.XTrain.map(row => {
            const singleRow = [row];
            return baseModels.map(m => m.model.predict(singleRow)[0]);
        });

        // Train meta-learner (LinearRegression) on base model predictions
        const metaLearner = new LinearRegression();
        metaLearner.fit(metaTrainFeatures, _state.yTrain);

        // Generate meta-features for test data
        const metaTestFeatures = _state.XTest.map(row => {
            const singleRow = [row];
            return baseModels.map(m => m.model.predict(singleRow)[0]);
        });

        // Predict with stacked model
        const stackedPred = metaLearner.predict(metaTestFeatures);

        const stackedR2 = rSquared(_state.yTest, stackedPred);
        const stackedMAE = meanAbsoluteError(_state.yTest, stackedPred);
        const stackedRMSE = rootMeanSquaredError(_state.yTest, stackedPred);
        const bestSingleR2 = baseModels[0].r2;

        // Create a stacked model proxy for predict
        const stackedModel = {
            baseModels: baseModels.map(m => m.model),
            metaLearner,
            predict(X) {
                const metaFeatures = X.map(row => {
                    const singleRow = [row];
                    return this.baseModels.map(m => m.predict(singleRow)[0]);
                });
                return this.metaLearner.predict(metaFeatures);
            },
            getParams() {
                return { type: 'stack', nBaseModels: this.baseModels.length, metaLearner: 'LinearRegression' };
            }
        };

        // Store stacked model in state for predict
        _state.stackedModel = stackedModel;

        // Show meta-learner coefficients
        const metaCoeffs = metaLearner.coefficients || [];
        const metaIntercept = metaLearner.intercept || 0;

        const plotId = 'stack-avp-plot';
        stackResults.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                <h4>スタッキング結果 (上位 ${topN} モデル → LinearRegression メタモデル)</h4>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    ベースモデル: ${baseModels.map(m => m.badge).join(', ')} → メタモデルが最適な重み付けを学習
                </p>

                ${metaCoeffs.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <h4 style="font-size: 0.95rem;">メタモデルの重み (各ベースモデルへの重み付け)</h4>
                    <div class="table-container">
                        <table class="table" style="font-size: 0.85rem;">
                            <thead>
                                <tr><th>ベースモデル</th><th>係数（重み）</th></tr>
                            </thead>
                            <tbody>
                                ${baseModels.map((m, i) => `
                                    <tr>
                                        <td><span class="badge badge-${m.badge.toLowerCase()}">${m.badge}</span> ${m.name}</td>
                                        <td>${formatNumber(metaCoeffs[i])}</td>
                                    </tr>
                                `).join('')}
                                <tr style="font-weight: 600;">
                                    <td>切片 (Intercept)</td>
                                    <td>${formatNumber(metaIntercept)}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                ` : ''}

                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr><th>指標</th><th>ベストモデル (${baseModels[0].badge})</th><th>スタッキングモデル</th><th>変化</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Test R²</td>
                                <td>${formatNumber(bestSingleR2)}</td>
                                <td>${formatNumber(stackedR2)}</td>
                                <td style="color: ${stackedR2 > bestSingleR2 ? '#10b981' : '#ef4444'};">
                                    ${stackedR2 > bestSingleR2 ? '+' : ''}${formatNumber(stackedR2 - bestSingleR2)}
                                </td>
                            </tr>
                            <tr>
                                <td>MAE</td>
                                <td>${formatNumber(baseModels[0].mae)}</td>
                                <td>${formatNumber(stackedMAE)}</td>
                                <td style="color: ${stackedMAE < baseModels[0].mae ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(stackedMAE - baseModels[0].mae)}
                                </td>
                            </tr>
                            <tr>
                                <td>RMSE</td>
                                <td>${formatNumber(baseModels[0].rmse)}</td>
                                <td>${formatNumber(stackedRMSE)}</td>
                                <td style="color: ${stackedRMSE < baseModels[0].rmse ? '#10b981' : '#ef4444'};">
                                    ${formatNumber(stackedRMSE - baseModels[0].rmse)}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                ${stackedR2 > bestSingleR2
                    ? `<p style="color: #10b981; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-check-circle"></i> スタッキングモデルがベストモデルを上回りました！predict_model でスタッキングモデルを使用できます。
                      </p>`
                    : `<p style="color: #f59e0b; font-weight: 600; margin-top: 1rem;">
                        <i class="fas fa-info-circle"></i> スタッキングモデルはベストモデルを上回りませんでした。他のアンサンブル手法を試してみてください。
                      </p>`
                }
                <div id="${plotId}" style="margin-top: 1.5rem;"></div>
            </div>
        `;

        renderActualVsPredicted(plotId, _state.yTest, stackedPred);

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

    // Update step indicator
    container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 8);

    finalizeResults.innerHTML = `<div style="text-align: center; padding: 1rem;">
        <i class="fas fa-spinner fa-spin" style="color: #dc2626;"></i>
        <span style="margin-left: 0.5rem;">全データで再学習中...</span>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        // Combine train and test data
        const XFull = [..._state.XTrain, ..._state.XTest];
        const yFull = [..._state.yTrain, ..._state.yTest];

        // Retrain the model on full data
        const params = result.model.getParams ? result.model.getParams() : {};
        const finalModel = new result.cls(params);
        finalModel.fit(XFull, yFull);

        // Store for predict
        _state.finalizedModel = finalModel;
        _state.isFinalized = true;

        // CV score on full data for reference
        const cvScores = crossValidate(finalModel, XFull, yFull, { cv: _state.cvFolds, scoring: 'r2' });
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
                    ${createMetricCard('CV R² (mean)', cvMean, `${_state.cvFolds}-Fold 全データCV`)}
                    ${createMetricCard('CV R² (std)', cvStd, '交差検証の標準偏差')}
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
                    { featureNames: _state.featureNames, scaler: _state.scaler, encoders: _state.encoders, targetCol: _state.targetCol, fileName: _state.fileName, taskType: 'regression' }
                );
                downloadJSON(exportData, makeModelFileName(_state.fileName, result.badge, 'regression'));
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

        // Use finalized model > stacked model > blended model > created model > original model
        const activeModel = _state.finalizedModel || _state.stackedModel || _state.blendedModel || _state.createdModel || result.model;
        const modelLabel = _state.finalizedModel ? `${result.name} (確定済み)`
                         : _state.stackedModel ? 'スタッキングモデル'
                         : _state.blendedModel ? 'ブレンドモデル'
                         : _state.createdModel ? `${_state.createdModelResult.name} (作成済み)`
                         : result.name;
        const prediction = activeModel.predict(processedInput);

        // Update step indicator to Predict
        container.querySelector('.step-indicator').outerHTML = createStepIndicator(STEPS, 9);

        predictResult.innerHTML = `
            <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: center;">
                <p style="font-size: 0.9rem; color: var(--text-secondary);">予測結果 (${modelLabel})</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #2563eb; margin: 0.5rem 0;">
                    ${formatNumber(prediction[0], 4)}
                </p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">
                    目的変数: ${_state.targetCol}
                </p>
                ${createDownloadButton('dl-predict-csv', '予測結果をCSVダウンロード')}
            </div>
        `;

        const dlPredBtn = predictResult.querySelector('#dl-predict-csv');
        if (dlPredBtn) {
            dlPredBtn.addEventListener('click', () => {
                const headers = ['特徴量名', '入力値'];
                const rows = featureNames.map((f, i) => [f, inputValues[i]]);
                rows.push(['予測結果', formatNumber(prediction[0], 4)]);
                downloadCSV(toCSV(headers, rows), makeExportFileName(_state.fileName, '回帰_予測結果'));
            });
        }
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
