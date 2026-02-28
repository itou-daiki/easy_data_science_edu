// ==========================================
// 回帰モデル比較 (AutoML) Module
// ==========================================
import { createSelect, createStepIndicator, formatNumber, renderPlot, renderActualVsPredicted, renderResidualPlot, renderFeatureImportance, createMetricCard } from '../utils.js';
import { prepareFeatures } from '../ml/preprocessing.js';
import { trainTestSplit } from '../ml/model_selection.js';
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

export function render(container, data, characteristics) {
    const numCols = characteristics.numericColumns;

    container.innerHTML = `
        <h2><i class="fas fa-robot" style="color: #d97706;"></i> 回帰モデル比較 (AutoML)</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            PyCaret のように複数の回帰モデルを一括学習・比較し、最適なモデルを見つけます。
        </p>

        ${createStepIndicator(['Setup', 'Compare', 'Evaluate'], 0)}

        <div id="setup-section" class="model-config">
            <h3><i class="fas fa-cog"></i> Step 1: セットアップ</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
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
    const selectedFeatures = Array.from(container.querySelectorAll('#feature-chips input:checked')).map(cb => cb.value);

    if (selectedFeatures.length === 0) {
        alert('特徴量を1つ以上選択してください。');
        return;
    }

    container.querySelector('#setup-section').style.display = 'none';
    container.querySelector('#compare-section').style.display = 'block';

    // Update step indicator
    container.querySelector('.step-indicator').innerHTML = createStepIndicator(['Setup', 'Compare', 'Evaluate'], 1).replace(/<\/?div[^>]*class="step-indicator"[^>]*>/g, '');

    const progressArea = container.querySelector('#progress-area');
    progressArea.innerHTML = `<div style="text-align: center; padding: 2rem;">
        <i class="fas fa-spinner fa-spin fa-2x" style="color: #d97706;"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
        <div id="model-progress" style="margin-top: 1rem;"></div>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const { X, y, featureNames } = prepareFeatures(data, targetCol, {
            selectedFeatures,
            task: 'regression'
        });

        const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize, randomState: 42 });

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

                const mae = meanAbsoluteError(yTest, yPred);
                const mse = meanSquaredError(yTest, yPred);
                const rmse = rootMeanSquaredError(yTest, yPred);
                const r2 = rSquared(yTest, yPred);
                const adjR2 = adjustedRSquared(yTest, yPred, featureNames.length);

                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    model,
                    mae, mse, rmse, r2, adjR2,
                    yPred,
                    featureImportance: model.getFeatureImportance ? model.getFeatureImportance() : null
                });
            } catch (err) {
                console.error(`${modelDef.name} failed:`, err);
                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    model: null,
                    mae: Infinity, mse: Infinity, rmse: Infinity, r2: -Infinity, adjR2: -Infinity,
                    yPred: null,
                    error: err.message
                });
            }
        }

        results.sort((a, b) => b.r2 - a.r2);

        renderComparisonResults(container, results, yTest, featureNames, XTest, XTrain, yTrain);
    } catch (error) {
        progressArea.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-triangle"></i> エラー: ${error.message}</p>`;
        console.error(error);
    }
}

function renderComparisonResults(container, results, yTest, featureNames, XTest, XTrain, yTrain) {
    const comparisonDiv = container.querySelector('#comparison-results');

    const bestR2 = Math.max(...results.filter(r => r.model).map(r => r.r2));

    let html = `
        <h3 style="margin-top: 1rem;"><i class="fas fa-trophy" style="color: #d97706;"></i> モデル比較結果</h3>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            テストデータ (${yTest.length} サンプル) での評価結果です。R² が高いほど良い予測です。
        </p>
        <div class="table-container">
            <table class="table model-comparison-table">
                <thead>
                    <tr>
                        <th>順位</th>
                        <th>モデル</th>
                        <th>MAE</th>
                        <th>MSE</th>
                        <th>RMSE</th>
                        <th>R²</th>
                        <th>Adj. R²</th>
                        <th>詳細</th>
                    </tr>
                </thead>
                <tbody>
                    ${results.map((r, i) => `
                        <tr class="${r.r2 === bestR2 && r.model ? 'highlight-row' : ''} ${!r.model ? 'error-row' : ''}">
                            <td>${r.model ? i + 1 : '-'}</td>
                            <td>
                                <span class="badge badge-${r.badge.toLowerCase()}">${r.badge}</span>
                                ${r.name}
                                ${r.r2 === bestR2 && r.model ? ' <i class="fas fa-crown" style="color: #d97706;"></i>' : ''}
                            </td>
                            <td>${r.model ? formatNumber(r.mae) : '<span style="color:#ef4444;">エラー</span>'}</td>
                            <td>${r.model ? formatNumber(r.mse) : '-'}</td>
                            <td>${r.model ? formatNumber(r.rmse) : '-'}</td>
                            <td><strong>${r.model ? formatNumber(r.r2) : '-'}</strong></td>
                            <td>${r.model ? formatNumber(r.adjR2) : '-'}</td>
                            <td>${r.model ? `<button class="btn-detail" data-index="${i}" style="padding: 0.25rem 0.75rem; font-size: 0.85rem;">詳細</button>` : '-'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;

    comparisonDiv.innerHTML = html;
    container.querySelector('#progress-area').innerHTML = '';

    // Detail button handlers
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

    // Update step indicator
    container.querySelector('.step-indicator').outerHTML = createStepIndicator(['Setup', 'Compare', 'Evaluate'], 2);

    const evalContent = container.querySelector('#evaluation-content');

    evalContent.innerHTML = `
        <h3><i class="fas fa-chart-bar" style="color: #d97706;"></i> ${result.name} の詳細評価</h3>

        <div class="metrics-grid" style="margin: 1.5rem 0;">
            ${createMetricCard('MAE', result.mae, '平均絶対誤差')}
            ${createMetricCard('MSE', result.mse, '平均二乗誤差')}
            ${createMetricCard('RMSE', result.rmse, '二乗平均平方根誤差')}
            ${createMetricCard('R²', result.r2, '決定係数')}
            ${createMetricCard('Adj. R²', result.adjR2, '自由度調整済みR²')}
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 2rem;">
            <div>
                <div id="actual-vs-pred-plot"></div>
            </div>
            <div>
                <div id="residual-plot"></div>
            </div>
        </div>

        ${result.featureImportance ? '<div id="feature-importance-plot" style="margin-top: 2rem;"></div>' : ''}

        <div style="margin-top: 2rem;">
            <h4>結果の解釈</h4>
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #d97706; line-height: 1.8;">
                ${interpretResults(result)}
            </div>
        </div>
    `;

    renderActualVsPredicted('actual-vs-pred-plot', yTest, result.yPred);
    renderResidualPlot('residual-plot', yTest, result.yPred);

    if (result.featureImportance && featureNames) {
        renderFeatureImportance('feature-importance-plot', featureNames, result.featureImportance);
    }

    evalSection.scrollIntoView({ behavior: 'smooth' });
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

    interpretation += `<li>MAE = ${formatNumber(result.mae)} : 予測値と実測値の平均的なずれは ${formatNumber(result.mae)} です。</li>`;
    interpretation += `<li>RMSE = ${formatNumber(result.rmse)} : 大きな誤差をより重く評価した指標で ${formatNumber(result.rmse)} です。</li>`;
    interpretation += `</ul>`;

    return interpretation;
}
