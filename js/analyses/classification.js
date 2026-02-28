// ==========================================
// 分類モデル比較 (AutoML) Module
// ==========================================
import { createSelect, createStepIndicator, formatNumber, renderPlot, renderConfusionMatrix, renderROCCurve, renderFeatureImportance, createMetricCard } from '../utils.js';
import { prepareFeatures } from '../ml/preprocessing.js';
import { trainTestSplit } from '../ml/model_selection.js';
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

export function render(container, data, characteristics) {
    const catCols = characteristics.categoricalColumns;
    const numCols = characteristics.numericColumns;
    const allCols = characteristics.allColumns || Object.keys(data[0]);

    // Find suitable target columns (categorical with 2-10 unique values)
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

        ${createStepIndicator(['Setup', 'Compare', 'Evaluate'], 0)}

        <div id="setup-section" class="model-config">
            <h3><i class="fas fa-cog"></i> Step 1: セットアップ</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
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
    const selectedFeatures = Array.from(container.querySelectorAll('#feature-chips input:checked')).map(cb => cb.value);

    if (selectedFeatures.length === 0) {
        alert('特徴量を1つ以上選択してください。');
        return;
    }

    container.querySelector('#setup-section').style.display = 'none';
    container.querySelector('#compare-section').style.display = 'block';

    container.querySelector('.step-indicator').innerHTML = createStepIndicator(['Setup', 'Compare', 'Evaluate'], 1).replace(/<\/?div[^>]*class="step-indicator"[^>]*>/g, '');

    const progressArea = container.querySelector('#progress-area');
    progressArea.innerHTML = `<div style="text-align: center; padding: 2rem;">
        <i class="fas fa-spinner fa-spin fa-2x" style="color: #0891b2;"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習・比較しています...</p>
        <div id="model-progress" style="margin-top: 1rem;"></div>
    </div>`;

    await new Promise(r => setTimeout(r, 100));

    try {
        const { X, y, featureNames, labelEncoder } = prepareFeatures(data, targetCol, {
            selectedFeatures,
            task: 'classification'
        });

        const classes = [...new Set(y)].sort((a, b) => a - b);
        const classLabels = labelEncoder ? classes.map(c => labelEncoder.inverseTransform([c])[0]) : classes.map(String);

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
                const yProba = model.predictProba ? model.predictProba(XTest) : null;

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
                    model,
                    acc, prec, rec, f1, cm, auc, ll,
                    yPred, yProba,
                    featureImportance: model.getFeatureImportance ? model.getFeatureImportance() : null
                });
            } catch (err) {
                console.error(`${modelDef.name} failed:`, err);
                results.push({
                    name: modelDef.name,
                    badge: modelDef.badge,
                    model: null,
                    acc: 0, prec: 0, rec: 0, f1: 0, cm: null, auc: null, ll: null,
                    yPred: null, yProba: null,
                    error: err.message
                });
            }
        }

        results.sort((a, b) => b.f1 - a.f1);

        renderComparisonResults(container, results, yTest, featureNames, classes, classLabels);
    } catch (error) {
        progressArea.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-triangle"></i> エラー: ${error.message}</p>`;
        console.error(error);
    }
}

function renderComparisonResults(container, results, yTest, featureNames, classes, classLabels) {
    const comparisonDiv = container.querySelector('#comparison-results');
    const bestF1 = Math.max(...results.filter(r => r.model).map(r => r.f1));
    const hasAuc = results.some(r => r.auc != null);

    let html = `
        <h3 style="margin-top: 1rem;"><i class="fas fa-trophy" style="color: #0891b2;"></i> モデル比較結果</h3>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            テストデータ (${yTest.length} サンプル) での評価結果です。F1スコアが高いほど良い分類です。
        </p>
        <div class="table-container">
            <table class="table model-comparison-table">
                <thead>
                    <tr>
                        <th>順位</th>
                        <th>モデル</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        ${hasAuc ? '<th>AUC</th>' : ''}
                        <th>詳細</th>
                    </tr>
                </thead>
                <tbody>
                    ${results.map((r, i) => `
                        <tr class="${r.f1 === bestF1 && r.model ? 'highlight-row' : ''} ${!r.model ? 'error-row' : ''}">
                            <td>${r.model ? i + 1 : '-'}</td>
                            <td>
                                <span class="badge badge-${r.badge.toLowerCase()}">${r.badge}</span>
                                ${r.name}
                                ${r.f1 === bestF1 && r.model ? ' <i class="fas fa-crown" style="color: #0891b2;"></i>' : ''}
                            </td>
                            <td>${r.model ? formatNumber(r.acc) : '<span style="color:#ef4444;">エラー</span>'}</td>
                            <td>${r.model ? formatNumber(r.prec) : '-'}</td>
                            <td>${r.model ? formatNumber(r.rec) : '-'}</td>
                            <td><strong>${r.model ? formatNumber(r.f1) : '-'}</strong></td>
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

    container.querySelector('.step-indicator').outerHTML = createStepIndicator(['Setup', 'Compare', 'Evaluate'], 2);

    const evalContent = container.querySelector('#evaluation-content');

    evalContent.innerHTML = `
        <h3><i class="fas fa-chart-bar" style="color: #0891b2;"></i> ${result.name} の詳細評価</h3>

        <div class="metrics-grid" style="margin: 1.5rem 0;">
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

    evalSection.scrollIntoView({ behavior: 'smooth' });
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

    interpretation += `<li>Accuracy = ${formatNumber(result.acc)} : 全体の ${(result.acc * 100).toFixed(1)}% を正しく分類できました。</li>`;
    interpretation += `<li>Precision = ${formatNumber(result.prec)} : 正と予測したものの ${(result.prec * 100).toFixed(1)}% が実際に正でした。</li>`;
    interpretation += `<li>Recall = ${formatNumber(result.rec)} : 実際に正のものの ${(result.rec * 100).toFixed(1)}% を検出できました。</li>`;

    if (result.auc != null) {
        interpretation += `<li>AUC = ${formatNumber(result.auc)} : ROC曲線下面積で、1.0に近いほどランダムより優れた分類です。</li>`;
    }

    interpretation += `</ul>`;
    return interpretation;
}
