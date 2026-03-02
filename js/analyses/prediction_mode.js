// ==========================================
// 予測モード Module
// JSONモデルファイルを読み込んで予測を実行
// ==========================================
import { formatNumber, deserializeModel } from '../utils.js';
import { StandardScaler, MinMaxScaler, LabelEncoder } from '../ml/preprocessing.js';

// Regression models
import { LinearRegression } from '../ml/regression/linear.js';
import { RidgeRegression } from '../ml/regression/ridge.js';
import { LassoRegression } from '../ml/regression/lasso.js';
import { DecisionTreeRegressor } from '../ml/regression/decision_tree.js';
import { RandomForestRegressor } from '../ml/regression/random_forest.js';
import { KNNRegressor } from '../ml/regression/knn.js';
import { GradientBoostingRegressor } from '../ml/regression/gradient_boosting.js';

// Classification models
import { LogisticRegression } from '../ml/classification/logistic.js';
import { DecisionTreeClassifier } from '../ml/classification/decision_tree.js';
import { RandomForestClassifier } from '../ml/classification/random_forest.js';
import { KNNClassifier } from '../ml/classification/knn.js';
import { GaussianNaiveBayes } from '../ml/classification/naive_bayes.js';
import { SVMClassifier } from '../ml/classification/svm.js';
import { GradientBoostingClassifier } from '../ml/classification/gradient_boosting.js';

const THEME = '#6366f1';

// ---------------------------------------------------------------------------
// Tree reconstruction
// ---------------------------------------------------------------------------

function deserializeTree(nodeData) {
    if (!nodeData) return null;
    if (nodeData.left === undefined && nodeData.right === undefined) {
        return { value: nodeData.value, classCounts: nodeData.classCounts };
    }
    return {
        featureIndex: nodeData.featureIndex,
        threshold: nodeData.threshold,
        left: deserializeTree(nodeData.left),
        right: deserializeTree(nodeData.right),
        value: nodeData.value,
        classCounts: nodeData.classCounts
    };
}

// ---------------------------------------------------------------------------
// Preprocessing reconstruction
// ---------------------------------------------------------------------------

function reconstructScaler(scalerData) {
    if (!scalerData) return null;
    if (scalerData.type === 'StandardScaler') {
        const s = new StandardScaler();
        s.means = scalerData.means;
        s.stds = scalerData.stds;
        s.isFitted = true;
        return s;
    }
    if (scalerData.type === 'MinMaxScaler') {
        const s = new MinMaxScaler();
        s.mins = scalerData.mins;
        s.maxs = scalerData.maxs;
        s.isFitted = true;
        return s;
    }
    return null;
}

function reconstructLabelEncoder(leData) {
    if (!leData) return null;
    const le = new LabelEncoder();
    le._classes = Object.freeze(leData.classes);
    const map = new Map();
    leData.classes.forEach((c, i) => map.set(c, i));
    le._classToIndex = map;
    le.isFitted = true;
    return le;
}

function reconstructEncoders(encodersData) {
    if (!encodersData) return null;
    const map = new Map();
    for (const enc of encodersData) {
        const le = new LabelEncoder();
        le._classes = Object.freeze(enc.classes);
        const classMap = new Map();
        enc.classes.forEach((c, i) => classMap.set(c, i));
        le._classToIndex = classMap;
        le.isFitted = true;
        map.set(enc.columnIndex, le);
    }
    return map;
}

// ---------------------------------------------------------------------------
// Model reconstruction
// ---------------------------------------------------------------------------

function reconstructModel(modelInfo, taskType) {
    const { badge, params } = modelInfo;

    switch (badge) {
        case 'Linear': {
            const m = new LinearRegression({});
            m.coefficients = params.coefficients;
            m.intercept = params.intercept;
            m.nFeatures = params.nFeatures;
            return m;
        }
        case 'Ridge': {
            const m = new RidgeRegression({ alpha: params.alpha });
            m.coefficients = params.coefficients;
            m.intercept = params.intercept;
            m.nFeatures = params.nFeatures;
            return m;
        }
        case 'Lasso': {
            const m = new LassoRegression({ alpha: params.alpha });
            m.coefficients = params.coefficients;
            m.intercept = params.intercept;
            m.nFeatures = params.nFeatures;
            return m;
        }
        case 'Tree': {
            const Cls = taskType === 'classification' ? DecisionTreeClassifier : DecisionTreeRegressor;
            const m = new Cls({ maxDepth: params.maxDepth, minSamplesSplit: params.minSamplesSplit, minSamplesLeaf: params.minSamplesLeaf });
            m.tree = deserializeTree(params.tree);
            m.nFeatures = params.nFeatures;
            if (params.nClasses) m.nClasses = params.nClasses;
            return m;
        }
        case 'RF': {
            const TreeCls = taskType === 'classification' ? DecisionTreeClassifier : DecisionTreeRegressor;
            const Cls = taskType === 'classification' ? RandomForestClassifier : RandomForestRegressor;
            const m = new Cls({ nEstimators: params.nEstimators, maxDepth: params.maxDepth, maxFeatures: params.maxFeatures });
            m.nFeatures = params.nFeatures;
            if (params.nClasses) m.nClasses = params.nClasses;
            m.trees = params.trees.map(tData => {
                const sub = new TreeCls({ maxDepth: tData.maxDepth });
                sub.tree = deserializeTree(tData.tree);
                sub.featureIndices = tData.featureIndices || null;
                sub.nFeatures = tData.featureIndices ? tData.featureIndices.length : params.nFeatures;
                if (tData.nClasses) sub.nClasses = tData.nClasses;
                return sub;
            });
            return m;
        }
        case 'KNN': {
            const Cls = taskType === 'classification' ? KNNClassifier : KNNRegressor;
            const m = new Cls({ nNeighbors: params.nNeighbors });
            m.XTrain = params.XTrain;
            m.yTrain = params.yTrain;
            m.weights = params.weights || 'uniform';
            if (params.nClasses) {
                m.nClasses = params.nClasses;
                m.classes = [...new Set(params.yTrain)].sort((a, b) => a - b);
            }
            return m;
        }
        case 'GBM': {
            if (taskType === 'classification') {
                const m = new GradientBoostingClassifier({ nEstimators: params.nEstimators, learningRate: params.learningRate, maxDepth: params.maxDepth || 3 });
                m.nFeatures = params.nFeatures;
                if (params.nClasses) m.nClasses = params.nClasses;
                if (params.classPriors) m.classPriors = params.classPriors;
                m.trees = params.trees.map(tData => {
                    const sub = new DecisionTreeRegressor({ maxDepth: tData.maxDepth });
                    sub.tree = deserializeTree(tData.tree);
                    return sub;
                });
                return m;
            }
            const m = new GradientBoostingRegressor({ nEstimators: params.nEstimators, learningRate: params.learningRate, maxDepth: params.maxDepth || 3 });
            m.initialPrediction = params.initialPrediction;
            m.nFeatures = params.nFeatures;
            m.trees = params.trees.map(tData => {
                const sub = new DecisionTreeRegressor({ maxDepth: tData.maxDepth });
                sub.tree = deserializeTree(tData.tree);
                return sub;
            });
            return m;
        }
        case 'LR': {
            const m = new LogisticRegression({ maxIter: params.maxIter });
            m.weights = params.weights.map(w => ({ w: w.w, b: w.b }));
            m.classes = params.classes;
            if (params.nClasses) m.nClasses = params.nClasses;
            return m;
        }
        case 'NB': {
            const m = new GaussianNaiveBayes({});
            m.classPriors = params.classPriors;
            m.means = params.means;
            m.variances = params.variances;
            m.classes = params.classes;
            if (params.nClasses) m.nClasses = params.nClasses;
            return m;
        }
        case 'SVM': {
            const m = new SVMClassifier({ C: params.C, maxIter: params.maxIter });
            m.weights = params.weights.map(w => ({ w: w.w, b: w.b }));
            m.classes = params.classes;
            if (params.nClasses) m.nClasses = params.nClasses;
            return m;
        }
        default:
            throw new Error(`未対応のモデルタイプ: ${badge}`);
    }
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

export function render(container, _data, _characteristics) {
    container.innerHTML = `
        <div style="max-width: 900px; margin: 0 auto;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 1.8rem; font-weight: 700; color: ${THEME};">
                    <i class="fas fa-crosshairs" style="margin-right: 0.5rem;"></i>
                    予測モード
                </h2>
                <p style="color: #64748b; font-size: 0.95rem;">
                    easyDataScienceで作成したモデル（JSONファイル）を読み込んで、新しいデータで予測を実行できます
                </p>
            </div>

            <div id="pm-upload-area" style="
                border: 2px dashed #c7d2fe; border-radius: 16px; padding: 3rem 2rem;
                text-align: center; background: #f5f3ff; cursor: pointer;
                transition: all 0.3s ease; margin-bottom: 2rem;
            ">
                <i class="fas fa-file-import" style="font-size: 3rem; color: ${THEME}; margin-bottom: 1rem; display: block;"></i>
                <h3 style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">モデルJSONファイルをアップロード</h3>
                <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">
                    ドラッグ＆ドロップ、またはクリックしてファイルを選択
                </p>
                <input type="file" id="pm-file-input" accept=".json" style="display: none;">
                <button id="pm-select-btn" style="
                    background: ${THEME}; color: white; border: none; padding: 0.6rem 1.5rem;
                    border-radius: 8px; font-size: 0.9rem; font-weight: 500; cursor: pointer;
                "><i class="fas fa-upload"></i> ファイルを選択</button>
            </div>

            <div id="pm-model-info" style="display: none;"></div>
            <div id="pm-predict-form" style="display: none;"></div>
            <div id="pm-result" style="display: none;"></div>
        </div>
    `;

    let loadedModel = null;
    let loadedScaler = null;
    let loadedEncoders = null;
    let loadedLabelEncoder = null;
    let modelData = null;

    const uploadArea = container.querySelector('#pm-upload-area');
    const fileInput = container.querySelector('#pm-file-input');
    const selectBtn = container.querySelector('#pm-select-btn');

    // Drag & drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = THEME;
        uploadArea.style.background = '#ede9fe';
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#c7d2fe';
        uploadArea.style.background = '#f5f3ff';
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#c7d2fe';
        uploadArea.style.background = '#f5f3ff';
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.json')) processFile(file);
    });

    selectBtn.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', (e) => {
        if (e.target === uploadArea || ['I', 'H3', 'P'].includes(e.target.tagName)) fileInput.click();
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) processFile(fileInput.files[0]);
    });

    function processFile(file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                modelData = deserializeModel(ev.target.result);
                loadModelFromData(modelData);
            } catch (err) {
                showError(err.message);
            }
        };
        reader.readAsText(file);
    }

    function showError(message) {
        const infoDiv = container.querySelector('#pm-model-info');
        infoDiv.innerHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
                <p style="color: #dc2626; margin: 0;"><i class="fas fa-exclamation-triangle"></i> ${message}</p>
            </div>
        `;
        infoDiv.style.display = 'block';
        container.querySelector('#pm-predict-form').style.display = 'none';
        container.querySelector('#pm-result').style.display = 'none';
    }

    function loadModelFromData(data) {
        try {
            const taskType = data.taskType || 'regression';

            loadedScaler = reconstructScaler(data.preprocessing?.scaler);
            loadedEncoders = reconstructEncoders(data.preprocessing?.encoders);
            loadedLabelEncoder = reconstructLabelEncoder(data.preprocessing?.labelEncoder);
            loadedModel = reconstructModel(data.modelInfo, taskType);

            renderModelInfo(data, taskType);
            renderPredictForm(data, taskType);
        } catch (err) {
            showError(`モデル復元エラー: ${err.message}`);
            console.error('Model reconstruction error:', err);
        }
    }

    function renderModelInfo(data, taskType) {
        const infoDiv = container.querySelector('#pm-model-info');
        const taskLabel = taskType === 'regression' ? '回帰' : '分類';
        const exportDate = data.exportDate ? new Date(data.exportDate).toLocaleString('ja-JP') : '不明';

        infoDiv.innerHTML = `
            <div style="background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 1rem 0; color: ${THEME};">
                    <i class="fas fa-check-circle" style="color: #10b981;"></i> モデル読み込み完了
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div style="background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">モデル</div>
                        <div style="font-weight: 600;">${data.modelInfo.name} (${data.modelInfo.badge})</div>
                    </div>
                    <div style="background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">タスク</div>
                        <div style="font-weight: 600;">${taskLabel}</div>
                    </div>
                    <div style="background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">目的変数</div>
                        <div style="font-weight: 600;">${data.targetCol}</div>
                    </div>
                    <div style="background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">特徴量数</div>
                        <div style="font-weight: 600;">${data.featureNames.length}個</div>
                    </div>
                    <div style="background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">データセット</div>
                        <div style="font-weight: 600;">${data.datasetName}</div>
                    </div>
                    <div style="background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">エクスポート日</div>
                        <div style="font-weight: 600;">${exportDate}</div>
                    </div>
                </div>
                ${data.classLabels ? `
                    <div style="margin-top: 0.75rem; background: #f5f3ff; padding: 0.75rem 1rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #64748b;">クラス</div>
                        <div style="font-weight: 600;">${data.classLabels.join(', ')}</div>
                    </div>
                ` : ''}
            </div>
        `;
        infoDiv.style.display = 'block';
    }

    function renderPredictForm(data, taskType) {
        const formDiv = container.querySelector('#pm-predict-form');

        const featureInputs = data.featureNames.map((name, i) => {
            const encoder = loadedEncoders?.get(i);
            if (encoder) {
                const options = encoder._classes.map(c => `<option value="${c}">${c}</option>`).join('');
                return `
                    <div style="margin-bottom: 0.75rem;">
                        <label style="display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.25rem; color: #374151;">
                            ${name} <span style="color: #a78bfa; font-size: 0.75rem;">(カテゴリ)</span>
                        </label>
                        <select id="pm-feat-${i}" class="pm-input" style="
                            width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #d1d5db;
                            border-radius: 6px; font-size: 0.9rem; background: white;
                        ">${options}</select>
                    </div>
                `;
            }
            return `
                <div style="margin-bottom: 0.75rem;">
                    <label style="display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.25rem; color: #374151;">
                        ${name} <span style="color: #64748b; font-size: 0.75rem;">(数値)</span>
                    </label>
                    <input type="number" id="pm-feat-${i}" class="pm-input" step="any" style="
                        width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #d1d5db;
                        border-radius: 6px; font-size: 0.9rem; box-sizing: border-box;
                    " placeholder="値を入力">
                </div>
            `;
        }).join('');

        formDiv.innerHTML = `
            <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 1rem 0;">
                    <i class="fas fa-keyboard" style="color: ${THEME};"></i> 特徴量を入力して予測
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0 1.5rem;">
                    ${featureInputs}
                </div>
                <button id="pm-predict-btn" style="
                    background: ${THEME}; color: white; border: none; padding: 0.75rem 2rem;
                    border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer;
                    margin-top: 1rem; transition: all 0.3s ease;
                "><i class="fas fa-play"></i> 予測を実行</button>
            </div>
        `;
        formDiv.style.display = 'block';

        container.querySelector('#pm-predict-btn').addEventListener('click', () => {
            runPrediction(data, taskType);
        });
    }

    function runPrediction(data, taskType) {
        const resultDiv = container.querySelector('#pm-result');

        const inputValues = [];
        for (let i = 0; i < data.featureNames.length; i++) {
            const input = container.querySelector(`#pm-feat-${i}`);
            if (!input || input.value === '') {
                showResultError(resultDiv, 'すべての特徴量に値を入力してください。');
                return;
            }

            const encoder = loadedEncoders?.get(i);
            if (encoder) {
                const encoded = encoder._classToIndex.get(input.value);
                if (encoded === undefined) {
                    showResultError(resultDiv, `カテゴリ値「${input.value}」は無効です。`);
                    return;
                }
                inputValues.push(encoded);
            } else {
                const val = parseFloat(input.value);
                if (isNaN(val)) {
                    showResultError(resultDiv, `「${data.featureNames[i]}」に正しい数値を入力してください。`);
                    return;
                }
                inputValues.push(val);
            }
        }

        try {
            let processedInput = [inputValues];
            if (loadedScaler) {
                processedInput = loadedScaler.transform(processedInput);
            }

            const prediction = loadedModel.predict(processedInput);

            if (taskType === 'regression') {
                resultDiv.innerHTML = `
                    <div style="background: white; border-radius: 12px; padding: 2rem; margin-top: 1.5rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <p style="font-size: 0.9rem; color: #64748b;">予測結果 (${data.modelInfo.name})</p>
                        <p style="font-size: 3rem; font-weight: 700; color: ${THEME}; margin: 0.5rem 0;">
                            ${formatNumber(prediction[0], 4)}
                        </p>
                        <p style="font-size: 0.85rem; color: #64748b;">目的変数: ${data.targetCol}</p>
                    </div>
                `;
            } else {
                const predictedLabel = loadedLabelEncoder
                    ? loadedLabelEncoder.inverseTransform(prediction)[0]
                    : prediction[0];

                let probaHtml = '';
                if (loadedModel.predictProba) {
                    try {
                        const proba = loadedModel.predictProba(processedInput);
                        if (proba && proba[0]) {
                            const classLabels = data.classLabels || proba[0].map((_, i) => `Class ${i}`);
                            probaHtml = `
                                <div style="margin-top: 1.5rem; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;">
                                    <p style="font-weight: 600; margin-bottom: 0.5rem;">クラス別確率:</p>
                                    ${classLabels.map((label, idx) => {
                                        const p = proba[0][idx] || 0;
                                        return `<div style="display: flex; align-items: center; margin: 0.35rem 0;">
                                            <span style="width: 100px; font-size: 0.85rem; flex-shrink: 0;">${label}</span>
                                            <div style="flex: 1; background: #e2e8f0; border-radius: 4px; height: 22px; margin: 0 0.5rem;">
                                                <div style="width: ${(p * 100).toFixed(1)}%; background: ${THEME}; border-radius: 4px; height: 100%; transition: width 0.5s;"></div>
                                            </div>
                                            <span style="font-size: 0.85rem; width: 55px; text-align: right;">${(p * 100).toFixed(1)}%</span>
                                        </div>`;
                                    }).join('')}
                                </div>
                            `;
                        }
                    } catch (_) { /* predictProba may not work for all reconstructed models */ }
                }

                resultDiv.innerHTML = `
                    <div style="background: white; border-radius: 12px; padding: 2rem; margin-top: 1.5rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <p style="font-size: 0.9rem; color: #64748b;">予測結果 (${data.modelInfo.name})</p>
                        <p style="font-size: 3rem; font-weight: 700; color: ${THEME}; margin: 0.5rem 0;">
                            ${predictedLabel}
                        </p>
                        <p style="font-size: 0.85rem; color: #64748b;">目的変数: ${data.targetCol}</p>
                        ${probaHtml}
                    </div>
                `;
            }

            resultDiv.style.display = 'block';
        } catch (err) {
            showResultError(resultDiv, `予測エラー: ${err.message}`);
            console.error('Prediction error:', err);
        }
    }

    function showResultError(div, message) {
        div.innerHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                <p style="color: #dc2626; margin: 0;"><i class="fas fa-exclamation-triangle"></i> ${message}</p>
            </div>
        `;
        div.style.display = 'block';
    }
}
