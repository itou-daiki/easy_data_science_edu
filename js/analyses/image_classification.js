// ==========================================
// 画像分類 (Image Classification) Module
// Transfer Learning with MobileNet + TensorFlow.js
// ==========================================
import { createStepIndicator, formatNumber, renderPlot, renderConfusionMatrix } from '../utils.js';

const STEPS = ['データ準備', '学習', '評価', '予測'];
const THEME_COLOR = '#059669';
const THEME_COLOR_LIGHT = 'rgba(5, 150, 105, 0.08)';
const THEME_COLOR_BORDER = 'rgba(5, 150, 105, 0.3)';
const IMAGE_SIZE = 224;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

// ==========================================
// TensorFlow.js Access
// ==========================================

function getTf() {
    if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js が読み込まれていません。ページをリロードしてください。');
    }
    return tf;
}

function getMobileNet() {
    if (typeof mobilenet === 'undefined') {
        throw new Error('MobileNet ライブラリが読み込まれていません。ページをリロードしてください。');
    }
    return mobilenet;
}

// ==========================================
// Image Utilities
// ==========================================

async function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            URL.revokeObjectURL(url);
            resolve(img);
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error(`画像の読み込みに失敗しました: ${file.name}`));
        };
        img.src = url;
    });
}

function imageToTensor(img) {
    const tfLib = getTf();
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    return tfLib.browser.fromPixels(imageData).toFloat().div(255.0);
}

function createThumbnail(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = () => reject(new Error('サムネイルの作成に失敗しました'));
        reader.readAsDataURL(file);
    });
}

function isValidImageFile(file) {
    return ACCEPTED_TYPES.includes(file.type);
}

// ==========================================
// Feature Extraction
// ==========================================

async function extractFeatures(mobileNetModel, images) {
    const tfLib = getTf();
    const embeddings = [];
    for (const img of images) {
        const embedding = mobileNetModel.infer(img, true);
        embeddings.push(embedding);
    }
    return tfLib.concat(embeddings, 0);
}

// ==========================================
// Classifier
// ==========================================

function buildClassifier(inputShape, numClasses) {
    const tfLib = getTf();
    const model = tfLib.sequential();
    model.add(tfLib.layers.dense({
        inputShape: [inputShape],
        units: 128,
        activation: 'relu',
        kernelRegularizer: tfLib.regularizers.l2({ l2: 0.001 })
    }));
    model.add(tfLib.layers.dropout({ rate: 0.3 }));
    model.add(tfLib.layers.dense({
        units: numClasses,
        activation: 'softmax'
    }));
    model.compile({
        optimizer: tfLib.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

// ==========================================
// State Management (immutable pattern)
// ==========================================

function createInitialState() {
    return Object.freeze({
        classes: [],
        mobileNetModel: null,
        classifier: null,
        trainingHistory: null,
        isTraining: false,
        currentStep: 0,
        highestStep: 0
    });
}

function updateState(state, updates) {
    return Object.freeze({ ...state, ...updates });
}

// ==========================================
// UI Helpers
// ==========================================

function renderStepIndicator(container, activeIndex, highestStep = 0) {
    const el = container.querySelector('.step-indicator-wrapper');
    if (!el) return;
    el.innerHTML = createStepIndicator(STEPS, activeIndex);
    el.querySelectorAll('.step').forEach((stepEl, i) => {
        if (i <= highestStep && i !== activeIndex) {
            stepEl.style.cursor = 'pointer';
            stepEl.dataset.stepIndex = i;
            stepEl.classList.add('ic-clickable-step');
            stepEl.addEventListener('mouseenter', () => { stepEl.style.opacity = '0.7'; });
            stepEl.addEventListener('mouseleave', () => { stepEl.style.opacity = '1'; });
        }
    });
}

function showStep(container, stepIndex, highestStep) {
    const stepIds = ['#ic-step-setup', '#ic-step-training', '#ic-step-evaluate', '#ic-step-predict'];
    stepIds.forEach((id, i) => {
        const el = container.querySelector(id);
        if (el) el.style.display = i === stepIndex ? '' : 'none';
    });
    const summaryDiv = container.querySelector('#ic-data-preparation-summary');
    if (summaryDiv) {
        summaryDiv.style.display = (stepIndex > 0 && summaryDiv.innerHTML.trim()) ? 'block' : 'none';
    }
    renderStepIndicator(container, stepIndex, highestStep);
    if (stepIndex === 1) {
        const runBtn = container.querySelector('#ic-run-training-btn');
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="fas fa-cogs"></i> 学習を実行';
        }
    }
}

function showError(container, message) {
    const errorEl = container.querySelector('#ic-error-message');
    if (errorEl) {
        errorEl.innerHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
                        padding: 1rem; margin: 1rem 0; color: #dc2626;">
                <i class="fas fa-exclamation-triangle"></i> ${message}
            </div>`;
        setTimeout(() => { errorEl.innerHTML = ''; }, 8000);
    }
}

function showSuccess(container, message) {
    const errorEl = container.querySelector('#ic-error-message');
    if (errorEl) {
        errorEl.innerHTML = `
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
                        padding: 1rem; margin: 1rem 0; color: #16a34a;">
                <i class="fas fa-check-circle"></i> ${message}
            </div>`;
        setTimeout(() => { errorEl.innerHTML = ''; }, 5000);
    }
}

function createClassCard(classData, classIndex) {
    const thumbnailsHtml = classData.thumbnails
        .map((thumb, i) => `
            <div style="position: relative; display: inline-block;">
                <img src="${thumb}" style="width: 48px; height: 48px; object-fit: cover;
                     border-radius: 6px; border: 1px solid var(--border-color);"
                     alt="${classData.name} - ${i + 1}">
                <button class="ic-remove-image" data-class="${classIndex}" data-image="${i}"
                        style="position: absolute; top: -4px; right: -4px; width: 16px; height: 16px;
                               background: #ef4444; color: white; border: none; border-radius: 50%;
                               font-size: 10px; cursor: pointer; display: flex; align-items: center;
                               justify-content: center; line-height: 1; padding: 0;">
                    <i class="fas fa-times" style="font-size: 8px;"></i>
                </button>
            </div>`)
        .join('');

    return `
        <div class="ic-class-card" data-class-index="${classIndex}"
             style="background: var(--surface); border: 1px solid var(--border-color);
                    border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 32px; height: 32px; background: ${THEME_COLOR_LIGHT};
                                border-radius: 8px; display: flex; align-items: center;
                                justify-content: center; color: ${THEME_COLOR}; font-weight: 700; font-size: 0.85rem;">
                        ${classIndex + 1}
                    </div>
                    <input type="text" class="ic-class-name" data-class="${classIndex}"
                           value="${classData.name}" placeholder="クラス名を入力"
                           style="border: 1px solid var(--border-color); border-radius: 6px;
                                  padding: 0.4rem 0.75rem; font-size: 0.9rem; width: 180px;">
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 0.8rem; color: var(--text-secondary);">
                        <i class="fas fa-images"></i> ${classData.files.length} 枚
                    </span>
                    <button class="ic-remove-class" data-class="${classIndex}"
                            style="background: none; border: 1px solid #fecaca; color: #ef4444;
                                   border-radius: 6px; padding: 0.25rem 0.5rem; font-size: 0.75rem; cursor: pointer;">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>

            <div class="ic-drop-zone" data-class="${classIndex}"
                 style="border: 2px dashed var(--border-color); border-radius: 8px; padding: 1rem;
                        text-align: center; cursor: pointer; transition: all 0.3s ease;
                        background: var(--background); min-height: 60px;">
                ${classData.thumbnails.length > 0
                    ? `<div style="display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-start;">
                         ${thumbnailsHtml}
                       </div>`
                    : `<div style="color: var(--text-secondary); font-size: 0.85rem;">
                         <i class="fas fa-cloud-upload-alt"></i> 画像をドラッグ＆ドロップまたはクリック
                       </div>`}
                <input type="file" class="ic-file-input" data-class="${classIndex}"
                       multiple accept="image/jpeg,image/png,image/webp" style="display: none;">
            </div>
        </div>`;
}

// ==========================================
// Main Render Function
// ==========================================

export function render(container, _data, _characteristics) {
    let state = createInitialState();
    state = updateState(state, {
        classes: [
            { name: 'クラス 1', files: [], thumbnails: [], images: [] },
            { name: 'クラス 2', files: [], thumbnails: [], images: [] }
        ]
    });

    container.innerHTML = `
        <h2><i class="fas fa-image" style="color: ${THEME_COLOR};"></i> 画像分類 (Image Classification)</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            MobileNet の転移学習を使って、ブラウザ上で画像分類モデルを構築します。
            データのアップロードから学習・予測まですべてブラウザ内で完結します。
        </p>

        <div class="step-indicator-wrapper">${createStepIndicator(STEPS, 0)}</div>
        <div id="ic-error-message"></div>

        <!-- Data Preparation Summary (persists across steps) -->
        <div id="ic-data-preparation-summary" style="display: none;"></div>

        <!-- Step 1: Data Preparation -->
        <div id="ic-step-setup" class="model-config">
            <h3><i class="fas fa-folder-open" style="color: ${THEME_COLOR};"></i> Step 1: 学習データの準備</h3>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0.5rem 0 1rem;">
                各クラスに最低2枚ずつ画像をアップロードしてください。クラスは2つ以上必要です。
            </p>
            <div id="ic-classes-container"></div>
            <button id="ic-add-class-btn" style="background: none; border: 2px dashed ${THEME_COLOR_BORDER};
                    color: ${THEME_COLOR}; border-radius: 8px; padding: 0.6rem 1.25rem;
                    font-size: 0.85rem; cursor: pointer; width: 100%; margin-top: 0.5rem;
                    transition: all 0.3s ease;">
                <i class="fas fa-plus"></i> クラスを追加
            </button>
            <div id="ic-data-summary" style="margin-top: 1rem; padding: 0.75rem; background: var(--background);
                 border-radius: 8px; font-size: 0.85rem; color: var(--text-secondary); display: none;">
            </div>
            <button id="ic-start-training-btn" class="btn-analysis"
                    style="background: ${THEME_COLOR}; margin-top: 1.5rem;" disabled>
                <i class="fas fa-play"></i> 学習を開始する
            </button>
        </div>

        <!-- Step 2: Training -->
        <div id="ic-step-training" style="display: none;">
            <div class="model-config">
                <button class="ic-back-btn" data-target-step="0"
                        style="background: none; border: none; color: var(--text-secondary);
                               font-size: 0.85rem; cursor: pointer; padding: 0.25rem 0; margin-bottom: 0.75rem;">
                    <i class="fas fa-arrow-left" style="margin-right: 0.3rem;"></i>データ準備に戻る
                </button>
                <h3><i class="fas fa-brain" style="color: ${THEME_COLOR};"></i> Step 2: モデル学習</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                    <div>
                        <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">エポック数:</label>
                        <select id="ic-epochs-select" class="form-select">
                            <option value="5">5</option>
                            <option value="10" selected>10</option>
                            <option value="20">20</option>
                            <option value="50">50</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">検証データ割合:</label>
                        <select id="ic-validation-select" class="form-select">
                            <option value="0.1">10%</option>
                            <option value="0.2" selected>20%</option>
                            <option value="0.3">30%</option>
                        </select>
                    </div>
                </div>
                <div id="ic-training-status" style="margin-top: 1rem;">
                    <div id="ic-training-progress" style="display: none;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span id="ic-progress-label" style="font-weight: 500; font-size: 0.9rem;">
                                準備中...
                            </span>
                            <span id="ic-progress-percent" style="color: ${THEME_COLOR}; font-weight: 700;">
                                0%
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div id="ic-progress-fill" class="progress-fill" style="width: 0%;
                                 background: linear-gradient(90deg, ${THEME_COLOR}, #34d399);"></div>
                        </div>
                        <div id="ic-training-log" style="margin-top: 0.75rem; font-family: monospace;
                             font-size: 0.8rem; color: var(--text-secondary); max-height: 150px;
                             overflow-y: auto; background: var(--background); border-radius: 6px; padding: 0.75rem;">
                        </div>
                    </div>
                </div>
                <button id="ic-run-training-btn" class="btn-analysis"
                        style="background: ${THEME_COLOR}; margin-top: 1rem;">
                    <i class="fas fa-cogs"></i> 学習を実行
                </button>
            </div>
        </div>

        <!-- Step 3: Evaluation -->
        <div id="ic-step-evaluate" style="display: none;">
            <div class="model-config">
                <button class="ic-back-btn" data-target-step="1"
                        style="background: none; border: none; color: var(--text-secondary);
                               font-size: 0.85rem; cursor: pointer; padding: 0.25rem 0; margin-bottom: 0.75rem;">
                    <i class="fas fa-arrow-left" style="margin-right: 0.3rem;"></i>学習設定に戻る
                </button>
                <h3><i class="fas fa-chart-bar" style="color: ${THEME_COLOR};"></i> Step 3: 評価結果</h3>
                <div class="metrics-grid" id="ic-metrics-grid"></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;">
                    <div>
                        <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem;">
                            <i class="fas fa-chart-line" style="color: ${THEME_COLOR};"></i> 学習曲線
                        </h4>
                        <div id="ic-loss-chart" style="width: 100%; height: 300px;"></div>
                    </div>
                    <div>
                        <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem;">
                            <i class="fas fa-chart-area" style="color: ${THEME_COLOR};"></i> 精度曲線
                        </h4>
                        <div id="ic-accuracy-chart" style="width: 100%; height: 300px;"></div>
                    </div>
                </div>
                <div id="ic-per-class-section" style="margin-top: 1.5rem;">
                    <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-table" style="color: ${THEME_COLOR};"></i> クラス別精度
                    </h4>
                    <div id="ic-per-class-table"></div>
                </div>
                <div id="ic-confusion-matrix-section" style="margin-top: 1.5rem;">
                    <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-th" style="color: ${THEME_COLOR};"></i> 混同行列
                    </h4>
                    <div id="ic-confusion-matrix" style="max-width: 500px; margin: 0 auto;"></div>
                </div>
                <button id="ic-go-predict-btn" class="btn-analysis"
                        style="background: ${THEME_COLOR}; margin-top: 1.5rem;">
                    <i class="fas fa-arrow-right"></i> 予測に進む
                </button>
            </div>
        </div>

        <!-- Step 4: Prediction -->
        <div id="ic-step-predict" style="display: none;">
            <div class="model-config">
                <button class="ic-back-btn" data-target-step="2"
                        style="background: none; border: none; color: var(--text-secondary);
                               font-size: 0.85rem; cursor: pointer; padding: 0.25rem 0; margin-bottom: 0.75rem;">
                    <i class="fas fa-arrow-left" style="margin-right: 0.3rem;"></i>評価結果に戻る
                </button>
                <h3><i class="fas fa-magic" style="color: ${THEME_COLOR};"></i> Step 4: 新しい画像を分類</h3>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">
                    分類したい画像をアップロードしてください。学習済みモデルで予測を行います。
                </p>
                <div id="ic-predict-drop-zone"
                     style="border: 2px dashed ${THEME_COLOR_BORDER}; border-radius: 12px; padding: 2rem;
                            text-align: center; cursor: pointer; transition: all 0.3s ease;
                            background: var(--background);">
                    <i class="fas fa-cloud-upload-alt" style="font-size: 2rem; color: ${THEME_COLOR}; margin-bottom: 0.5rem;"></i>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">
                        画像をドラッグ＆ドロップ、またはクリックしてファイルを選択
                    </p>
                    <input type="file" id="ic-predict-file-input" accept="image/jpeg,image/png,image/webp" style="display: none;">
                </div>
                <div id="ic-prediction-result" style="display: none; margin-top: 1.5rem;"></div>
            </div>
        </div>
    `;

    // --- Initialize UI ---
    const getState = () => state;
    const setState = (newState) => { state = newState; };
    renderClassCards(container, state);
    setupSetupEventListeners(container, getState, setState);

    // --- Setup: Add Class Button ---
    const addClassBtn = container.querySelector('#ic-add-class-btn');
    addClassBtn.addEventListener('click', () => {
        const newClasses = [
            ...state.classes,
            { name: `クラス ${state.classes.length + 1}`, files: [], thumbnails: [], images: [] }
        ];
        state = updateState(state, { classes: newClasses });
        renderClassCards(container, state);
        setupSetupEventListeners(container, getState, setState);
        updateDataSummary(container, state);
    });

    // --- Step Navigation Helpers ---
    let predictionListenersAttached = false;

    function handleStepChange(targetStep) {
        if (state.isTraining) return;
        state = updateState(state, { currentStep: targetStep });
        showStep(container, targetStep, state.highestStep);
        attachStepClickHandlers();
        if (targetStep === 3 && !predictionListenersAttached && state.classifier) {
            setupPredictionListeners(container, getState);
            predictionListenersAttached = true;
        }
    }

    function attachStepClickHandlers() {
        container.querySelectorAll('.ic-clickable-step').forEach(stepEl => {
            stepEl.addEventListener('click', () => {
                handleStepChange(parseInt(stepEl.dataset.stepIndex, 10));
            });
        });
    }

    // Back buttons (static elements, set up once)
    container.querySelectorAll('.ic-back-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            handleStepChange(parseInt(btn.dataset.targetStep, 10));
        });
    });

    // --- Setup: Start Training Button ---
    const startTrainingBtn = container.querySelector('#ic-start-training-btn');
    startTrainingBtn.addEventListener('click', () => {
        state = updateState(state, { highestStep: Math.max(state.highestStep, 1) });
        renderDataPreparationSummary(container, state);
        handleStepChange(1);
    });

    // --- Training: Run Training Button ---
    const runTrainingBtn = container.querySelector('#ic-run-training-btn');
    runTrainingBtn.addEventListener('click', async () => {
        if (state.isTraining) return;
        state = updateState(state, { isTraining: true });
        runTrainingBtn.disabled = true;
        runTrainingBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 学習中...';

        try {
            const result = await runTraining(container, state);
            state = updateState(state, {
                mobileNetModel: result.mobileNetModel,
                classifier: result.classifier,
                trainingHistory: result.history,
                isTraining: false,
                currentStep: 2,
                highestStep: Math.max(state.highestStep, 2)
            });
            showStep(container, 2, state.highestStep);
            attachStepClickHandlers();
            renderEvaluation(container, state);
        } catch (error) {
            console.error('Training error:', error);
            showError(container, `学習中にエラーが発生しました: ${error.message}`);
            state = updateState(state, { isTraining: false });
            runTrainingBtn.disabled = false;
            runTrainingBtn.innerHTML = '<i class="fas fa-cogs"></i> 学習を実行';
        }
    });

    // --- Evaluation: Go to Predict ---
    const goPredictBtn = container.querySelector('#ic-go-predict-btn');
    goPredictBtn.addEventListener('click', () => {
        state = updateState(state, { highestStep: Math.max(state.highestStep, 3) });
        handleStepChange(3);
    });
}

// ==========================================
// Class Cards Rendering & Event Listeners
// ==========================================

function renderClassCards(container, state) {
    const classesContainer = container.querySelector('#ic-classes-container');
    classesContainer.innerHTML = state.classes.map((cls, i) => createClassCard(cls, i)).join('');
}

function setupSetupEventListeners(container, getState, setState) {
    // Drop zones
    container.querySelectorAll('.ic-drop-zone').forEach(zone => {
        const classIdx = parseInt(zone.dataset.class, 10);

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.style.borderColor = THEME_COLOR;
            zone.style.background = THEME_COLOR_LIGHT;
        });

        zone.addEventListener('dragleave', () => {
            zone.style.borderColor = 'var(--border-color)';
            zone.style.background = 'var(--background)';
        });

        zone.addEventListener('drop', async (e) => {
            e.preventDefault();
            zone.style.borderColor = 'var(--border-color)';
            zone.style.background = 'var(--background)';
            const files = Array.from(e.dataTransfer.files).filter(isValidImageFile);
            if (files.length === 0) {
                showError(container, '対応する画像形式（JPG, PNG, WEBP）のみアップロードできます。');
                return;
            }
            const newState = await addImagesToClass(getState, classIdx, files);
            setState(newState);
            renderClassCards(container, newState);
            setupSetupEventListeners(container, () => getState(), setState);
            updateDataSummary(container, newState);
        });

        zone.addEventListener('click', (e) => {
            if (e.target.closest('.ic-remove-image')) return;
            const fileInput = zone.querySelector('.ic-file-input');
            fileInput.click();
        });

        const fileInput = zone.querySelector('.ic-file-input');
        fileInput.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files).filter(isValidImageFile);
            if (files.length === 0) return;
            const newState = await addImagesToClass(getState, classIdx, files);
            setState(newState);
            renderClassCards(container, newState);
            setupSetupEventListeners(container, () => getState(), setState);
            updateDataSummary(container, newState);
        });
    });

    // Remove image buttons
    container.querySelectorAll('.ic-remove-image').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const classIdx = parseInt(btn.dataset.class, 10);
            const imageIdx = parseInt(btn.dataset.image, 10);
            const currentState = getState();
            const cls = currentState.classes[classIdx];
            const newCls = {
                ...cls,
                files: cls.files.filter((_, i) => i !== imageIdx),
                thumbnails: cls.thumbnails.filter((_, i) => i !== imageIdx),
                images: cls.images.filter((_, i) => i !== imageIdx)
            };
            const newClasses = currentState.classes.map((c, i) => i === classIdx ? newCls : c);
            const newState = updateState(currentState, { classes: newClasses });
            setState(newState);
            renderClassCards(container, newState);
            setupSetupEventListeners(container, () => getState(), setState);
            updateDataSummary(container, newState);
        });
    });

    // Remove class buttons
    container.querySelectorAll('.ic-remove-class').forEach(btn => {
        btn.addEventListener('click', () => {
            const classIdx = parseInt(btn.dataset.class, 10);
            const currentState = getState();
            if (currentState.classes.length <= 2) {
                showError(container, 'クラスは最低2つ必要です。');
                return;
            }
            const newClasses = currentState.classes.filter((_, i) => i !== classIdx);
            const newState = updateState(currentState, { classes: newClasses });
            setState(newState);
            renderClassCards(container, newState);
            setupSetupEventListeners(container, () => getState(), setState);
            updateDataSummary(container, newState);
        });
    });

    // Class name inputs — use 'input' event for immediate updates
    container.querySelectorAll('.ic-class-name').forEach(input => {
        input.addEventListener('input', () => {
            const classIdx = parseInt(input.dataset.class, 10);
            const currentState = getState();
            const newClasses = currentState.classes.map((c, i) =>
                i === classIdx ? { ...c, name: input.value || `クラス ${classIdx + 1}` } : c
            );
            const newState = updateState(currentState, { classes: newClasses });
            setState(newState);
        });
    });
}

async function addImagesToClass(getState, classIdx, files) {
    // Perform async I/O first
    const newThumbnails = await Promise.all(files.map(f => createThumbnail(f)));
    const newImages = await Promise.all(files.map(f => loadImageFromFile(f)));
    // Re-read state AFTER async work to avoid losing concurrent updates
    const freshState = getState();
    const cls = freshState.classes[classIdx];
    const updatedClass = {
        ...cls,
        files: [...cls.files, ...files],
        thumbnails: [...cls.thumbnails, ...newThumbnails],
        images: [...cls.images, ...newImages]
    };
    const newClasses = freshState.classes.map((c, i) => i === classIdx ? updatedClass : c);
    return updateState(freshState, { classes: newClasses });
}

function updateDataSummary(container, state) {
    const summaryEl = container.querySelector('#ic-data-summary');
    const startBtn = container.querySelector('#ic-start-training-btn');
    const validClasses = state.classes.filter(c => c.files.length >= 2);
    const totalImages = state.classes.reduce((sum, c) => sum + c.files.length, 0);
    const isReady = validClasses.length >= 2 && state.classes.every(c => c.files.length >= 2);

    if (totalImages > 0) {
        summaryEl.style.display = 'block';
        const classDetails = state.classes
            .map(c => `${c.name}: ${c.files.length}枚`)
            .join(' / ');

        const warnings = [];
        state.classes.forEach(c => {
            if (c.files.length < 2) {
                warnings.push(`「${c.name}」に画像が不足しています（最低2枚）`);
            }
        });

        summaryEl.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <i class="fas fa-info-circle" style="color: ${THEME_COLOR};"></i>
                <strong>データ概要:</strong> ${state.classes.length}クラス / 合計 ${totalImages}枚
            </div>
            <div style="font-size: 0.8rem; margin-left: 1.5rem;">${classDetails}</div>
            ${warnings.length > 0 ? `
                <div style="color: #dc2626; font-size: 0.8rem; margin-top: 0.5rem; margin-left: 1.5rem;">
                    <i class="fas fa-exclamation-circle"></i> ${warnings.join('、')}
                </div>` : ''}
        `;
    } else {
        summaryEl.style.display = 'none';
    }

    startBtn.disabled = !isReady;
    if (isReady) {
        startBtn.style.opacity = '1';
    } else {
        startBtn.style.opacity = '0.5';
    }
}

// ==========================================
// Data Preparation Summary (persistent)
// ==========================================

function renderDataPreparationSummary(container, state) {
    const summaryDiv = container.querySelector('#ic-data-preparation-summary');
    const totalImages = state.classes.reduce((sum, c) => sum + c.files.length, 0);

    const classCards = state.classes.map(cls => {
        const thumbs = cls.thumbnails.slice(0, 6).map(thumb =>
            `<img src="${thumb}" style="width: 36px; height: 36px; object-fit: cover;
                  border-radius: 4px; border: 1px solid var(--border-color);">`
        ).join('');
        const extra = cls.thumbnails.length > 6
            ? `<span style="font-size: 0.75rem; color: var(--text-secondary);">+${cls.thumbnails.length - 6}</span>`
            : '';
        return `
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0;">
                <span style="font-weight: 600; min-width: 80px; color: ${THEME_COLOR};">${cls.name}</span>
                <div style="display: flex; gap: 3px; flex-wrap: wrap; align-items: center;">
                    ${thumbs}${extra}
                </div>
                <span style="font-size: 0.8rem; color: var(--text-secondary); margin-left: auto;">${cls.files.length}枚</span>
            </div>`;
    }).join('');

    summaryDiv.style.display = 'block';
    summaryDiv.innerHTML = `
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
                    padding: 1.25rem; margin-bottom: 1.5rem;">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1.1rem; color: #166534;">
                <i class="fas fa-check-circle" style="margin-right: 0.5rem;"></i>
                Step 1: データ準備 (完了)
            </h3>
            <div style="display: flex; gap: 1.5rem; margin-bottom: 0.75rem; font-size: 0.9rem;">
                <span><i class="fas fa-layer-group" style="color: ${THEME_COLOR}; margin-right: 0.3rem;"></i>
                    <strong>${state.classes.length}</strong> クラス</span>
                <span><i class="fas fa-images" style="color: ${THEME_COLOR}; margin-right: 0.3rem;"></i>
                    合計 <strong>${totalImages}</strong> 枚</span>
            </div>
            ${classCards}
        </div>
    `;
}

// ==========================================
// Training
// ==========================================

async function runTraining(container, state) {
    const tfLib = getTf();
    const progressEl = container.querySelector('#ic-training-progress');
    const progressLabel = container.querySelector('#ic-progress-label');
    const progressPercent = container.querySelector('#ic-progress-percent');
    const progressFill = container.querySelector('#ic-progress-fill');
    const trainingLog = container.querySelector('#ic-training-log');

    progressEl.style.display = 'block';
    trainingLog.innerHTML = '';

    const log = (msg) => {
        trainingLog.innerHTML += msg + '<br>';
        trainingLog.scrollTop = trainingLog.scrollHeight;
    };

    // Load MobileNet
    progressLabel.textContent = 'MobileNet を読み込み中...';
    progressPercent.textContent = '10%';
    progressFill.style.width = '10%';
    log('[INFO] MobileNet モデルを読み込んでいます...');

    const mobileNetLib = getMobileNet();
    const mobileNetModel = await mobileNetLib.load({ version: 2, alpha: 1.0 });
    log('[OK] MobileNet の読み込み完了');

    // Extract features
    progressLabel.textContent = '特徴量を抽出中...';
    progressPercent.textContent = '30%';
    progressFill.style.width = '30%';
    log('[INFO] 画像から特徴量を抽出しています...');

    const allFeatures = [];
    const allLabels = [];
    const classNames = state.classes.map(c => c.name);
    const numClasses = classNames.length;

    for (let ci = 0; ci < state.classes.length; ci++) {
        const cls = state.classes[ci];
        for (const img of cls.images) {
            const embedding = mobileNetModel.infer(img, true);
            allFeatures.push(embedding);
            allLabels.push(ci);
        }
        log(`[OK] ${cls.name}: ${cls.images.length}枚の特徴量抽出完了`);
    }

    const featuresTensor = tfLib.concat(allFeatures, 0);
    const featureShape = featuresTensor.shape[1];

    // Create one-hot labels
    const labelsTensor = tfLib.oneHot(tfLib.tensor1d(allLabels, 'int32'), numClasses);

    log(`[INFO] 特徴量サイズ: ${featuresTensor.shape}, ラベル数: ${allLabels.length}`);

    // Shuffle data
    const numSamples = allLabels.length;
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
    const shuffledIndices = tfLib.tensor1d(indices, 'int32');
    const shuffledFeatures = tfLib.gather(featuresTensor, shuffledIndices);
    const shuffledLabels = tfLib.gather(labelsTensor, shuffledIndices);

    // Split validation
    const valRatio = parseFloat(container.querySelector('#ic-validation-select').value);
    const valSize = Math.max(1, Math.floor(numSamples * valRatio));
    const trainSize = numSamples - valSize;

    const trainFeatures = shuffledFeatures.slice(0, trainSize);
    const trainLabels = shuffledLabels.slice(0, trainSize);
    const valFeatures = shuffledFeatures.slice(trainSize);
    const valLabels = shuffledLabels.slice(trainSize);

    // Build classifier
    progressLabel.textContent = 'モデルを構築中...';
    progressPercent.textContent = '40%';
    progressFill.style.width = '40%';

    const classifier = buildClassifier(featureShape, numClasses);
    log('[OK] 分類モデルの構築完了');

    // Train
    const epochs = parseInt(container.querySelector('#ic-epochs-select').value, 10);
    progressLabel.textContent = '学習中...';
    log(`[INFO] 学習開始 (エポック: ${epochs}, 訓練: ${trainSize}, 検証: ${valSize})`);

    const history = { loss: [], accuracy: [], val_loss: [], val_accuracy: [] };

    await classifier.fit(trainFeatures, trainLabels, {
        epochs,
        batchSize: Math.min(32, trainSize),
        validationData: [valFeatures, valLabels],
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const progress = 40 + Math.round(((epoch + 1) / epochs) * 55);
                progressPercent.textContent = `${progress}%`;
                progressFill.style.width = `${progress}%`;
                progressLabel.textContent = `学習中... (Epoch ${epoch + 1}/${epochs})`;

                history.loss.push(logs.loss);
                history.accuracy.push(logs.acc);
                history.val_loss.push(logs.val_loss);
                history.val_accuracy.push(logs.val_acc);

                log(`Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, ` +
                    `acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, ` +
                    `val_acc: ${logs.val_acc.toFixed(4)}`);
            }
        }
    });

    progressLabel.textContent = '学習完了！';
    progressPercent.textContent = '100%';
    progressFill.style.width = '100%';
    log('[OK] 学習が完了しました');

    // Compute per-class accuracy and confusion matrix for evaluation
    const valPredTensor = classifier.predict(valFeatures);
    const valPredIndices = valPredTensor.argMax(1).arraySync();
    const valTrueIndices = valLabels.argMax(1).arraySync();

    // Cleanup intermediate tensors
    featuresTensor.dispose();
    labelsTensor.dispose();
    shuffledIndices.dispose();
    shuffledFeatures.dispose();
    shuffledLabels.dispose();
    trainFeatures.dispose();
    trainLabels.dispose();
    valFeatures.dispose();
    valLabels.dispose();
    valPredTensor.dispose();
    allFeatures.forEach(t => t.dispose());

    return {
        mobileNetModel,
        classifier,
        history: {
            ...history,
            classNames,
            numClasses,
            featureShape,
            valPredIndices,
            valTrueIndices,
            trainSize,
            valSize
        }
    };
}

// ==========================================
// Evaluation Rendering
// ==========================================

function renderEvaluation(container, state) {
    const history = state.trainingHistory;
    const classNames = history.classNames;

    // Metrics
    const finalAcc = history.accuracy[history.accuracy.length - 1];
    const finalValAcc = history.val_accuracy[history.val_accuracy.length - 1];
    const finalLoss = history.loss[history.loss.length - 1];
    const finalValLoss = history.val_loss[history.val_loss.length - 1];

    const metricsGrid = container.querySelector('#ic-metrics-grid');
    metricsGrid.innerHTML = `
        <div class="metric-card">
            <div class="metric-label">訓練精度</div>
            <div class="metric-value" style="color: ${THEME_COLOR};">${formatNumber(finalAcc, 4)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">検証精度</div>
            <div class="metric-value" style="color: ${THEME_COLOR};">${formatNumber(finalValAcc, 4)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">訓練損失</div>
            <div class="metric-value" style="color: #f59e0b;">${formatNumber(finalLoss, 4)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">検証損失</div>
            <div class="metric-value" style="color: #f59e0b;">${formatNumber(finalValLoss, 4)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">訓練データ数</div>
            <div class="metric-value" style="color: var(--text-primary);">${history.trainSize}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">検証データ数</div>
            <div class="metric-value" style="color: var(--text-primary);">${history.valSize}</div>
        </div>
    `;

    // Loss chart
    const epochsArr = history.loss.map((_, i) => i + 1);
    renderPlot('ic-loss-chart', [
        {
            x: epochsArr, y: history.loss,
            mode: 'lines+markers', name: '訓練損失',
            line: { color: THEME_COLOR, width: 2 }, marker: { size: 5 }
        },
        {
            x: epochsArr, y: history.val_loss,
            mode: 'lines+markers', name: '検証損失',
            line: { color: '#ef4444', width: 2, dash: 'dash' }, marker: { size: 5 }
        }
    ], {
        title: '損失 (Loss)',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss' },
        height: 300,
        legend: { x: 0.6, y: 0.95 }
    });

    // Accuracy chart
    renderPlot('ic-accuracy-chart', [
        {
            x: epochsArr, y: history.accuracy,
            mode: 'lines+markers', name: '訓練精度',
            line: { color: THEME_COLOR, width: 2 }, marker: { size: 5 }
        },
        {
            x: epochsArr, y: history.val_accuracy,
            mode: 'lines+markers', name: '検証精度',
            line: { color: '#3b82f6', width: 2, dash: 'dash' }, marker: { size: 5 }
        }
    ], {
        title: '精度 (Accuracy)',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Accuracy', range: [0, 1.05] },
        height: 300,
        legend: { x: 0.6, y: 0.1 }
    });

    // Per-class accuracy
    renderPerClassAccuracy(container, history);

    // Confusion matrix
    renderConfusionMatrixSection(container, history);
}

function renderPerClassAccuracy(container, history) {
    const classNames = history.classNames;
    const valPred = history.valPredIndices;
    const valTrue = history.valTrueIndices;

    const perClass = classNames.map((name, ci) => {
        const trueForClass = valTrue.filter((v) => v === ci).length;
        const correctForClass = valTrue.filter((v, i) => v === ci && valPred[i] === ci).length;
        const acc = trueForClass > 0 ? correctForClass / trueForClass : 0;
        return { name, total: trueForClass, correct: correctForClass, accuracy: acc };
    });

    const tableEl = container.querySelector('#ic-per-class-table');
    let html = '<div class="table-container"><table class="table">';
    html += '<thead><tr><th>クラス</th><th>検証データ数</th><th>正解数</th><th>精度</th></tr></thead><tbody>';
    perClass.forEach(pc => {
        html += `<tr>
            <td><strong>${pc.name}</strong></td>
            <td>${pc.total}</td>
            <td>${pc.correct}</td>
            <td>${formatNumber(pc.accuracy, 4)}</td>
        </tr>`;
    });
    html += '</tbody></table></div>';
    tableEl.innerHTML = html;
}

function renderConfusionMatrixSection(container, history) {
    const classNames = history.classNames;
    const numClasses = classNames.length;
    const valPred = history.valPredIndices;
    const valTrue = history.valTrueIndices;

    const matrix = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    valTrue.forEach((t, i) => {
        matrix[t][valPred[i]]++;
    });

    renderConfusionMatrix('ic-confusion-matrix', matrix, classNames);
}

// ==========================================
// Prediction
// ==========================================

function setupPredictionListeners(container, getState) {
    const dropZone = container.querySelector('#ic-predict-drop-zone');
    const fileInput = container.querySelector('#ic-predict-file-input');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = THEME_COLOR;
        dropZone.style.background = THEME_COLOR_LIGHT;
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = THEME_COLOR_BORDER;
        dropZone.style.background = 'var(--background)';
    });

    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropZone.style.borderColor = THEME_COLOR_BORDER;
        dropZone.style.background = 'var(--background)';
        const file = Array.from(e.dataTransfer.files).find(isValidImageFile);
        if (!file) {
            showError(container, '対応する画像形式（JPG, PNG, WEBP）のみ使用できます。');
            return;
        }
        await predictImage(container, getState(), file);
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (!isValidImageFile(file)) {
            showError(container, '対応する画像形式（JPG, PNG, WEBP）のみ使用できます。');
            return;
        }
        await predictImage(container, getState(), file);
        fileInput.value = '';
    });
}

async function predictImage(container, state, file) {
    const resultEl = container.querySelector('#ic-prediction-result');

    if (!state.mobileNetModel || !state.classifier) {
        showError(container, 'モデルが読み込まれていません。先に学習を完了してください。');
        return;
    }

    resultEl.style.display = 'block';
    resultEl.innerHTML = `
        <div style="text-align: center; padding: 1rem; color: var(--text-secondary);">
            <i class="fas fa-spinner fa-spin"></i> 予測中...
        </div>`;

    try {
        const img = await loadImageFromFile(file);
        const thumbnailSrc = await createThumbnail(file);

        // Extract features and predict
        const embedding = state.mobileNetModel.infer(img, true);
        const prediction = state.classifier.predict(embedding);
        const probabilities = prediction.arraySync()[0];
        const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
        const classNames = state.trainingHistory.classNames;
        const predictedClass = classNames[predictedIndex];
        const confidence = probabilities[predictedIndex];

        // Cleanup tensors
        embedding.dispose();
        prediction.dispose();

        // Sort class probabilities descending
        const sortedProbs = classNames
            .map((name, i) => ({ name, probability: probabilities[i] }))
            .sort((a, b) => b.probability - a.probability);

        resultEl.innerHTML = `
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 1.5rem; align-items: start;">
                <div style="text-align: center;">
                    <img src="${thumbnailSrc}"
                         style="width: 180px; height: 180px; object-fit: cover; border-radius: 12px;
                                border: 2px solid var(--border-color); box-shadow: var(--shadow-md);">
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">${file.name}</p>
                </div>
                <div>
                    <div style="background: ${THEME_COLOR_LIGHT}; border: 1px solid ${THEME_COLOR_BORDER};
                                border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.25rem;">予測結果</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: ${THEME_COLOR};">
                            ${predictedClass}
                        </div>
                        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                            信頼度: ${(confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;">クラス別確率:</div>
                    ${sortedProbs.map(sp => `
                        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem;">
                            <span style="width: 80px; font-size: 0.8rem; text-align: right; color: var(--text-secondary);">
                                ${sp.name}
                            </span>
                            <div style="flex: 1; height: 20px; background: var(--border-color); border-radius: 4px; overflow: hidden;">
                                <div style="width: ${(sp.probability * 100).toFixed(1)}%; height: 100%;
                                            background: ${sp.name === predictedClass ? THEME_COLOR : '#94a3b8'};
                                            border-radius: 4px; transition: width 0.5s ease;"></div>
                            </div>
                            <span style="width: 50px; font-size: 0.8rem; font-weight: 600; color: var(--text-primary);">
                                ${(sp.probability * 100).toFixed(1)}%
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div id="ic-probability-chart" style="margin-top: 1rem;"></div>
        `;

        // Render probability bar chart
        renderPlot('ic-probability-chart', [{
            type: 'bar',
            x: sortedProbs.map(sp => sp.name),
            y: sortedProbs.map(sp => sp.probability),
            marker: {
                color: sortedProbs.map(sp =>
                    sp.name === predictedClass ? THEME_COLOR : '#94a3b8'
                )
            },
            text: sortedProbs.map(sp => `${(sp.probability * 100).toFixed(1)}%`),
            textposition: 'outside',
            hovertemplate: '%{x}: %{y:.4f}<extra></extra>'
        }], {
            title: 'クラス別予測確率',
            yaxis: { title: '確率', range: [0, 1.1] },
            height: 300,
            margin: { t: 40, b: 60 }
        });

    } catch (error) {
        console.error('Prediction error:', error);
        resultEl.innerHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
                        padding: 1rem; color: #dc2626;">
                <i class="fas fa-exclamation-triangle"></i> 予測中にエラーが発生しました: ${error.message}
            </div>`;
    }
}
