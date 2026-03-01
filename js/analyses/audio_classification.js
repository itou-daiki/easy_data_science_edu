// ==========================================
// 音声分類 (Audio Classification) Module
// Web Audio API + TensorFlow.js によるブラウザ音声分類
// ==========================================
import { createStepIndicator, formatNumber, renderPlot } from '../utils.js';

const STEPS = ['データ準備', '学習', '評価', '予測'];
const ACCENT = '#7c3aed';
const ACCENT_LIGHT = '#ede9fe';
const ACCENT_BORDER = '#c4b5fd';
const RECORDING_DURATION_MS = 3000;
const NUM_FRAMES = 20;
const FEATURES_PER_FRAME = 4;
const FEATURE_DIM = NUM_FRAMES * FEATURES_PER_FRAME; // 80

// ==========================================
// TensorFlow.js Loader
// ==========================================

async function ensureTensorFlow() {
    if (typeof tf !== 'undefined') return;
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js';
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('TensorFlow.js の読み込みに失敗しました。'));
        document.head.appendChild(script);
    });
}

// ==========================================
// FFT Implementation (radix-2)
// ==========================================

function simpleFFT(signal) {
    const n = Math.pow(2, Math.ceil(Math.log2(Math.max(signal.length, 2))));
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < signal.length; i++) re[i] = signal[i];

    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    // Cooley-Tukey iterative FFT
    for (let len = 2; len <= n; len *= 2) {
        const halfLen = len / 2;
        const angle = -2 * Math.PI / len;
        const wRe = Math.cos(angle);
        const wIm = Math.sin(angle);
        for (let i = 0; i < n; i += len) {
            let curRe = 1, curIm = 0;
            for (let j = 0; j < halfLen; j++) {
                const tRe = curRe * re[i + j + halfLen] - curIm * im[i + j + halfLen];
                const tIm = curRe * im[i + j + halfLen] + curIm * re[i + j + halfLen];
                re[i + j + halfLen] = re[i + j] - tRe;
                im[i + j + halfLen] = im[i + j] - tIm;
                re[i + j] += tRe;
                im[i + j] += tIm;
                const newCurRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = newCurRe;
            }
        }
    }

    const result = [];
    for (let i = 0; i < n; i++) {
        result.push({ re: re[i], im: im[i] });
    }
    return result;
}

// ==========================================
// Audio Feature Extraction
// ==========================================

function extractAudioFeatures(audioBuffer) {
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const frameSize = Math.floor(sampleRate * 0.025); // 25ms frames
    const totalLength = channelData.length;

    if (totalLength < frameSize) {
        return new Array(FEATURE_DIM).fill(0);
    }

    const frameStep = Math.max(1, Math.floor((totalLength - frameSize) / (NUM_FRAMES - 1)));
    const features = [];

    for (let i = 0; i < NUM_FRAMES; i++) {
        const start = Math.min(i * frameStep, totalLength - frameSize);
        const frame = channelData.slice(start, start + frameSize);

        // RMS energy
        let sumSq = 0;
        for (let j = 0; j < frame.length; j++) sumSq += frame[j] * frame[j];
        const rms = Math.sqrt(sumSq / frame.length);

        // Zero crossing rate
        let zcr = 0;
        for (let j = 1; j < frame.length; j++) {
            if ((frame[j] >= 0) !== (frame[j - 1] >= 0)) zcr++;
        }
        zcr /= frame.length;

        // FFT-based spectral features
        const fft = simpleFFT(frame);
        const halfN = Math.floor(fft.length / 2);
        const magnitudes = new Float32Array(halfN);
        let totalMag = 0;
        for (let j = 0; j < halfN; j++) {
            magnitudes[j] = Math.sqrt(fft[j].re * fft[j].re + fft[j].im * fft[j].im);
            totalMag += magnitudes[j];
        }
        if (totalMag === 0) totalMag = 1;

        // Spectral centroid
        let centroid = 0;
        for (let j = 0; j < halfN; j++) centroid += magnitudes[j] * j;
        centroid /= totalMag;

        // Spectral rolloff (85%)
        let cumSum = 0;
        let rolloff = halfN - 1;
        const threshold = totalMag * 0.85;
        for (let j = 0; j < halfN; j++) {
            cumSum += magnitudes[j];
            if (cumSum >= threshold) { rolloff = j; break; }
        }

        features.push(
            rms,
            zcr,
            halfN > 0 ? centroid / halfN : 0,
            halfN > 0 ? rolloff / halfN : 0
        );
    }

    return features;
}

// ==========================================
// Audio Recording
// ==========================================

async function startRecording(durationMs = RECORDING_DURATION_MS) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    return new Promise((resolve, reject) => {
        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
        mediaRecorder.onerror = (e) => {
            stream.getTracks().forEach(t => t.stop());
            reject(new Error('録音中にエラーが発生しました。'));
        };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            try {
                const blob = new Blob(chunks, { type: 'audio/webm' });
                const arrayBuffer = await blob.arrayBuffer();
                const audioCtx = new AudioContext();
                const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                await audioCtx.close();
                resolve({ audioBuffer, blob });
            } catch (err) {
                reject(new Error('録音データのデコードに失敗しました。'));
            }
        };
        mediaRecorder.start();
        setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, durationMs);
    });
}

async function loadAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new AudioContext();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    await audioCtx.close();
    const blob = new Blob([arrayBuffer], { type: file.type });
    return { audioBuffer, blob };
}

// ==========================================
// Waveform Drawing
// ==========================================

function drawWaveformThumbnail(canvas, audioBuffer) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = ACCENT_LIGHT;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i < width; i++) {
        const start = i * step;
        let min = 1, max = -1;
        for (let j = 0; j < step && start + j < data.length; j++) {
            const val = data[start + j];
            if (val < min) min = val;
            if (val > max) max = val;
        }
        const yMin = ((1 - min) / 2) * height;
        const yMax = ((1 - max) / 2) * height;
        ctx.moveTo(i, yMin);
        ctx.lineTo(i, yMax);
    }
    ctx.stroke();
}

// ==========================================
// Normalization Helpers
// ==========================================

function computeNormStats(allFeatures) {
    const dim = allFeatures[0].length;
    const means = new Array(dim).fill(0);
    const stds = new Array(dim).fill(0);
    const n = allFeatures.length;

    for (const feat of allFeatures) {
        for (let i = 0; i < dim; i++) means[i] += feat[i];
    }
    for (let i = 0; i < dim; i++) means[i] /= n;

    for (const feat of allFeatures) {
        for (let i = 0; i < dim; i++) stds[i] += (feat[i] - means[i]) ** 2;
    }
    for (let i = 0; i < dim; i++) stds[i] = Math.sqrt(stds[i] / n) || 1;

    return { means, stds };
}

function normalizeFeatures(features, stats) {
    return features.map((v, i) => (v - stats.means[i]) / stats.stds[i]);
}

// ==========================================
// Main Render Function
// ==========================================

export function render(container, _data, _characteristics) {
    const state = {
        classes: [],
        model: null,
        normStats: null,
        trainHistory: null,
        currentStep: 0
    };

    container.innerHTML = buildInitialHTML();
    bindStep1Events(container, state);
}

function buildInitialHTML() {
    return `
        <h2><i class="fas fa-microphone" style="color: ${ACCENT};"></i> 音声分類</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            マイクまたはファイルから音声サンプルを収集し、TensorFlow.js で分類モデルを学習します。
        </p>
        ${createStepIndicator(STEPS, 0)}
        <div id="ac-step1" class="model-config" style="margin-top: 1.5rem;">
            <h3><i class="fas fa-database" style="color: ${ACCENT};"></i> Step 1: データ準備</h3>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 1rem;">
                各クラスごとに2つ以上の音声サンプルを録音またはアップロードしてください（3秒クリップ）。
            </p>
            <div style="margin-bottom: 1rem;">
                <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">クラスを追加:</label>
                <div style="display: flex; gap: 0.5rem;">
                    <input id="ac-class-name" type="text" placeholder="クラス名（例: 拍手、声）"
                        style="flex: 1; padding: 0.5rem 0.75rem; border: 1px solid ${ACCENT_BORDER};
                               border-radius: 8px; font-size: 0.95rem; outline: none;">
                    <button id="ac-add-class" style="padding: 0.5rem 1rem; background: ${ACCENT};
                        color: white; border: none; border-radius: 8px; cursor: pointer;
                        font-weight: 600; white-space: nowrap;">
                        <i class="fas fa-plus"></i> 追加
                    </button>
                </div>
            </div>
            <div id="ac-classes-container"></div>
            <button id="ac-btn-train" class="btn-analysis" style="background: ${ACCENT}; display: none;">
                <i class="fas fa-play"></i> 学習を開始（Step 2へ）
            </button>
        </div>
        <div id="ac-step2" style="display: none;"></div>
        <div id="ac-step3" style="display: none;"></div>
        <div id="ac-step4" style="display: none;"></div>
    `;
}

// ==========================================
// Step 1: Data Preparation
// ==========================================

function bindStep1Events(container, state) {
    const addBtn = container.querySelector('#ac-add-class');
    const nameInput = container.querySelector('#ac-class-name');
    const trainBtn = container.querySelector('#ac-btn-train');

    const addClass = () => {
        const name = nameInput.value.trim();
        if (!name) return;
        if (state.classes.some(c => c.name === name)) {
            alert('同じ名前のクラスが既に存在します。');
            return;
        }
        const newClass = { name, samples: [] };
        const updatedClasses = [...state.classes, newClass];
        state.classes = updatedClasses;
        nameInput.value = '';
        renderClasses(container, state);
        updateTrainButton(state, trainBtn);
    };

    addBtn.addEventListener('click', addClass);
    nameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') addClass();
    });

    trainBtn.addEventListener('click', () => runTraining(container, state));
}

function updateTrainButton(state, trainBtn) {
    const ready = state.classes.length >= 2 &&
        state.classes.every(c => c.samples.length >= 2);
    trainBtn.style.display = ready ? 'flex' : 'none';
}

function renderClasses(container, state) {
    const classesContainer = container.querySelector('#ac-classes-container');
    const trainBtn = container.querySelector('#ac-btn-train');

    classesContainer.innerHTML = state.classes.map((cls, ci) => `
        <div class="ac-class-block" data-class-index="${ci}"
             style="border: 1px solid ${ACCENT_BORDER}; border-radius: 12px;
                    padding: 1rem; margin-bottom: 1rem; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; color: ${ACCENT};">
                    <i class="fas fa-tag"></i> ${cls.name}
                    <span style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 400;">
                        (${cls.samples.length} サンプル)
                    </span>
                </h4>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="ac-record-btn" data-ci="${ci}"
                        style="padding: 0.4rem 0.75rem; background: #ef4444; color: white;
                               border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem;">
                        <i class="fas fa-microphone"></i> 録音
                    </button>
                    <label style="padding: 0.4rem 0.75rem; background: #3b82f6; color: white;
                                  border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
                                  display: inline-flex; align-items: center; gap: 0.3rem;">
                        <i class="fas fa-upload"></i> アップロード
                        <input type="file" accept="audio/*" class="ac-upload-input" data-ci="${ci}"
                               style="display: none;" multiple>
                    </label>
                    <button class="ac-remove-class" data-ci="${ci}"
                        style="padding: 0.4rem 0.5rem; background: #f1f5f9; color: #64748b;
                               border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem;">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="ac-recording-indicator" data-ci="${ci}"
                 style="display: none; text-align: center; padding: 0.5rem;
                        background: #fef2f2; border-radius: 8px; margin-bottom: 0.5rem;">
                <i class="fas fa-circle" style="color: #ef4444; animation: blink 1s infinite;"></i>
                <span style="margin-left: 0.5rem; color: #dc2626; font-weight: 600;">録音中... (3秒)</span>
            </div>
            <div class="ac-samples-list" data-ci="${ci}"
                 style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                ${cls.samples.map((s, si) => sampleCard(ci, si, s)).join('')}
            </div>
        </div>
    `).join('');

    // Add blink animation if not present
    if (!document.getElementById('ac-blink-style')) {
        const style = document.createElement('style');
        style.id = 'ac-blink-style';
        style.textContent = '@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }';
        document.head.appendChild(style);
    }

    // Bind record buttons
    classesContainer.querySelectorAll('.ac-record-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const ci = parseInt(btn.dataset.ci, 10);
            const indicator = classesContainer.querySelector(`.ac-recording-indicator[data-ci="${ci}"]`);
            btn.disabled = true;
            indicator.style.display = 'block';
            try {
                const { audioBuffer, blob } = await startRecording(RECORDING_DURATION_MS);
                const features = extractAudioFeatures(audioBuffer);
                const updatedSamples = [...state.classes[ci].samples, { audioBuffer, blob, features }];
                state.classes = state.classes.map((c, i) =>
                    i === ci ? { ...c, samples: updatedSamples } : c
                );
                renderClasses(container, state);
                updateTrainButton(state, trainBtn);
            } catch (err) {
                handleAudioError(err);
            } finally {
                indicator.style.display = 'none';
                btn.disabled = false;
            }
        });
    });

    // Bind upload inputs
    classesContainer.querySelectorAll('.ac-upload-input').forEach(input => {
        input.addEventListener('change', async (e) => {
            const ci = parseInt(input.dataset.ci, 10);
            const files = Array.from(e.target.files);
            for (const file of files) {
                try {
                    const { audioBuffer, blob } = await loadAudioFile(file);
                    const features = extractAudioFeatures(audioBuffer);
                    const updatedSamples = [...state.classes[ci].samples, { audioBuffer, blob, features }];
                    state.classes = state.classes.map((c, i) =>
                        i === ci ? { ...c, samples: updatedSamples } : c
                    );
                } catch (err) {
                    alert(`ファイル "${file.name}" の読み込みに失敗しました: ${err.message}`);
                }
            }
            renderClasses(container, state);
            updateTrainButton(state, trainBtn);
        });
    });

    // Bind remove class buttons
    classesContainer.querySelectorAll('.ac-remove-class').forEach(btn => {
        btn.addEventListener('click', () => {
            const ci = parseInt(btn.dataset.ci, 10);
            state.classes = state.classes.filter((_, i) => i !== ci);
            renderClasses(container, state);
            updateTrainButton(state, trainBtn);
        });
    });

    // Bind remove sample buttons
    classesContainer.querySelectorAll('.ac-remove-sample').forEach(btn => {
        btn.addEventListener('click', () => {
            const ci = parseInt(btn.dataset.ci, 10);
            const si = parseInt(btn.dataset.si, 10);
            const updatedSamples = state.classes[ci].samples.filter((_, i) => i !== si);
            state.classes = state.classes.map((c, i) =>
                i === ci ? { ...c, samples: updatedSamples } : c
            );
            renderClasses(container, state);
            updateTrainButton(state, trainBtn);
        });
    });

    // Draw waveforms
    classesContainer.querySelectorAll('.ac-waveform-canvas').forEach(canvas => {
        const ci = parseInt(canvas.dataset.ci, 10);
        const si = parseInt(canvas.dataset.si, 10);
        const sample = state.classes[ci]?.samples[si];
        if (sample) drawWaveformThumbnail(canvas, sample.audioBuffer);
    });

    // Bind audio play buttons
    classesContainer.querySelectorAll('.ac-play-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const ci = parseInt(btn.dataset.ci, 10);
            const si = parseInt(btn.dataset.si, 10);
            const sample = state.classes[ci]?.samples[si];
            if (sample) {
                const url = URL.createObjectURL(sample.blob);
                const audio = new Audio(url);
                audio.onended = () => URL.revokeObjectURL(url);
                audio.play().catch(() => {});
            }
        });
    });
}

function sampleCard(ci, si, _sample) {
    return `
        <div style="display: flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.5rem;
                    background: ${ACCENT_LIGHT}; border-radius: 8px; font-size: 0.8rem;">
            <canvas class="ac-waveform-canvas" data-ci="${ci}" data-si="${si}"
                    width="80" height="30"
                    style="border-radius: 4px; cursor: pointer;"></canvas>
            <button class="ac-play-btn" data-ci="${ci}" data-si="${si}"
                    style="background: none; border: none; color: ${ACCENT}; cursor: pointer; font-size: 1rem;">
                <i class="fas fa-play-circle"></i>
            </button>
            <button class="ac-remove-sample" data-ci="${ci}" data-si="${si}"
                    style="background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 0.85rem;">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
}

function handleAudioError(err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        alert('マイクの使用が許可されていません。ブラウザの設定からマイクへのアクセスを許可してください。');
    } else if (err.name === 'NotFoundError') {
        alert('マイクが見つかりません。マイクが接続されているか確認してください。');
    } else if (err.name === 'NotReadableError') {
        alert('マイクにアクセスできません。他のアプリケーションが使用中の可能性があります。');
    } else {
        alert(`音声エラー: ${err.message}`);
    }
}

// ==========================================
// Step 2: Training
// ==========================================

async function runTraining(container, state) {
    container.querySelector('#ac-step1').style.display = 'none';
    const step2 = container.querySelector('#ac-step2');
    step2.style.display = 'block';

    // Update step indicator
    const stepIndicator = container.querySelector('.step-indicator');
    stepIndicator.outerHTML = createStepIndicator(STEPS, 1);

    step2.innerHTML = `
        <div class="model-config">
            <h3><i class="fas fa-brain" style="color: ${ACCENT};"></i> Step 2: 特徴抽出と学習</h3>
            <div id="ac-train-progress" style="text-align: center; padding: 2rem;">
                <i class="fas fa-spinner fa-spin fa-2x" style="color: ${ACCENT};"></i>
                <p style="margin-top: 1rem; font-weight: 600;">TensorFlow.js を読み込み中...</p>
            </div>
        </div>
    `;

    try {
        await ensureTensorFlow();
    } catch (err) {
        step2.querySelector('#ac-train-progress').innerHTML = `
            <p style="color: #ef4444; font-weight: 600;">
                <i class="fas fa-exclamation-triangle"></i> ${err.message}
            </p>
            <p style="color: var(--text-secondary); margin-top: 0.5rem;">
                インターネット接続を確認してページを再読み込みしてください。
            </p>
        `;
        return;
    }

    const progressEl = step2.querySelector('#ac-train-progress');
    progressEl.innerHTML = `
        <i class="fas fa-cogs fa-2x" style="color: ${ACCENT};"></i>
        <p style="margin-top: 1rem; font-weight: 600;">特徴量を抽出しています...</p>
    `;

    await pause(100);

    // Prepare training data
    const allFeatures = [];
    const allLabels = [];
    const classNames = state.classes.map(c => c.name);
    const numClasses = classNames.length;

    for (let ci = 0; ci < state.classes.length; ci++) {
        for (const sample of state.classes[ci].samples) {
            allFeatures.push(sample.features);
            allLabels.push(ci);
        }
    }

    // Normalize features
    const normStats = computeNormStats(allFeatures);
    state.normStats = normStats;
    const normalizedFeatures = allFeatures.map(f => normalizeFeatures(f, normStats));

    progressEl.innerHTML = `
        <i class="fas fa-brain fa-2x" style="color: ${ACCENT};"></i>
        <p style="margin-top: 1rem; font-weight: 600;">モデルを学習中...</p>
        <div id="ac-epoch-info" style="margin-top: 0.75rem; color: var(--text-secondary);"></div>
        <div style="width: 80%; margin: 1rem auto; background: #f1f5f9; border-radius: 8px; height: 8px;">
            <div id="ac-progress-bar" style="width: 0%; height: 100%; background: ${ACCENT};
                 border-radius: 8px; transition: width 0.3s;"></div>
        </div>
    `;

    await pause(100);

    // Build model
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [FEATURE_DIM], units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    const xs = tf.tensor2d(normalizedFeatures);
    const ys = tf.tensor1d(allLabels, 'int32');

    const totalEpochs = 100;
    const epochInfo = progressEl.querySelector('#ac-epoch-info');
    const progressBar = progressEl.querySelector('#ac-progress-bar');
    const historyLog = { loss: [], acc: [] };

    await model.fit(xs, ys, {
        epochs: totalEpochs,
        batchSize: Math.min(32, normalizedFeatures.length),
        shuffle: true,
        validationSplit: normalizedFeatures.length >= 10 ? 0.2 : 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                historyLog.loss.push(logs.loss);
                historyLog.acc.push(logs.acc);
                const pct = ((epoch + 1) / totalEpochs * 100).toFixed(0);
                progressBar.style.width = `${pct}%`;
                epochInfo.textContent =
                    `Epoch ${epoch + 1}/${totalEpochs} — Loss: ${formatNumber(logs.loss)} — Accuracy: ${formatNumber(logs.acc)}`;
            }
        }
    });

    xs.dispose();
    ys.dispose();

    state.model = model;
    state.trainHistory = historyLog;
    state.currentStep = 2;

    // Compute per-class predictions for evaluation
    const predictions = [];
    for (let ci = 0; ci < state.classes.length; ci++) {
        for (const sample of state.classes[ci].samples) {
            const norm = normalizeFeatures(sample.features, normStats);
            const pred = model.predict(tf.tensor2d([norm]));
            const predArr = Array.from(pred.dataSync());
            pred.dispose();
            const predClass = predArr.indexOf(Math.max(...predArr));
            predictions.push({ trueClass: ci, predClass, probs: predArr });
        }
    }

    state.predictions = predictions;
    state.classNames = classNames;

    showStep3(container, state);
}

// ==========================================
// Step 3: Evaluation
// ==========================================

function showStep3(container, state) {
    const stepIndicator = container.querySelector('.step-indicator');
    stepIndicator.outerHTML = createStepIndicator(STEPS, 2);

    container.querySelector('#ac-step2').style.display = 'none';
    const step3 = container.querySelector('#ac-step3');
    step3.style.display = 'block';

    const { trainHistory, predictions, classNames } = state;
    const numClasses = classNames.length;

    // Overall accuracy
    const correct = predictions.filter(p => p.trueClass === p.predClass).length;
    const overallAcc = correct / predictions.length;

    // Per-class accuracy
    const perClass = classNames.map((name, ci) => {
        const classP = predictions.filter(p => p.trueClass === ci);
        const classCorrect = classP.filter(p => p.predClass === ci).length;
        return {
            name,
            accuracy: classP.length > 0 ? classCorrect / classP.length : 0,
            total: classP.length,
            correct: classCorrect
        };
    });

    // Confusion matrix
    const confMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
    for (const p of predictions) {
        confMatrix[p.trueClass][p.predClass]++;
    }

    step3.innerHTML = `
        <div class="model-config">
            <h3><i class="fas fa-chart-bar" style="color: ${ACCENT};"></i> Step 3: 評価</h3>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 1rem; margin: 1.5rem 0;">
                <div class="metric-card">
                    <div class="metric-label">全体正解率</div>
                    <div class="metric-value" style="color: ${ACCENT};">${(overallAcc * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">クラス数</div>
                    <div class="metric-value" style="color: ${ACCENT};">${numClasses}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">サンプル数</div>
                    <div class="metric-value" style="color: ${ACCENT};">${predictions.length}</div>
                </div>
            </div>

            <h4 style="margin: 1.5rem 0 0.75rem;"><i class="fas fa-chart-line"></i> 学習曲線</h4>
            <div id="ac-history-plot" style="width: 100%; height: 350px;"></div>

            <h4 style="margin: 1.5rem 0 0.75rem;"><i class="fas fa-bullseye"></i> クラス別正解率</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
                ${perClass.map(pc => `
                    <div style="background: ${ACCENT_LIGHT}; border-radius: 10px; padding: 0.75rem 1rem;
                                border-left: 4px solid ${ACCENT};">
                        <div style="font-weight: 600; color: ${ACCENT};">${pc.name}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.25rem;">
                            ${(pc.accuracy * 100).toFixed(1)}%
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            ${pc.correct} / ${pc.total} 正解
                        </div>
                    </div>
                `).join('')}
            </div>

            <h4 style="margin: 1.5rem 0 0.75rem;"><i class="fas fa-th"></i> 混同行列</h4>
            <div id="ac-confusion-matrix" style="width: 100%; max-width: 500px;"></div>

            <button id="ac-btn-predict" class="btn-analysis" style="background: ${ACCENT}; margin-top: 1.5rem;">
                <i class="fas fa-arrow-right"></i> 予測へ進む（Step 4）
            </button>
        </div>
    `;

    // Plot training history
    const epochs = trainHistory.loss.map((_, i) => i + 1);
    renderPlot('ac-history-plot', [
        {
            x: epochs,
            y: trainHistory.loss,
            mode: 'lines',
            name: '損失 (Loss)',
            line: { color: '#ef4444', width: 2 }
        },
        {
            x: epochs,
            y: trainHistory.acc,
            mode: 'lines',
            name: '正解率 (Accuracy)',
            yaxis: 'y2',
            line: { color: ACCENT, width: 2 }
        }
    ], {
        title: '学習履歴',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss', side: 'left' },
        yaxis2: { title: 'Accuracy', side: 'right', overlaying: 'y', range: [0, 1] },
        legend: { x: 0.5, y: -0.2, orientation: 'h', xanchor: 'center' },
        height: 350
    });

    // Confusion matrix
    renderPlot('ac-confusion-matrix', [{
        z: confMatrix,
        x: classNames,
        y: classNames,
        type: 'heatmap',
        colorscale: [[0, '#ede9fe'], [1, '#7c3aed']],
        showscale: true,
        text: confMatrix.map(row => row.map(v => v.toString())),
        texttemplate: '%{text}',
        textfont: { size: 14 },
        hoverongaps: false
    }], {
        title: '混同行列',
        xaxis: { title: '予測値', side: 'bottom' },
        yaxis: { title: '実測値', autorange: 'reversed' },
        height: 400,
        width: Math.max(350, classNames.length * 80 + 150)
    });

    // Bind predict button
    container.querySelector('#ac-btn-predict').addEventListener('click', () => {
        showStep4(container, state);
    });
}

// ==========================================
// Step 4: Prediction
// ==========================================

function showStep4(container, state) {
    const stepIndicator = container.querySelector('.step-indicator');
    stepIndicator.outerHTML = createStepIndicator(STEPS, 3);

    container.querySelector('#ac-step3').style.display = 'none';
    const step4 = container.querySelector('#ac-step4');
    step4.style.display = 'block';

    step4.innerHTML = `
        <div class="model-config">
            <h3><i class="fas fa-magic" style="color: ${ACCENT};"></i> Step 4: 予測</h3>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 1.5rem;">
                新しい音声を録音またはアップロードして分類結果を確認しましょう。
            </p>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
                <button id="ac-pred-record" style="padding: 0.75rem 1.5rem; background: #ef4444;
                    color: white; border: none; border-radius: 8px; cursor: pointer;
                    font-weight: 600; font-size: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-microphone"></i> 録音して予測
                </button>
                <label style="padding: 0.75rem 1.5rem; background: #3b82f6; color: white;
                              border: none; border-radius: 8px; cursor: pointer;
                              font-weight: 600; font-size: 1rem; display: inline-flex;
                              align-items: center; gap: 0.5rem;">
                    <i class="fas fa-upload"></i> ファイルで予測
                    <input type="file" accept="audio/*" id="ac-pred-upload" style="display: none;">
                </label>
            </div>
            <div id="ac-pred-recording-indicator"
                 style="display: none; text-align: center; padding: 0.75rem;
                        background: #fef2f2; border-radius: 8px; margin-bottom: 1rem;">
                <i class="fas fa-circle" style="color: #ef4444; animation: blink 1s infinite;"></i>
                <span style="margin-left: 0.5rem; color: #dc2626; font-weight: 600;">録音中... (3秒)</span>
            </div>
            <div id="ac-pred-result"></div>
        </div>
    `;

    const recordBtn = step4.querySelector('#ac-pred-record');
    const uploadInput = step4.querySelector('#ac-pred-upload');
    const indicator = step4.querySelector('#ac-pred-recording-indicator');

    recordBtn.addEventListener('click', async () => {
        recordBtn.disabled = true;
        indicator.style.display = 'block';
        try {
            const { audioBuffer, blob } = await startRecording(RECORDING_DURATION_MS);
            indicator.style.display = 'none';
            await showPrediction(container, state, audioBuffer, blob);
        } catch (err) {
            indicator.style.display = 'none';
            handleAudioError(err);
        } finally {
            recordBtn.disabled = false;
        }
    });

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const { audioBuffer, blob } = await loadAudioFile(file);
            await showPrediction(container, state, audioBuffer, blob);
        } catch (err) {
            alert(`ファイルの読み込みに失敗しました: ${err.message}`);
        }
    });
}

async function showPrediction(container, state, audioBuffer, blob) {
    const resultDiv = container.querySelector('#ac-pred-result');
    const features = extractAudioFeatures(audioBuffer);
    const norm = normalizeFeatures(features, state.normStats);
    const predTensor = state.model.predict(tf.tensor2d([norm]));
    const probs = Array.from(predTensor.dataSync());
    predTensor.dispose();

    const maxIdx = probs.indexOf(Math.max(...probs));
    const predictedClass = state.classNames[maxIdx];
    const confidence = probs[maxIdx];

    const plotId = 'ac-pred-prob-chart-' + Date.now();
    const waveformId = 'ac-pred-waveform-' + Date.now();

    resultDiv.innerHTML = `
        <div style="border: 2px solid ${ACCENT}; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">予測結果</div>
                <div style="font-size: 2rem; font-weight: 700; color: ${ACCENT};">
                    ${predictedClass}
                </div>
                <div style="font-size: 1rem; color: var(--text-secondary); margin-top: 0.25rem;">
                    信頼度: ${(confidence * 100).toFixed(1)}%
                </div>
            </div>

            <h4 style="margin: 1rem 0 0.5rem;"><i class="fas fa-wave-square"></i> 波形</h4>
            <canvas id="${waveformId}" width="600" height="100"
                    style="width: 100%; height: 100px; border-radius: 8px; background: ${ACCENT_LIGHT};"></canvas>

            <h4 style="margin: 1.5rem 0 0.5rem;"><i class="fas fa-chart-bar"></i> クラス別確率</h4>
            <div id="${plotId}" style="width: 100%; height: 280px;"></div>
        </div>
    `;

    // Draw waveform
    const canvas = resultDiv.querySelector(`#${waveformId}`);
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 200;
    drawWaveformThumbnail(canvas, audioBuffer);

    // Probability bar chart
    const sortedIndices = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
    renderPlot(plotId, [{
        type: 'bar',
        x: sortedIndices.map(i => probs[i]),
        y: sortedIndices.map(i => state.classNames[i]),
        orientation: 'h',
        marker: {
            color: sortedIndices.map(i => i === maxIdx ? ACCENT : '#c4b5fd')
        },
        text: sortedIndices.map(i => `${(probs[i] * 100).toFixed(1)}%`),
        textposition: 'outside',
        hovertemplate: '%{y}: %{x:.1%}<extra></extra>'
    }], {
        title: 'クラス別予測確率',
        xaxis: { title: '確率', range: [0, 1.15], tickformat: '.0%' },
        yaxis: { automargin: true },
        height: 280,
        margin: { l: 120, r: 60, t: 40, b: 50 }
    });
}

// ==========================================
// Utility
// ==========================================

function pause(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
