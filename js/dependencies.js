// Third-party libraries are loaded only when a workflow needs them.
const LIBRARIES = {
    plotly: {
        src: 'https://cdn.plot.ly/plotly-2.35.2.min.js',
        integrity: 'sha384-cCVCZkAjYNxaYKbM8lsArLznDF/SvMFr1jcZrvOpSTCa0W40ZAdLzHCEulnUa5i7',
        globalName: 'Plotly'
    },
    xlsx: {
        src: 'https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js',
        integrity: 'sha384-EnyY0/GSHQGSxSgMwaIPzSESbqoOLSexfnSMN2AP+39Ckmn92stwABZynq1JyzdT',
        globalName: 'XLSX'
    },
    math: {
        src: 'https://cdnjs.cloudflare.com/ajax/libs/mathjs/11.7.0/math.min.js',
        integrity: 'sha384-SPOwnyI13fBtl6XAAt0A5mYOQP2tz7Kgd00jQAghR2Ck6IWEVfGTaoHU7yfPB6/c',
        globalName: 'math'
    },
    tf: {
        src: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js',
        integrity: 'sha384-R4iglwC8w7UAyfRq7VUmXEPjrvYnoXIqhsbTUN07o6o8yKYeiTY0/Z6DzbsKgWZ3',
        globalName: 'tf'
    },
    mobilenet: {
        src: 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0/dist/mobilenet.min.js',
        integrity: 'sha384-7ZhlpAW9lptV643xV0jWb+SkDEsvvzCa2kT+UhVnD/RFvNyOJZRNHrs9UBDkzyuW',
        globalName: 'mobilenet'
    },
    poseDetection: {
        src: 'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js',
        integrity: 'sha384-pjg3ZYfXdbqsHl42YTyNHeHMzvrwQUvEuQ7tTn6+tDmoUkSD3ItK1sBa73y1l0H3',
        globalName: 'poseDetection'
    }
};

const pendingLoads = new Map();

function loadLibrary(name) {
    const definition = LIBRARIES[name];
    if (!definition) return Promise.reject(new Error(`Unknown library: ${name}`));
    if (globalThis[definition.globalName]) return Promise.resolve(globalThis[definition.globalName]);
    if (pendingLoads.has(name)) return pendingLoads.get(name);

    const promise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = definition.src;
        script.integrity = definition.integrity;
        script.crossOrigin = 'anonymous';
        script.dataset.dependency = name;
        script.onload = () => resolve(globalThis[definition.globalName]);
        script.onerror = () => {
            pendingLoads.delete(name);
            script.remove();
            reject(new Error(`${name} の安全な読み込みに失敗しました。通信状態を確認して再試行してください。`));
        };
        document.head.appendChild(script);
    });
    pendingLoads.set(name, promise);
    return promise;
}

export function ensureSpreadsheetLibrary() {
    return loadLibrary('xlsx');
}

export async function ensureTensorFlow() {
    return loadLibrary('tf');
}

export async function ensureAnalysisDependencies(analysisType) {
    const needsPlotly = new Set([
        'eda', 'preprocessing', 'regression', 'classification', 'learning_guide',
        'image_classification', 'audio_classification', 'video_analysis'
    ]);
    if (needsPlotly.has(analysisType)) await loadLibrary('plotly');

    if (analysisType === 'regression') await loadLibrary('math');
    if (analysisType === 'image_classification') {
        await loadLibrary('tf');
        await loadLibrary('mobilenet');
    }
    if (analysisType === 'audio_classification') await loadLibrary('tf');
    if (analysisType === 'video_analysis') {
        await loadLibrary('tf');
        await loadLibrary('poseDetection');
    }
}
