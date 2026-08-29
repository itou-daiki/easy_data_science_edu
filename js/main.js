// ==========================================
// easyDataScience - Main Entry Point
// ==========================================
import { showError, showLoadingMessage, hideLoadingMessage, toggleCollapsible, renderDataPreview, renderSummaryStatistics, escapeHtml } from './utils.js';
import { setupAIAssistSettingsUI, clearAIAssistPanelContext } from './ai_assistant.js';
import { initializeI18n } from './i18n.js';
import { ensureSpreadsheetLibrary, ensureAnalysisDependencies } from './dependencies.js';

// ==========================================
// Global Variables
// ==========================================
export let currentData = null;
export let dataCharacteristics = null;
let activeAnalysisModule = null;
let lastAnalysisTrigger = null;

// ==========================================
// DOM Elements
// ==========================================
const loadingScreen = document.getElementById('loading-screen');
const mainApp = document.getElementById('main-app');
const uploadArea = document.getElementById('main-upload-area');
const uploadBtn = document.getElementById('main-upload-btn');
const fileInput = document.getElementById('main-data-file');
const fileInfo = document.getElementById('main-file-info');
const demoBtn = document.getElementById('load-demo-btn');
const featureGrid = document.querySelector('.feature-grid');
const aiSettingsBtn = document.getElementById('ai-settings-btn');
const aiSettingsModal = document.getElementById('ai-settings-modal');
const closeAISettingsModal = document.getElementById('close-ai-settings-modal');
const geminiApiKeyInput = document.getElementById('gemini-api-key-input');
const geminiModelInput = document.getElementById('gemini-model-input');
const geminiIncludePreviewInput = document.getElementById('gemini-include-preview-input');
const saveAISettingsBtn = document.getElementById('save-ai-settings-btn');
const clearAISettingsBtn = document.getElementById('clear-ai-settings-btn');
const aiSettingsStatus = document.getElementById('ai-settings-status');
const aiStatusBadge = document.getElementById('ai-status-badge');

// ==========================================
// Initialization
// ==========================================
initializeI18n();

document.addEventListener('DOMContentLoaded', () => {
    loadingScreen.style.display = 'none';
    mainApp.style.display = 'block';
    setupEventListeners();
});

// ==========================================
// Event Listeners
// ==========================================
function setupEventListeners() {
    setupAIAssistSettingsUI({
        button: aiSettingsBtn,
        modal: aiSettingsModal,
        closeButton: closeAISettingsModal,
        apiKeyInput: geminiApiKeyInput,
        modelInput: geminiModelInput,
        includePreviewInput: geminiIncludePreviewInput,
        status: aiSettingsStatus,
        saveButton: saveAISettingsBtn,
        clearButton: clearAISettingsBtn,
        badge: aiStatusBadge
    });

    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) handleFile(file);
    });
    uploadArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
    uploadArea.addEventListener('drop', (event) => {
        event.preventDefault();
        uploadArea.classList.remove('drag-over');
        const file = event.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    const demoModal = document.getElementById('demo-modal');
    const closeDemoModal = document.getElementById('close-demo-modal');
    let lastModalTrigger = null;

    const openModal = (modal, trigger) => {
        lastModalTrigger = trigger || document.activeElement;
        modal.style.display = 'block';
        modal.setAttribute('aria-hidden', 'false');
        const first = modal.querySelector('button, input, select, textarea, [href], [tabindex]:not([tabindex="-1"])');
        if (first) first.focus();
    };
    const hideModal = (modal) => {
        modal.style.display = 'none';
        modal.setAttribute('aria-hidden', 'true');
        if (lastModalTrigger?.focus) lastModalTrigger.focus();
    };
    const trapModalFocus = (modal, event) => {
        if (event.key === 'Escape') {
            event.preventDefault();
            hideModal(modal);
            return;
        }
        if (event.key !== 'Tab') return;
        const focusable = [...modal.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [href], [tabindex]:not([tabindex="-1"])')]
            .filter(element => element.offsetParent !== null);
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    };

    demoBtn.addEventListener('click', () => openModal(demoModal, demoBtn));

    closeDemoModal.addEventListener('click', () => {
        hideModal(demoModal);
    });

    window.addEventListener('click', (event) => {
        if (event.target === demoModal) {
            hideModal(demoModal);
        }
    });

    document.querySelectorAll('.demo-option-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const fileName = btn.dataset.demo;
            hideModal(demoModal);
            loadDemoData(fileName);
        });
    });

    document.querySelectorAll('.collapsible-header').forEach(configureCollapsibleHeader);

    [demoModal, aiSettingsModal].forEach(modal => {
        modal.addEventListener('keydown', event => trapModalFocus(modal, event));
    });
    aiSettingsBtn.addEventListener('click', () => {
        lastModalTrigger = aiSettingsBtn;
        aiSettingsModal.setAttribute('aria-hidden', 'false');
    });

    featureGrid.addEventListener('click', (event) => {
        const card = event.target.closest('.feature-card');
        if (!card) return;

        const requires = card.dataset.requires;
        if (requires === 'none') {
            if (!card.onclick) {
                showAnalysisView(card.dataset.analysis);
            }
            return;
        }

        if (!currentData) {
            showError('分析を開始するには、データをアップロードするかデモデータを試してください。');
            return;
        }
    });

    featureGrid.querySelectorAll('.feature-card').forEach(card => {
        card.setAttribute('role', 'button');
        card.tabIndex = 0;
        card.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                card.click();
            }
        });
    });
}

function configureCollapsibleHeader(header) {
    if (header.dataset.a11yReady === 'true') return;
    const content = header.nextElementSibling;
    if (!content) return;
    header.dataset.a11yReady = 'true';
    header.setAttribute('role', 'button');
    header.tabIndex = 0;
    if (!content.id) content.id = `collapsible-${Math.random().toString(36).slice(2)}`;
    header.setAttribute('aria-controls', content.id);
    const expanded = !header.classList.contains('collapsed');
    header.setAttribute('aria-expanded', String(expanded));
    content.hidden = !expanded;
    header.addEventListener('click', () => toggleCollapsible(header));
    header.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            toggleCollapsible(header);
        }
    });
}

// ==========================================
// File Handling & Data Processing
// ==========================================
const MAX_DATA_FILE_SIZE = 25 * 1024 * 1024;
const MAX_DATA_ROWS = 50000;
const MAX_DATA_COLUMNS = 500;
const MAX_DATA_CELLS = 2000000;

function validateWorksheetShape(worksheet) {
    if (!worksheet?.['!ref']) throw new Error('シートに読み取れるデータがありません。');
    const range = XLSX.utils.decode_range(worksheet['!ref']);
    const rowCount = range.e.r - range.s.r + 1;
    const columnCount = range.e.c - range.s.c + 1;
    if (rowCount > MAX_DATA_ROWS) throw new Error(`行数が上限（${MAX_DATA_ROWS.toLocaleString()}行）を超えています。`);
    if (columnCount > MAX_DATA_COLUMNS) throw new Error(`列数が上限（${MAX_DATA_COLUMNS}列）を超えています。`);
    if (rowCount * columnCount > MAX_DATA_CELLS) throw new Error('データのセル数が上限（2,000,000セル）を超えています。');
}

async function handleFile(file) {
    const lowerName = file.name.toLowerCase();
    const isCsv = lowerName.endsWith('.csv');
    const isSpreadsheet = lowerName.endsWith('.xlsx') || lowerName.endsWith('.xls');
    if (!isCsv && !isSpreadsheet) {
        showError('CSVまたはExcelファイルを選択してください。');
        return;
    }
    if (file.size > MAX_DATA_FILE_SIZE) {
        showError('ファイルが大きすぎます（上限25 MiB）。');
        return;
    }

    try {
        await ensureSpreadsheetLibrary();
    } catch (error) {
        console.error(error);
        showError(error.message);
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = e.target.result;
            let jsonData;

            if (isCsv) {
                const workbook = XLSX.read(data, { type: 'string', raw: true, sheetRows: MAX_DATA_ROWS + 1 });
                const worksheet = workbook.Sheets[workbook.SheetNames[0]];
                validateWorksheetShape(worksheet);
                jsonData = XLSX.utils.sheet_to_json(worksheet);
            } else {
                const workbook = XLSX.read(data, { type: 'array', sheetRows: MAX_DATA_ROWS + 1 });
                const worksheet = workbook.Sheets[workbook.SheetNames[0]];
                validateWorksheetShape(worksheet);
                jsonData = XLSX.utils.sheet_to_json(worksheet);
            }

            if (jsonData.length === 0) {
                showError('ファイルにデータが含まれていません。');
                return;
            }
            processData(file.name, jsonData);
        } catch (error) {
            console.error(error);
            showError(error.message || 'ファイルの読み込みに失敗しました。');
        }
    };
    reader.onerror = () => showError('ファイルを読み取れませんでした。');
    if (isCsv) {
        reader.readAsText(file);
    } else {
        reader.readAsArrayBuffer(file);
    }
}

async function loadDemoData(fileName) {
    showLoadingMessage(`デモデータ (${fileName}) を読み込み中...`);
    try {
        await ensureSpreadsheetLibrary();
        const response = await fetch(`./datasets/${fileName}`);
        if (!response.ok) throw new Error(`Server responded with ${response.status}`);

        const text = await response.text();
        const workbook = XLSX.read(text, { type: 'string', raw: true, sheetRows: MAX_DATA_ROWS + 1 });
        const worksheet = workbook.Sheets[workbook.SheetNames[0]];
        validateWorksheetShape(worksheet);
        const jsonData = XLSX.utils.sheet_to_json(worksheet);

        processData(fileName, jsonData);
    } catch (error) {
        console.error(error);
        showError(`デモデータ (${fileName}) の読み込みに失敗しました。`);
    }
}

function processData(fileName, jsonData) {
    clearAIAssistPanelContext();
    currentData = jsonData;

    const characteristics = analyzeDataCharacteristics(jsonData);
    // Store filename (without extension) for CSV export naming
    characteristics.fileName = fileName.replace(/\.[^.]+$/, '');
    window.dataCharacteristics = characteristics;
    dataCharacteristics = characteristics;

    updateFileInfo(fileName, jsonData);

    renderDataPreview('dataframe-container', currentData, 'データプレビュー');
    renderSummaryStatistics('summary-stats-container', currentData, characteristics, '要約統計量');

    const dataPreviewSection = document.getElementById('data-preview-section');
    dataPreviewSection.style.display = 'block';

    dataPreviewSection.querySelectorAll('.collapsible-header').forEach(header => {
        const newHeader = header.cloneNode(true);
        header.parentNode.replaceChild(newHeader, header);
        configureCollapsibleHeader(newHeader);
    });

    updateFeatureCards();
    hideLoadingMessage();
}

function analyzeDataCharacteristics(data) {
    if (!data || data.length === 0) return null;
    const characteristics = { numericColumns: [], categoricalColumns: [], textColumns: [], allColumns: [] };
    const columns = Object.keys(data[0]);
    characteristics.allColumns = columns;

    columns.forEach(col => {
        const values = data.map(row => row[col]).filter(val => val != null);
        if (values.length === 0) return;

        const isNumeric = values.every(val =>
            (typeof val === 'number' && Number.isFinite(val))
            || (typeof val === 'string' && val.trim() !== '' && Number.isFinite(Number(val)))
        );

        if (isNumeric) {
            characteristics.numericColumns.push(col);
            data.forEach(row => {
                if (row[col] != null) row[col] = Number(row[col]);
            });

            const numericValues = data.map(row => row[col]).filter(val => val != null);
            const uniqueValues = new Set(numericValues);

            if (uniqueValues.size <= 10) {
                characteristics.categoricalColumns.push(col);
            }
        } else {
            const uniqueValues = new Set(values);
            if (uniqueValues.size <= 20 || (uniqueValues.size / values.length < 0.5 && values.length > 5)) {
                characteristics.categoricalColumns.push(col);
            } else {
                characteristics.textColumns.push(col);
            }
        }
    });
    return characteristics;
}
window.analyzeDataCharacteristics = analyzeDataCharacteristics;

// ==========================================
// UI Updates & View Management
// ==========================================
function updateFileInfo(fileName, data) {
    const nRows = data.length;
    const nCols = Object.keys(data[0] || {}).length;

    fileInfo.innerHTML = `
        <h3 style="margin: 0 0 1rem 0; font-size: 1.25rem; display: flex; align-items: center; gap: 0.5rem; color: #1e293b;">
            <i class="fas fa-info-circle" style="color: #1e90ff;"></i> データ情報
        </h3>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem;">
            <div style="flex: 2; min-width: 200px; background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e90ff;">
                <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.25rem;">
                    <i class="fas fa-file-excel" style="margin-right: 0.5rem; color: #1e90ff;"></i>ファイル名
                </div>
                <div style="font-weight: bold; color: #1e293b; font-size: 1.1rem; word-break: break-all;">
                    ${escapeHtml(fileName)}
                </div>
            </div>
            <div style="flex: 1; min-width: 120px; background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e90ff;">
                <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.25rem;">
                    <i class="fas fa-list-ol" style="margin-right: 0.5rem; color: #1e90ff;"></i>行数
                </div>
                <div style="font-weight: bold; color: #1e293b; font-size: 1.5rem;">
                    ${nRows.toLocaleString()}
                </div>
            </div>
            <div style="flex: 1; min-width: 120px; background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e90ff;">
                <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.25rem;">
                    <i class="fas fa-columns" style="margin-right: 0.5rem; color: #1e90ff;"></i>列数
                </div>
                <div style="font-weight: bold; color: #1e293b; font-size: 1.5rem;">
                    ${nCols.toLocaleString()}
                </div>
            </div>
        </div>
    `;
    fileInfo.style.display = 'block';
}

function updateFeatureCards() {
    if (!dataCharacteristics) return;
    const counts = {
        numeric: dataCharacteristics.numericColumns.length,
        categorical: dataCharacteristics.categoricalColumns.length,
        text: dataCharacteristics.textColumns.length
    };

    featureGrid.querySelectorAll('.feature-card').forEach(card => {
        const req = card.dataset.requires;

        if (req === 'none') {
            enableCard(card);
            return;
        }

        if (!req) {
            enableCard(card);
            return;
        }

        if (req === 'regression') {
            const ready = counts.numeric >= 1 && dataCharacteristics.allColumns.length >= 2;
            ready ? enableCard(card) : disableCard(card);
            return;
        }

        if (req === 'classification') {
            const hasTargetCandidate = dataCharacteristics.allColumns.some(column => {
                const values = currentData.map(row => row[column]).filter(value => value != null && value !== '');
                const uniqueCount = new Set(values).size;
                return uniqueCount >= 2 && uniqueCount <= 20;
            });
            const ready = hasTargetCandidate && dataCharacteristics.allColumns.length >= 2;
            ready ? enableCard(card) : disableCard(card);
            return;
        }

        const meetsRequirements = req.split(',').every(r => {
            const [type, count] = r.split(':');
            return counts[type] >= parseInt(count, 10);
        });
        meetsRequirements ? enableCard(card) : disableCard(card);
    });
}
window.updateFeatureCards = updateFeatureCards;

function enableCard(card) {
    card.classList.remove('disabled');
    const requirementText = card.querySelector('.feature-card-requirement');
    if (requirementText) requirementText.style.display = 'none';
    card.onclick = () => showAnalysisView(card.dataset.analysis);
    card.setAttribute('aria-disabled', 'false');
}

function disableCard(card) {
    card.classList.add('disabled');
    const requirementText = card.querySelector('.feature-card-requirement');
    if (requirementText) requirementText.style.display = 'block';
    card.onclick = null;
    card.setAttribute('aria-disabled', 'true');
}

async function showAnalysisView(analysisType) {
    const focusedCard = document.activeElement?.closest?.('.feature-card');
    lastAnalysisTrigger = focusedCard || featureGrid.querySelector(`.feature-card[data-analysis="${CSS.escape(analysisType)}"]`);
    clearAIAssistPanelContext();
    document.getElementById('navigation-section').style.display = 'none';
    document.getElementById('upload-section-main').style.display = 'none';

    const analysisHeader = document.getElementById('analysis-header');
    const analysisArea = document.getElementById('analysis-area');
    const analysisContent = document.getElementById('analysis-content');

    analysisContent.innerHTML = `<div class="loading"><i class="fas fa-spinner fa-spin"></i> 分析モジュールを読み込み中...</div>`;

    analysisHeader.style.display = 'flex';
    analysisArea.style.display = 'block';

    try {
        if (activeAnalysisModule?.dispose) activeAnalysisModule.dispose();
        activeAnalysisModule = null;
        await ensureAnalysisDependencies(analysisType);
        const modulePath = `./analyses/${analysisType}.js`;
        const analysisModule = await import(modulePath);
        activeAnalysisModule = analysisModule;
        analysisModule.render(analysisContent, currentData, dataCharacteristics);
        const heading = analysisContent.querySelector('h2');
        if (heading) {
            heading.tabIndex = -1;
            heading.focus({ preventScroll: true });
        }
    } catch (error) {
        console.error(error);
        analysisContent.innerHTML = `<p class="error-message" role="alert">分析機能の読み込みに失敗しました。(${escapeHtml(analysisType)}.js)<br>エラー詳細: ${escapeHtml(error.message)}</p>`;
    }
}

window.backToHome = () => {
    if (activeAnalysisModule?.dispose) activeAnalysisModule.dispose();
    activeAnalysisModule = null;
    clearAIAssistPanelContext();
    const analysisContent = document.getElementById('analysis-content');
    analysisContent.querySelectorAll('.js-plotly-plot').forEach(plot => globalThis.Plotly?.purge(plot));
    analysisContent.replaceChildren();
    document.getElementById('analysis-header').style.display = 'none';
    document.getElementById('analysis-area').style.display = 'none';
    document.getElementById('navigation-section').style.display = 'block';
    document.getElementById('upload-section-main').style.display = 'block';
    if (lastAnalysisTrigger?.isConnected) lastAnalysisTrigger.focus({ preventScroll: true });
};
