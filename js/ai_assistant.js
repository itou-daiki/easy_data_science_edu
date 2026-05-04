// ==========================================
// Gemini AI Interpretation Assistant
// ==========================================

const SETTINGS_KEY = 'easyDataScience.geminiSettings.v1';
const DEFAULT_MODEL = 'gemini-2.5-flash';
const MAX_PREVIEW_ROWS = 10;
const MAX_SUMMARY_COLUMNS = 14;
const MAX_RESULT_ITEMS = 24;

let lastPanelRequest = null;

export function getAISettings() {
    try {
        const raw = sessionStorage.getItem(SETTINGS_KEY);
        if (!raw) return { apiKey: '', model: DEFAULT_MODEL };
        const parsed = JSON.parse(raw);
        return {
            apiKey: parsed.apiKey || '',
            model: parsed.model || DEFAULT_MODEL
        };
    } catch {
        return { apiKey: '', model: DEFAULT_MODEL };
    }
}

export function isAIAssistActive() {
    return getAISettings().apiKey.trim().length > 0;
}

export function clearAISettings() {
    sessionStorage.removeItem(SETTINGS_KEY);
    window.dispatchEvent(new CustomEvent('ai-assist-settings-changed'));
}

export function setupAIAssistSettingsUI(elements) {
    const {
        button,
        modal,
        closeButton,
        apiKeyInput,
        modelInput,
        status,
        saveButton,
        clearButton,
        badge
    } = elements;

    if (!button || !modal || !apiKeyInput || !modelInput || !saveButton || !clearButton) return;

    const updateStatus = (message = '') => {
        const settings = getAISettings();
        const active = settings.apiKey.trim().length > 0;
        if (badge) {
            badge.textContent = active ? '有効' : '未設定';
            badge.classList.toggle('active', active);
        }
        if (status) {
            status.textContent = message || (active
                ? '生成AI支援は有効です。「解釈を生成」を押すと、分析文脈の要約がGoogle Gemini APIへ送信されます。APIキーはこのブラウザタブ内にのみ保持されます。'
                : 'Gemini APIキーを入力すると、分析結果ページで解釈補助を利用できます。送信は有効化後に「解釈を生成」を押した時だけ行われます。');
            status.classList.toggle('active', active);
        }
        modelInput.value = settings.model || DEFAULT_MODEL;
        apiKeyInput.value = '';
    };

    const closeModal = () => {
        modal.style.display = 'none';
    };

    button.addEventListener('click', () => {
        updateStatus();
        modal.style.display = 'block';
        setTimeout(() => apiKeyInput.focus(), 0);
    });

    if (closeButton) closeButton.addEventListener('click', closeModal);

    window.addEventListener('click', (event) => {
        if (event.target === modal) closeModal();
    });

    saveButton.addEventListener('click', () => {
        const apiKey = apiKeyInput.value.trim();
        const model = (modelInput.value.trim() || DEFAULT_MODEL).replace(/^models\//, '');

        if (!apiKey) {
            updateStatus('APIキーを入力してください。');
            return;
        }

        sessionStorage.setItem(SETTINGS_KEY, JSON.stringify({ apiKey, model }));
        updateStatus('生成AI支援を有効化しました。分析結果ページで補助パネルを利用できます。');
        window.dispatchEvent(new CustomEvent('ai-assist-settings-changed'));
    });

    clearButton.addEventListener('click', () => {
        clearAISettings();
        updateStatus('生成AI支援を無効化しました。');
    });

    window.addEventListener('ai-assist-settings-changed', () => {
        updateStatus();
        if (isAIAssistActive() && lastPanelRequest) {
            renderAIAssistPanel(lastPanelRequest);
        } else {
            removeAIAssistPanel();
        }
    });

    updateStatus();
}

export function buildAnalysisContext({ data, characteristics, method, resultSummary = {}, notes = [] }) {
    return {
        method,
        preview: createDataPreview(data),
        summaryStatistics: createSummaryStatistics(data, characteristics),
        dataStructure: createDataStructure(data, characteristics),
        resultSummary: normalizeResultSummary(resultSummary),
        notes: Array.isArray(notes) ? notes : [notes].filter(Boolean)
    };
}

export function renderAIAssistPanel({ context, title = '生成AI解釈補助' }) {
    lastPanelRequest = { context, title };

    removeAIAssistPanel();
    if (!isAIAssistActive() || !context) return;

    const panel = document.createElement('aside');
    panel.id = 'ai-assist-floating-panel';
    panel.className = 'ai-assist-floating';
    panel.setAttribute('aria-label', '生成AIによる解釈補助');

    panel.innerHTML = `
        <div class="ai-assist-header">
            <div>
                <div class="ai-assist-title"><i class="fas fa-magic"></i> ${escapeHtml(title)}</div>
                <div class="ai-assist-subtitle">${escapeHtml(context.method || '分析結果')}</div>
            </div>
            <div class="ai-assist-icon-actions">
                <button type="button" class="ai-assist-icon-button" data-action="collapse" title="折りたたむ">
                    <i class="fas fa-minus"></i>
                </button>
                <button type="button" class="ai-assist-icon-button" data-action="close" title="閉じる">
                    <i class="fas fa-xmark"></i>
                </button>
            </div>
        </div>
        <div class="ai-assist-body">
            <div class="ai-assist-context">
                <strong>読み取り対象</strong>
                <span>${escapeHtml(createContextLine(context))}</span>
            </div>
            <div class="ai-assist-privacy-note">
                <i class="fas fa-lock"></i>
                <span>実行時、この要約文脈が Google Gemini API へ送信されます。アップロードファイル全体は送信しません。</span>
            </div>
            <button type="button" class="ai-assist-generate">
                <i class="fas fa-lightbulb"></i> 解釈を生成
            </button>
            <div class="ai-assist-output" aria-live="polite">
                分析結果の表や指標を踏まえて、初学者向けに要点を整理します。
            </div>
        </div>
    `;

    document.body.appendChild(panel);

    const body = panel.querySelector('.ai-assist-body');
    const output = panel.querySelector('.ai-assist-output');
    const generateButton = panel.querySelector('.ai-assist-generate');

    panel.querySelector('[data-action="close"]').addEventListener('click', () => {
        panel.remove();
    });

    panel.querySelector('[data-action="collapse"]').addEventListener('click', () => {
        const collapsed = panel.classList.toggle('collapsed');
        body.style.display = collapsed ? 'none' : 'block';
        panel.querySelector('[data-action="collapse"] i').className = collapsed ? 'fas fa-plus' : 'fas fa-minus';
    });

    generateButton.addEventListener('click', async () => {
        generateButton.disabled = true;
        output.textContent = 'Gemini に解釈を依頼しています...';

        try {
            const responseText = await requestGeminiInterpretation(context);
            output.textContent = responseText;
        } catch (error) {
            output.textContent = error.message;
        } finally {
            generateButton.disabled = false;
        }
    });
}

export function removeAIAssistPanel() {
    const existing = document.getElementById('ai-assist-floating-panel');
    if (existing) existing.remove();
}

export function clearAIAssistPanelContext() {
    lastPanelRequest = null;
    removeAIAssistPanel();
}

async function requestGeminiInterpretation(context) {
    const settings = getAISettings();
    if (!settings.apiKey.trim()) {
        throw new Error('Gemini APIキーが未設定です。ページ上部の「生成AI支援」からキーを入力してください。');
    }

    const model = (settings.model || DEFAULT_MODEL).replace(/^models\//, '');
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    const prompt = buildPrompt(context);

    let response;
    try {
        response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-goog-api-key': settings.apiKey
            },
            body: JSON.stringify({
                contents: [{
                    role: 'user',
                    parts: [{ text: prompt }]
                }],
                generationConfig: {
                    temperature: 0.2,
                    maxOutputTokens: 1400
                }
            })
        });
    } catch {
        throw new Error('Gemini APIに接続できませんでした。ネットワーク接続またはブラウザの通信制限を確認してください。');
    }

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        const apiMessage = payload.error?.message ? ` 詳細: ${payload.error.message}` : '';
        if (response.status === 400 || response.status === 404) {
            throw new Error(`Geminiモデル名またはリクエスト内容を確認してください。${apiMessage}`);
        }
        if (response.status === 401 || response.status === 403) {
            throw new Error(`Gemini APIキーが無効、または権限がありません。APIキーを確認してください。${apiMessage}`);
        }
        throw new Error(`Gemini APIの呼び出しに失敗しました。HTTP ${response.status}.${apiMessage}`);
    }

    const text = payload.candidates?.[0]?.content?.parts
        ?.map(part => part.text || '')
        .filter(Boolean)
        .join('\n')
        .trim();

    if (!text) {
        throw new Error('Geminiから解釈文を取得できませんでした。少し時間を置いて再実行してください。');
    }

    return text;
}

function buildPrompt(context) {
    const compactContext = JSON.stringify(context, null, 2);
    return `あなたは日本語でデータ分析を学ぶ初学者を支援するチューターです。
以下の分析画面に表示されている情報だけを根拠に、ユーザーが結果を理解できるように説明してください。
推測で断定せず、不確実な点は「追加確認が必要」と明示してください。

出力は日本語で、次の構成にしてください。
1. まず押さえる結論
2. データ構造と要約統計量から読めること
3. 分析手法と主要指標の読み方
4. 結果の注意点
5. 次に確認するとよいこと

分析画面の情報:
${compactContext}`;
}

function createDataPreview(data) {
    if (!Array.isArray(data)) return [];
    return data.slice(0, MAX_PREVIEW_ROWS).map(row => {
        const cleanRow = {};
        Object.entries(row || {}).forEach(([key, value]) => {
            cleanRow[key] = truncateValue(value);
        });
        return cleanRow;
    });
}

function createDataStructure(data, characteristics) {
    const rowCount = Array.isArray(data) ? data.length : 0;
    const columns = characteristics?.allColumns || (data?.[0] ? Object.keys(data[0]) : []);
    return {
        rowCount,
        columnCount: columns.length,
        numericColumns: characteristics?.numericColumns || [],
        categoricalColumns: characteristics?.categoricalColumns || [],
        textColumns: characteristics?.textColumns || []
    };
}

function createSummaryStatistics(data, characteristics) {
    if (!Array.isArray(data) || data.length === 0) return [];
    const numericColumns = (characteristics?.numericColumns || [])
        .filter(col => Object.prototype.hasOwnProperty.call(data[0], col))
        .slice(0, MAX_SUMMARY_COLUMNS);

    return numericColumns.map(col => {
        const values = data
            .map(row => row[col])
            .filter(value => value != null && value !== '' && Number.isFinite(Number(value)))
            .map(Number);
        const missing = data.length - values.length;
        if (values.length === 0) {
            return { column: col, count: 0, missing };
        }

        const sorted = [...values].sort((a, b) => a - b);
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
        const variance = values.length > 1
            ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
            : 0;

        return {
            column: col,
            count: values.length,
            missing,
            mean: roundNumber(mean),
            std: roundNumber(Math.sqrt(variance)),
            min: roundNumber(sorted[0]),
            median: roundNumber(quantile(sorted, 0.5)),
            max: roundNumber(sorted[sorted.length - 1])
        };
    });
}

function normalizeResultSummary(resultSummary) {
    if (Array.isArray(resultSummary)) return resultSummary.slice(0, MAX_RESULT_ITEMS);
    if (resultSummary && typeof resultSummary === 'object') {
        return Object.fromEntries(Object.entries(resultSummary).slice(0, MAX_RESULT_ITEMS));
    }
    return resultSummary ? { summary: String(resultSummary) } : {};
}

function createContextLine(context) {
    const rows = context.dataStructure?.rowCount ?? 0;
    const cols = context.dataStructure?.columnCount ?? 0;
    const resultKeys = Object.keys(context.resultSummary || {}).length;
    return `先頭${context.preview?.length || 0}件、要約統計量${context.summaryStatistics?.length || 0}列、${rows}行${cols}列、結果項目${resultKeys}件`;
}

function truncateValue(value) {
    if (value == null) return value;
    if (typeof value === 'number') return roundNumber(value);
    const text = String(value);
    return text.length > 80 ? `${text.slice(0, 77)}...` : text;
}

function roundNumber(value) {
    if (!Number.isFinite(value)) return value;
    return Number(value.toFixed(6));
}

function quantile(sortedValues, q) {
    if (sortedValues.length === 0) return null;
    const pos = (sortedValues.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    const next = sortedValues[base + 1];
    return next !== undefined ? sortedValues[base] + rest * (next - sortedValues[base]) : sortedValues[base];
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
