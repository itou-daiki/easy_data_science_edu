// ==========================================
// Gemini AI Interpretation Assistant
// ==========================================
import { getLanguage, LANGUAGE_CHANGE_EVENT, tr } from './i18n.js';

const SETTINGS_KEY = 'easyDataScience.geminiSettings.v1';
const DEFAULT_MODEL = 'gemini-2.5-flash';
const GEMINI_FALLBACK_MODEL = 'gemini-2.5-flash';
const MAX_PREVIEW_ROWS = 10;
const MAX_SUMMARY_COLUMNS = 14;
const MAX_RESULT_ITEMS = 24;
const INTERPRETATION_MAX_OUTPUT_TOKENS = 1800;
const CHAT_MAX_OUTPUT_TOKENS = 1200;

let lastPanelRequest = null;
let lastContextFingerprint = '';
let chatHistory = [];

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
    chatHistory = [];
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
            badge.textContent = tr(active ? '有効' : '未設定');
            badge.classList.toggle('active', active);
        }
        if (status) {
            status.textContent = message || (active
                ? aiText(
                    '生成AI支援は有効です。「解釈を生成」や追加質問を押すと、分析文脈の要約がGoogle Gemini APIへ送信されます。AI用テキストのコピーも利用できます。',
                    'Generative AI support is enabled. When you generate an interpretation or ask a follow-up, a summarized analysis context is sent to the Google Gemini API. Copying text for another AI is also available.'
                )
                : aiText(
                    'APIキー未設定でも、分析結果ページでAI用テキストをコピーできます。Geminiで解釈生成や追加質問を使う場合だけAPIキーを入力してください。',
                    'You can copy text for another AI without an API key. Enter a key only when you want Gemini to generate an interpretation or answer follow-up questions.'
                ));
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
            updateStatus(aiText('APIキーを入力してください。', 'Enter an API key.'));
            return;
        }

        sessionStorage.setItem(SETTINGS_KEY, JSON.stringify({ apiKey, model }));
        updateStatus(aiText(
            '生成AI支援を有効化しました。分析結果ページで解釈生成、追加質問、AI用テキストコピーを利用できます。',
            'Generative AI support is enabled. You can generate interpretations, ask follow-up questions, and copy text for another AI on result pages.'
        ));
        window.dispatchEvent(new CustomEvent('ai-assist-settings-changed'));
    });

    clearButton.addEventListener('click', () => {
        clearAISettings();
        updateStatus(aiText(
            'Gemini APIキーを削除しました。解釈生成と追加質問は無効ですが、AI用テキストのコピーは利用できます。',
            'The Gemini API key was removed. Interpretation and follow-up questions are disabled, but copying text for another AI remains available.'
        ));
    });

    window.addEventListener('ai-assist-settings-changed', () => {
        updateStatus();
        if (lastPanelRequest) {
            renderAIAssistPanel(lastPanelRequest);
        } else {
            removeAIAssistPanel();
        }
    });

    window.addEventListener(LANGUAGE_CHANGE_EVENT, () => {
        updateStatus();
        if (lastPanelRequest) renderAIAssistPanel(lastPanelRequest);
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

export function renderAIAssistPanel({
    context,
    title = '生成AI解釈補助',
    ready = true,
    unavailableReason = '分析に使う変数を選択し、結果を生成すると利用できます。'
}) {
    lastPanelRequest = { context, title, ready, unavailableReason };

    removeAIAssistPanel();
    if (!context) return;

    const contextFingerprint = createContextFingerprint(context);
    if (contextFingerprint !== lastContextFingerprint) {
        chatHistory = [];
        lastContextFingerprint = contextFingerprint;
    }

    const hasApiKey = isAIAssistActive();
    const copyDisabled = !ready;
    const generateDisabled = !ready || !hasApiKey;
    const chatDisabled = !ready || !hasApiKey;
    const copyTitle = ready
        ? aiText(
            '分析結果を他の生成AIへ貼り付けるためのテキストとしてコピーします',
            'Copy the analysis context for use with another generative AI'
        )
        : tr(unavailableReason);
    const generateTitle = !ready
        ? tr(unavailableReason)
        : (hasApiKey
            ? aiText('Geminiで解釈を生成', 'Generate an interpretation with Gemini')
            : aiText('Geminiで解釈を生成するにはAPIキーを設定してください', 'Set an API key to generate an interpretation with Gemini'));
    const chatTitle = !ready
        ? tr(unavailableReason)
        : (hasApiKey
            ? aiText('分析結果についてGeminiに追加質問します', 'Ask Gemini a follow-up question about the results')
            : aiText('追加質問にはGemini APIキーが必要です', 'A Gemini API key is required for follow-up questions'));
    const outputMessage = createInitialOutputMessage({ ready, hasApiKey, unavailableReason });

    const panel = document.createElement('aside');
    panel.id = 'ai-assist-floating-panel';
    panel.className = 'ai-assist-floating';
    panel.setAttribute('aria-label', aiText('生成AIによる解釈補助', 'AI interpretation support'));

    panel.innerHTML = `
        <div class="ai-assist-header">
            <div>
                <div class="ai-assist-title"><i class="fas fa-magic"></i> ${escapeHtml(tr(title))}</div>
                <div class="ai-assist-subtitle">${escapeHtml(tr(context.method || '分析結果'))}</div>
            </div>
            <div class="ai-assist-icon-actions">
                <button type="button" class="ai-assist-icon-button" data-action="collapse" title="${escapeHtml(tr('折りたたむ'))}">
                    <i class="fas fa-minus"></i>
                </button>
                <button type="button" class="ai-assist-icon-button" data-action="close" title="${escapeHtml(tr('閉じる'))}">
                    <i class="fas fa-xmark"></i>
                </button>
            </div>
        </div>
        <div class="ai-assist-body">
            <div class="ai-assist-context">
                <strong>${escapeHtml(tr('読み取り対象'))}</strong>
                <span>${escapeHtml(createContextLine(context))}</span>
            </div>
            <div class="ai-assist-privacy-note">
                <i class="fas fa-lock"></i>
                <span>${escapeHtml(aiText(
                    'コピーはブラウザ内で完結します。「解釈を生成」または追加質問を押した場合だけ、要約文脈が Google Gemini API へ送信されます。アップロードファイル全体は送信しません。',
                    'Copying stays in your browser. A summarized context is sent to the Google Gemini API only when you generate an interpretation or ask a follow-up. The entire uploaded file is never sent.'
                ))}</span>
            </div>
            <div class="ai-assist-actions">
                <button type="button" class="ai-assist-copy" ${copyDisabled ? 'disabled' : ''} title="${escapeHtml(copyTitle)}">
                    <i class="fas fa-copy"></i> ${escapeHtml(tr('AI用テキストをコピー'))}
                </button>
                <button type="button" class="ai-assist-generate" ${generateDisabled ? 'disabled' : ''} title="${escapeHtml(generateTitle)}">
                    <i class="fas fa-lightbulb"></i> ${escapeHtml(tr('解釈を生成'))}
                </button>
            </div>
            <div class="ai-assist-output" aria-live="polite">
                ${escapeHtml(outputMessage)}
            </div>
            <div class="ai-assist-chat-area">
                <textarea class="ai-assist-chat-input" rows="2" placeholder="${escapeHtml(tr('例: この結果をレポート用に短く書くと？'))}" title="${escapeHtml(chatTitle)}" ${chatDisabled ? 'disabled' : ''}></textarea>
                <button type="button" class="ai-assist-chat-send" title="${escapeHtml(chatTitle)}" ${chatDisabled ? 'disabled' : ''}>
                    <i class="fas fa-paper-plane"></i> ${escapeHtml(tr('質問'))}
                </button>
            </div>
        </div>
    `;

    document.body.appendChild(panel);

    const body = panel.querySelector('.ai-assist-body');
    const output = panel.querySelector('.ai-assist-output');
    const copyButton = panel.querySelector('.ai-assist-copy');
    const generateButton = panel.querySelector('.ai-assist-generate');
    const chatInput = panel.querySelector('.ai-assist-chat-input');
    const chatButton = panel.querySelector('.ai-assist-chat-send');

    panel.querySelector('[data-action="close"]').addEventListener('click', () => {
        panel.remove();
    });

    panel.querySelector('[data-action="collapse"]').addEventListener('click', () => {
        const collapsed = panel.classList.toggle('collapsed');
        body.style.display = collapsed ? 'none' : 'block';
        panel.querySelector('[data-action="collapse"] i').className = collapsed ? 'fas fa-plus' : 'fas fa-minus';
    });

    copyButton.addEventListener('click', async () => {
        if (!ready) return;

        const originalHtml = copyButton.innerHTML;
        copyButton.disabled = true;
        try {
            await copyTextToClipboard(buildPrompt(context));
            copyButton.innerHTML = `<i class="fas fa-check"></i> ${escapeHtml(tr('コピーしました'))}`;
            output.textContent = aiText(
                'AI用テキストをクリップボードにコピーしました。外部AIに貼り付けて利用できます。',
                'The analysis text was copied to the clipboard. Paste it into another AI service to use it.'
            );
        } catch {
            output.textContent = aiText(
                'クリップボードへのコピーに失敗しました。ブラウザの権限設定を確認してください。',
                'Could not copy to the clipboard. Check the browser permission settings.'
            );
        } finally {
            setTimeout(() => {
                copyButton.innerHTML = originalHtml;
                copyButton.disabled = copyDisabled;
            }, 1600);
        }
    });

    generateButton.addEventListener('click', async () => {
        if (!ready) return;

        generateButton.disabled = true;
        generateButton.classList.add('is-loading');
        output.textContent = aiText('Gemini に解釈を依頼しています...', 'Asking Gemini to interpret the results...');

        try {
            const responseText = await requestGeminiInterpretation(context);
            chatHistory = [{ role: 'assistant', text: responseText }];
            output.textContent = responseText;
        } catch (error) {
            output.textContent = error.message;
        } finally {
            generateButton.disabled = !ready || !isAIAssistActive();
            chatInput.disabled = !ready || !isAIAssistActive();
            chatButton.disabled = !ready || !isAIAssistActive();
            generateButton.classList.remove('is-loading');
        }
    });

    const sendChatMessage = async () => {
        if (!ready || !isAIAssistActive()) return;

        const question = chatInput.value.trim();
        if (!question) return;

        chatInput.value = '';
        chatInput.disabled = true;
        chatButton.disabled = true;
        generateButton.disabled = true;

        const previousOutput = output.textContent.trim();
        output.textContent = `${previousOutput}\n\n${aiText('質問', 'Question')}: ${question}\n\n${aiText('回答を生成しています...', 'Generating an answer...')}`;

        try {
            const answer = await requestGeminiChat(context, question);
            chatHistory.push({ role: 'user', text: question }, { role: 'assistant', text: answer });
            chatHistory = chatHistory.slice(-10);
            output.textContent = `${previousOutput}\n\n${aiText('質問', 'Question')}: ${question}\n\n${answer}`;
        } catch (error) {
            output.textContent = `${previousOutput}\n\n${aiText('質問', 'Question')}: ${question}\n\n${aiText('回答に失敗しました。', 'Could not generate an answer.')}\n${error.message}`;
        } finally {
            const disabled = !ready || !isAIAssistActive();
            chatInput.disabled = disabled;
            chatButton.disabled = disabled;
            generateButton.disabled = disabled;
        }
    };

    chatButton.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendChatMessage();
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

function createInitialOutputMessage({ ready, hasApiKey, unavailableReason }) {
    if (!ready) return tr(unavailableReason);
    if (!hasApiKey) {
        return aiText(
            'AI用テキストをコピーして、ChatGPT、Gemini、Claudeなどに貼り付けて使えます。Geminiで直接生成や追加質問を使う場合は、ページ上部の「生成AI支援」からAPIキーを設定してください。',
            'Copy the analysis text and paste it into ChatGPT, Gemini, Claude, or another AI. To generate directly with Gemini or ask follow-up questions, set an API key under “Generative AI support” at the top of the page.'
        );
    }
    return aiText(
        '分析結果の表や指標を踏まえて、解釈生成・追加質問・AI用テキストコピーを利用できます。',
        'Use the result tables and metrics to generate an interpretation, ask follow-up questions, or copy text for another AI.'
    );
}

function createContextFingerprint(context) {
    return JSON.stringify({
        method: context.method,
        dataStructure: context.dataStructure,
        resultSummary: context.resultSummary,
        notes: context.notes
    });
}

async function copyTextToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
    }

    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    textarea.style.top = '0';
    document.body.appendChild(textarea);
    textarea.select();

    const copied = document.execCommand('copy');
    document.body.removeChild(textarea);
    if (!copied) {
        throw new Error('copy failed');
    }
}

async function requestGeminiInterpretation(context) {
    return requestGemini(buildPrompt(context), INTERPRETATION_MAX_OUTPUT_TOKENS);
}

async function requestGeminiChat(context, question) {
    return requestGemini(buildChatPrompt(context, question), CHAT_MAX_OUTPUT_TOKENS);
}

async function requestGemini(prompt, maxOutputTokens) {
    const settings = getAISettings();
    if (!settings.apiKey.trim()) {
        throw new Error(aiText(
            'Gemini APIキーが未設定です。ページ上部の「生成AI支援」からキーを入力してください。',
            'No Gemini API key is configured. Enter a key under “Generative AI support” at the top of the page.'
        ));
    }

    const errors = [];
    for (const model of createGeminiModelChain(settings.model)) {
        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
        let response;
        try {
            response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-goog-api-key': settings.apiKey
                },
                body: JSON.stringify({
                    system_instruction: {
                        parts: [{
                            text: aiText(
                                'あなたはデータサイエンス教育のチューターです。提供された分析結果だけを根拠に、日本語で初学者にもわかるように説明してください。性能指標だけでなく、データ量、前処理、過学習、評価設計、限界も扱い、因果関係は断定しないでください。',
                                'You are a data science tutor. Explain the supplied analysis in clear English for a beginner, using only the provided results. Address sample size, preprocessing, overfitting, evaluation design, and limitations as well as performance metrics. Do not claim causation.'
                            )
                        }]
                    },
                    contents: [{
                        role: 'user',
                        parts: [{ text: prompt }]
                    }],
                    generationConfig: {
                        temperature: 0.2,
                        topP: 0.8,
                        maxOutputTokens
                    }
                })
            });
        } catch {
            throw new Error(aiText(
                'Gemini APIに接続できませんでした。ネットワーク接続またはブラウザの通信制限を確認してください。',
                'Could not connect to the Gemini API. Check the network connection and browser communication restrictions.'
            ));
        }

        const responseText = await response.text();
        const payload = parseJsonSafely(responseText);
        if (!response.ok) {
            const apiMessage = payload.error?.message || responseText || `HTTP ${response.status}`;
            errors.push({ model, status: response.status, message: apiMessage });
            if (shouldTryFallbackGeminiModel(model, response.status, apiMessage)) {
                continue;
            }
            throw createGeminiRequestError(errors);
        }

        const text = extractGeminiText(payload);
        if (!text) {
            errors.push({
                model,
                status: response.status,
                message: aiText('Geminiから解釈文を取得できませんでした。', 'Gemini returned no interpretation text.')
            });
            continue;
        }
        return text;
    }

    throw createGeminiRequestError(errors);
}

function createGeminiModelChain(model) {
    const normalized = (model || DEFAULT_MODEL).replace(/^models\//, '');
    return [normalized, GEMINI_FALLBACK_MODEL].filter((item, index, array) => item && array.indexOf(item) === index);
}

function shouldTryFallbackGeminiModel(model, status, errorText) {
    if (model === GEMINI_FALLBACK_MODEL) return false;
    if (![400, 403, 404].includes(status)) return false;
    return /model|not found|not supported|unavailable|permission|access|preview|quota|billing/i.test(String(errorText));
}

function createGeminiRequestError(errors) {
    const latest = errors.at(-1);
    const detail = errors
        .map(error => `${error.model}: HTTP ${error.status} ${String(error.message || '').slice(0, 180)}`)
        .join('\n');

    if (latest?.status === 401 || latest?.status === 403) {
        return new Error(`${aiText(
            'Gemini APIキーが無効、または権限がありません。APIキーと利用設定を確認してください。',
            'The Gemini API key is invalid or lacks permission. Check the key and API settings.'
        )}\n${detail}`);
    }
    if (latest?.status === 400 || latest?.status === 404) {
        return new Error(`${aiText(
            'Geminiモデル名またはリクエスト内容を確認してください。',
            'Check the Gemini model name and request settings.'
        )}\n${detail}`);
    }
    return new Error(`${aiText(
        'Gemini APIの呼び出しに失敗しました。',
        'The Gemini API request failed.'
    )}\n${detail}`);
}

function parseJsonSafely(text) {
    try {
        return JSON.parse(text);
    } catch {
        return {};
    }
}

function extractGeminiText(payload) {
    return payload.candidates?.[0]?.content?.parts
        ?.map(part => part.text || '')
        .filter(Boolean)
        .join('\n')
        .trim();
}

function buildPrompt(context) {
    const compactContext = JSON.stringify(context, null, 2);
    if (getLanguage() === 'en') {
        return `You are tutoring a beginner who is learning data analysis.
Use only the information shown in the analysis context below to explain the results.

Use exactly these six section headings:
1. What the results show
2. Numbers to notice
3. Reliability and validity check
4. Interpretation cautions
5. Report examples
6. What to check next

Length and structure:
- Aim for roughly 700-1,000 words; do not stop at a superficial summary
- Include 2-4 bullet points in each of sections 1-4
- In “Report examples,” provide both a short report paragraph and a slightly more detailed version
- In “What to check next,” give three concrete actions the user can take

Constraints:
- Start with specific variables, models, statistics, metrics, or cautions from the supplied results
- Do not begin with a generic explanation of the analysis method
- Prioritize the most important finding visible in the results and include numerical evidence
- In the reliability and validity section, address applicable issues such as sample size, missing values, outliers, overfitting, data leakage, the difference between CV and test evaluation, class imbalance, and feature count
- Do not invent values or conclusions that are not supplied
- Clearly say when additional verification is required
- If the context does not contain usable statistics, say that the result table could not be read sufficiently instead of filling the gap with general advice
- Even when performance looks strong, discuss the independent test, CV variability, data volume, and nature of the target cautiously
- Do not infer causation from correlation or regression alone
- Use natural, beginner-friendly English while preserving enough detail to explain evidence, meaning, and limitations
- Do not use Markdown level-two or larger headings

Poor opening example:
“This analysis compares the performance of machine learning models.”

Better opening example:
“The random forest achieved a test R² of 0.82 and an RMSE of 12.4, the lowest error among the compared models, but the wide variation in CV R² means its stability needs further checking.”

Analysis context:
${compactContext}`;
    }

    return `あなたは日本語でデータ分析を学ぶ初学者を支援するチューターです。
以下の分析画面に表示されている情報だけを根拠に、ユーザーが結果を理解できるように説明してください。

出力形式（見出しはこの6つだけ）:
1. 結果から言えること
2. 注目すべき数値
3. 信頼性と妥当性チェック
4. 解釈で注意すること
5. レポート例
6. 次に確認すること

分量の目安:
- 全体で900〜1400字程度を目安にし、短すぎる要約で終わらせない
- 1〜4の各見出しには2〜4個の箇条書きを入れる
- 「レポート例」には、短いレポート文と少し詳しいレポート文の2種類を書く
- 「次に確認すること」は、ユーザーが次に操作・確認できる具体的な行動を3つ書く

制約:
- 1文目から、分析結果にある具体的な変数名・モデル名・統計量・性能指標・注意点などに基づいて説明する
- 「この分析は何を調べるものです」のような分析手法の一般説明で始めない
- 表示されている結果から読み取れる最も重要な内容を優先し、数値を必ず含める
- 「信頼性と妥当性チェック」では、サンプルサイズ、欠損、外れ値、過学習、データリーク、CVとテスト評価の違い、クラス不均衡、特徴量数など、該当する注意点を必ず扱う
- 与えられた情報にない数値や結論を作らない
- 不確実な点は「追加確認が必要」と明示する
- 分析結果表や抽出テキストに具体的な統計量がない場合は、一般論で埋めず「結果表を十分に読み取れませんでした」と明記する
- 分類・回帰の性能が良く見えても、独立テスト、CVのばらつき、データ量、目的変数の性質を踏まえて慎重に述べる
- 相関や回帰だけで因果関係を断定しない
- 初学者にわかる自然な日本語で、根拠・意味・注意点がわかる十分な説明量にする
- Markdownの大見出し（##など）は使わない

悪い出力例:
「この分析は、機械学習モデルの性能を比較するものです。」

良い出力例:
「ランダムフォレストのTest R²は0.82、RMSEは12.4で、比較したモデルの中では誤差が小さい一方、CV R²のばらつきが大きいため安定性には注意が必要です。」

分析画面の情報:
${compactContext}`;
}

function buildChatPrompt(context, question) {
    const compactContext = JSON.stringify(context, null, 2);
    const history = chatHistory
        .slice(-8)
        .map(item => `${item.role === 'user' ? aiText('ユーザー', 'User') : 'AI'}: ${item.text}`)
        .join('\n\n');

    if (getLanguage() === 'en') {
        return `You are tutoring a beginner who is learning data analysis.
Answer the follow-up question using only the analysis context and conversation below.

Rules:
- Answer the question directly first
- Prefer specific variables, models, metrics, statistics, cautions, and numerical evidence shown in the context
- When relevant, check reliability and validity issues such as sample size, missing values, outliers, overfitting, data leakage, CV versus test evaluation, class imbalance, and feature count
- Do not guess information that is absent; say “This cannot be determined from the results shown here”
- Do not infer causation from correlation or regression alone
- Keep the answer focused and use bullets when helpful
- Do not use Markdown level-two or larger headings

Analysis context:
${compactContext}

Conversation so far:
${history || 'There is no previous conversation.'}

Follow-up question:
${question}`;
    }

    return `あなたは日本語でデータ分析を学ぶ初学者を支援するチューターです。
以下の分析画面に表示されている情報と、これまでの会話だけを根拠に、ユーザーの追加質問へ答えてください。

回答ルール:
- 最初に質問へ直接答える
- 具体的な変数名、モデル名、性能指標、統計量、注意点など、表示されている数値を優先して使う
- 必要に応じて、サンプルサイズ、欠損、外れ値、過学習、データリーク、CVとテスト評価の違い、クラス不均衡、特徴量数などの信頼性・妥当性を確認する
- 分析結果にない情報は推測せず、「この画面の結果だけでは判断できません」と言う
- 相関や回帰だけで因果関係を断定しない
- 長くなりすぎないように、必要なら箇条書きで答える
- Markdownの大見出し（##など）は使わない

分析画面の情報:
${compactContext}

これまでの会話:
${history || 'まだ会話はありません。'}

ユーザーの追加質問:
${question}`;
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
    if (getLanguage() === 'en') {
        return `${context.preview?.length || 0} preview rows, ${context.summaryStatistics?.length || 0} summary columns, ${rows} rows x ${cols} columns, ${resultKeys} result items`;
    }
    return `先頭${context.preview?.length || 0}件、要約統計量${context.summaryStatistics?.length || 0}列、${rows}行${cols}列、結果項目${resultKeys}件`;
}

function aiText(japanese, english) {
    return getLanguage() === 'en' ? english : japanese;
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
