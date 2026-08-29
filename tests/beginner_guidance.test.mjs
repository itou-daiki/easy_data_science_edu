import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

globalThis.localStorage = { getItem: () => 'ja', setItem: () => {} };

const { createBeginnerGuide } = await import('../js/utils.js');
await Promise.all([
    import('../js/analyses/regression.js'),
    import('../js/analyses/classification.js')
]);

const html = createBeginnerGuide({
    title: { ja: '<確認>', en: 'Read "carefully"' },
    purpose: { ja: '目的', en: 'Purpose' },
    lookFor: { ja: '見る場所', en: 'Where to look' },
    nextAction: { ja: '次の操作', en: 'Next action' },
    caution: { ja: '断定しない', en: 'Do not overclaim' },
    terms: [{
        term: { ja: '平均', en: 'Mean' },
        meaning: { ja: '<script>alert(1)</script>', en: 'Average value' }
    }]
});

assert.match(html, /class="beginner-guide"/);
assert.equal((html.match(/class="beginner-guide-step"/g) || []).length, 3);
assert.match(html, /data-i18n-en="Purpose"/);
assert.match(html, /data-i18n-en="Read &quot;carefully&quot;"/);
assert.doesNotMatch(html, /<script>/);
assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
assert.match(html, /ことばの意味/);

const files = await Promise.all([
    readFile(new URL('../index.html', import.meta.url), 'utf8'),
    readFile(new URL('../css/style.css', import.meta.url), 'utf8'),
    readFile(new URL('../js/utils.js', import.meta.url), 'utf8'),
    readFile(new URL('../js/analyses/eda.js', import.meta.url), 'utf8'),
    readFile(new URL('../js/analyses/preprocessing.js', import.meta.url), 'utf8'),
    readFile(new URL('../js/analyses/regression.js', import.meta.url), 'utf8'),
    readFile(new URL('../js/analyses/classification.js', import.meta.url), 'utf8'),
    readFile(new URL('../js/ai_assistant.js', import.meta.url), 'utf8'),
    readFile(new URL('../js/main.js', import.meta.url), 'utf8'),
    readFile(new URL('../manual.html', import.meta.url), 'utf8')
]);
const [indexHtml, styleCss, utilsSource, edaSource, preprocessingSource, regressionSource, classificationSource, aiSource, mainSource, manualHtml] = files;

assert.match(indexHtml, /class="beginner-start"/);
assert.match(indexHtml, /この順番なら迷いません/);
assert.match(styleCss, /\.beginner-guide-steps\s*\{/);
assert.match(styleCss, /@media \(max-width: 480px\)/);
assert.match(styleCss, /\.create-model-config-grid\s*\{/);
assert.match(utilsSource, /renderConfusionMatrix[\s\S]*?autosize:\s*true/);
assert.doesNotMatch(utilsSource, /renderConfusionMatrix[\s\S]*?width:\s*450/);
[edaSource, preprocessingSource, regressionSource, classificationSource, aiSource].forEach(source => {
    assert.match(source, /createBeginnerGuide\(/);
});
assert.match(regressionSource, /MAE/);
assert.match(classificationSource, /混同行列/);
assert.match(aiSource, /plain_summary/);
assert.match(aiSource, /next_steps/);
assert.match(mainSource, /dataPreviewSection\.querySelectorAll\('\.collapsible-header'\)\.forEach\(configureCollapsibleHeader\)/);
assert.doesNotMatch(mainSource, /cloneNode\(true\)/);
assert.match(manualHtml, /id="sources"/);
assert.match(manualHtml, /W3C WAI/);
assert.match(manualHtml, /UNESCO/);

console.log('beginner_guidance.test.mjs: all assertions passed');
