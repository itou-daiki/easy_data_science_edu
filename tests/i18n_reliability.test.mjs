import assert from 'node:assert/strict';

globalThis.localStorage = { getItem: () => null };

const { tr } = await import('../js/i18n.js');

const cases = [
    'データ分割方法:',
    'テストデータ割合と交差検証 Fold 数を選ぶ',
    '無作為層化（独立な行）',
    '陽性クラス（PR/AP）:',
    '3列をOne-Hot Encoding',
    '時系列 expanding-window CV（gap=2行）',
    'Precision-Recall（陽性: yes）',
    '予測スコア: 82.5%',
    '15. 動画・画像分析',
    'メディアは外部送信されません。アノテーションJSONは5 MiB以下で、読込時に形式と件数を検証します。ポーズ推定は遮蔽、画角、衣服、複数人により誤ることがあります。',
    '回帰の新規入力に表示される値は点予測であり、個別の将来観測に対する予測区間ではない',
    '1クラスあたり最大200枚まで追加できます。',
    'Geminiへの送信内容を確認',
    'Interactions APIへの送信では履歴保存を要求しません（store=false）。',
    '分析結果の解釈をGeminiに依頼するための接続設定です。APIキーなしでも、分析結果ページの「AI用テキストをコピー」は使えます。',
    'APIキーはページのメモリ内だけに保持され、再読み込みすると消えます。',
    '履歴保存を要求せず（store=false）、先頭10件の値は既定では送りません。',
    'ブラウザではキーを完全に秘匿できません。利用制限を設定した専用キーを使用してください。',
    '既定は安定版 gemini-3.7-flash です。指定モデルが利用できない場合は安定版 gemini-3.6-flash へ自動で切り替えます。',
    '解釈は8節の構造化JSONとして受信し、型、項目数、文字数を検証してから表示します',
    '回答の末尾で、実際に使われたモデル、総トークン数、互換モデルへの切替を確認できます',
    'クラスごとの比率を保った未学習データで評価しています（固定seed）。各クラスの件数が少ない場合、評価値は大きく変動します。',
    'カードごとに必要なデータ条件があります。回帰は数値の目的変数と別の特徴量、分類は2〜20種類の値を持つ目的変数候補と別の特徴量が必要です。'
];

cases.forEach(value => {
    const translated = tr(value, 'en');
    assert.notEqual(translated, value, `missing English translation: ${value}`);
    assert.doesNotMatch(translated, /[ぁ-んァ-ヶ一-龠]/, `Japanese remains after translation: ${translated}`);
});

assert.equal(tr('無作為層化（独立な行）', 'ja'), '無作為層化（独立な行）');

console.log('i18n_reliability.test.mjs: all assertions passed');
