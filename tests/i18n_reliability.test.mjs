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
