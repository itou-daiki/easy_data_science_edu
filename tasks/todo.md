# Iteration 16: 前処理込み交差検証への修正

## 目標
回帰・分類の比較、作成、チューニング、ファイナライズ時のCVで、欠損補完、変換、エンコード、特徴量選択、標準化をfoldごとの訓練データだけでfitし、検証foldからの前処理リークを防ぐ。

## タスク

- [x] 1. 既存の前処理パイプラインとCV/GridSearchの境界を確認する
- [x] 2. fold指定の訓練/検証行に対して前処理をfit/transformする関数を追加する
- [x] 3. 前処理込みCVと前処理込みGridSearchを追加する
- [x] 4. 回帰のcompare/create/tune/finalize CVを前処理込みCVへ切り替える
- [x] 5. 分類のcompare/create/tune/finalize CVを前処理込みCVへ切り替える
- [x] 6. 画面文言と信頼性チェックの説明を更新する
- [x] 7. 静的チェック、簡易実行確認、差分レビューを行う
- [x] 8. コミットしてプッシュする

## 成功基準

- CVの各foldで前処理が訓練foldだけにfitされる
- 回帰・分類の比較順位とチューニングが新しいCV経路を使う
- 画面上のCV説明が「前処理後データCV」ではなく「前処理込みCV」になっている
- `node --check`、ES module import確認、簡易CV実行確認、`git diff --check` が通る

## レビュー

- `prepareTrainValidationFeatures()` を追加し、指定されたfoldの訓練行だけで欠損補完、カテゴリエンコード、外れ値除去、特徴量変換、多重共線性除去、標準化をfitし、検証行へtransformできるようにした。
- `crossValidateWithPreprocessing()` と `gridSearchWithPreprocessing()` を追加し、raw row単位でfold分割してから各fold内で前処理をfitする経路を作った。
- 回帰のcompare/create/tune/finalizeのCVを前処理込みCVへ切り替えた。
- 分類のcompare/create/tune/finalizeのCVを前処理込みCVへ切り替えた。
- 信頼性チェック、画面文言、使い方マニュアルから「前処理後データCV」の注意を外し、「foldごとの訓練データだけで前処理をfitするCV」に更新した。
- `node --check js/ml/preprocessing.js`、`node --check js/ml/model_selection.js`、`node --check js/analyses/regression.js`、`node --check js/analyses/classification.js`、`node --check js/analysis_quality.js` は成功。
- ES module import確認は警告のみで成功。
- デモCSVを使った前処理込みCVと前処理込みGridSearchの簡易実行確認は成功。
- `git diff --check` は成功。
- 残リスク: スタッキングのOOFメタ特徴量生成は、まだ保持訓練データ全体でfit済みの前処理行列を使っている。比較・チューニングのCVリークは解消したが、スタッキングは次の改善候補。
