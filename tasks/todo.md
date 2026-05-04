# Iteration 18: スタッキングOOFの前処理リーク修正

## 目標
回帰・分類の `stack_models` でメタ学習器に渡すOut-of-Fold予測を、保持訓練データ全体でfit済みの前処理行列ではなく、foldごとの訓練データだけでfitした前処理から生成する。

## タスク

- [x] 1. 回帰・分類のstack実装と前処理fold関数の接続点を確認する
- [x] 2. 回帰stackのOOFメタ特徴量をfold内前処理fitに切り替える
- [x] 3. 分類stackのOOFメタ特徴量をfold内前処理fitに切り替える
- [x] 4. stack結果の画面文言を前処理込みOOFに更新する
- [x] 5. 静的チェックと簡易stack実行確認を実施する
- [x] 6. コミットしてプッシュする

## 成功基準

- 回帰stackのOOF生成で `_state.XTrain` をfold分割しない
- 分類stackのOOF生成で `_state.XTrain` / `_state.yTrain` をfold分割しない
- `prepareTrainValidationFeatures()` を使い、各foldで前処理をfitする
- `node --check`、ES module import確認、簡易stack相当実行確認、`git diff --check` が通る

## レビュー

- 回帰stackのOOF生成を `_state.XTrain` のfold分割から `_state.trainRows` のfold分割へ変更した。
- 回帰stackでは各foldで `prepareTrainValidationFeatures()` を呼び、fold訓練行だけで前処理をfitしてからベースモデルの検証fold予測を生成するようにした。
- 分類stackのOOF生成を `_state.XTrain` / `_state.yTrain` のfold分割から `_state.trainRows` とraw target由来のstack targetのfold分割へ変更した。
- 分類stackでも各foldで `prepareTrainValidationFeatures()` を呼び、fold訓練行だけで前処理をfitするようにした。
- 分類stackの確率特徴量は、モデルごとの `classes` をグローバルなクラス順に揃えてからメタ特徴量へ入れるようにした。
- stack結果の画面文言を「foldごとに前処理をfitしたOut-of-Fold予測」に更新した。
- `node --check js/analyses/regression.js` と `node --check js/analyses/classification.js` は成功。
- ES module import確認は警告のみで成功。
- デモCSVを使い、回帰・分類の前処理込みOOFメタ特徴量が全行・全列で数値として生成されることを確認した。
- `_state.XTrain` をfold分割する旧stack経路が残っていないことを `rg` で確認した。
- `git diff --check` は成功。
