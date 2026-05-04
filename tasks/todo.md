# Iteration 19: finalize_modelの全データ前処理整合

## 目標

`finalize_model` 実行後に「全データで再学習したモデル」と、`predict_model` / モデルJSONで使われる前処理器が同じfit母集団になるようにする。教育用ツールとして、確定モデルの説明と実際の予測挙動がずれない状態にする。

## タスク

- [x] 1. 回帰・分類のfinalize実装と予測前処理の参照先を確認する
- [x] 2. 全データfit用の前処理経路を追加する
- [x] 3. 回帰finalizeで全データfit済み前処理を保存・利用する
- [x] 4. 分類finalizeで全データfit済み前処理・ラベル情報を保存・利用する
- [x] 5. 静的チェックと簡易検証を実施する
- [x] 6. コミットしてプッシュする

## 成功基準

- `finalize_model` の学習データが保持訓練fit済み行列ではなく、全データfit済み前処理から生成される
- `predict_model` が確定モデルと同じ全データfit済み前処理器を使う
- モデルJSONエクスポートの `featureNames` / `scaler` / `encoders` / `labelEncoder` が確定モデル用に更新される
- `node --check`、ES module import確認、簡易finalize相当検証、`git diff --check` が通る

## レビュー

- `prepareTrainTestFeatures()` で `testSize: 0` を許可し、テスト分割を作らず全有効行に前処理をfitできるようにした。
- 空のholdout変換で `StandardScaler.transform([])` に進まないよう、`transformRows()` に空配列処理を追加した。
- 回帰 `finalize_model` は `_state.XTrain` / `_state.XTest` の結合ではなく、全データfit済み前処理から `XFull` / `yFull` を作って再学習するようにした。
- 分類 `finalize_model` も全データfit済み前処理から再学習し、確定モデル用の `labelEncoder` / class labels を保存するようにした。
- `predict_model` は確定モデルがある場合、確定モデル用の前処理器を優先して入力を変換するようにした。
- モデルJSONエクスポートは確定モデル用の `featureNames` / `scaler` / `encoders` / `labelEncoder` / `classLabels` を優先するようにした。
- `node --check`、ES module import確認、`git diff --check` は成功。
- デモCSVで `testSize: 0` の全データ前処理、入力変換、DecisionTreeによるfinalize相当のfit/predictが回帰・分類とも成功。
- 旧finalize経路である `_state.XTrain` / `_state.XTest` の単純結合が残っていないことを `rg` で確認した。
