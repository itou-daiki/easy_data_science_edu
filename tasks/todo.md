# Iteration 17: 乱数モデルの再現性対応

## 目標
RandomForest、GradientBoosting、SVMなど乱数を使うモデルで `Math.random()` への直接依存をなくし、`randomState` を指定すれば同じデータ・同じ設定で同じ結果になるようにする。

## タスク

- [x] 1. `Math.random()` 使用箇所と乱数モデルの `getParams()` を確認する
- [x] 2. seed付き乱数ヘルパーを追加する
- [x] 3. 回帰RandomForest/GradientBoostingを `randomState` 対応にする
- [x] 4. 分類RandomForest/GradientBoosting/SVMを `randomState` 対応にする
- [x] 5. 回帰・分類UIのモデル定義に `randomState: 42` を明示する
- [x] 6. 静的チェックと簡易再現性テストを実施する
- [x] 7. コミットしてプッシュする

## 成功基準

- `rg "Math.random" js/ml` で対象ML実装に直接利用が残らない
- 同じ `randomState` の同一モデルを2回fitして、予測と特徴量重要度が一致する
- 異なる `randomState` では、少なくとも乱数を使うモデルの内部サンプル/特徴量選択が変わり得る
- `node --check`、ES module import確認、`git diff --check` が通る

## レビュー

- `js/ml/random.js` を追加し、Mulberry32ベースのseed付き乱数、整数サンプリング、Fisher-Yatesシャッフルを共通化した。
- 回帰RandomForestのbootstrapと特徴量サンプリングを `randomState` で再現可能にした。
- 回帰GradientBoostingのsubsampleを `randomState` で再現可能にした。
- 分類RandomForestのbootstrapと特徴量サンプリングを `randomState` で再現可能にした。
- 分類GradientBoostingのsubsampleを `randomState` で再現可能にした。
- 分類SVMのSGDサンプル順を `randomState` で再現可能にした。
- 各モデルの `getParams()` に `randomState` を含め、clone/finalize/export時にseedが落ちないようにした。
- 回帰・分類UIの乱数モデル定義に `randomState: 42` を明示した。
- `rg "Math.random" js/ml` で直接利用が残っていないことを確認した。
- 対象ファイルの `node --check` とES module import確認は成功。
- 同じseedで予測・確率・特徴量重要度が一致し、異なるseedで内部サンプリングが変わる簡易再現性テストは成功。
- `git diff --check` は成功。
