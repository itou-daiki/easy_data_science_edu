# Iteration 4: interpret_model + Learning Curve

## 目標
PyCaret トレース率 ~70% → ~85%
- interpret_model (Permutation Importance, PDP)
- plot_model (Learning Curve)
- ステップ: Setup → Compare → Tune → Interpret → Predict (5段)

## タスク

- [x] 1. model_selection.js に permutationImportance() 追加
- [x] 2. model_selection.js に learningCurve() 追加
- [x] 3. utils.js に renderPermutationImportance() 追加
- [x] 4. utils.js に renderPDP() 追加
- [x] 5. utils.js に renderLearningCurve() 追加
- [x] 6. regression.js に interpret_model セクション追加
- [x] 7. classification.js に interpret_model セクション追加
- [x] 8. ブラウザ検証 (Playwright)
- [x] 9. progress.md 更新

## 成功基準
- [x] Permutation Importance がプロットされること
- [x] PDP が各特徴量で描画されること
- [x] Learning Curve が描画されること
- [x] コンソールエラー: 0 (favicon除く)

## 結果
全検証パス。判定: 採用。
