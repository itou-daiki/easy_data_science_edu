# easyDataScience 自律改善プログレス

## ベースライン
- テスト: N/A (新規プロジェクト)
- ファイル数: 1 (README.md のみ)
- 機能数: 0

## Iteration 1 (2026-03-01)
- **目標**: コアインフラ構築 + ML アルゴリズム実装 + 分析モジュール
- **変更**: 27ファイル新規作成 (~8,300行)
- **成功基準**: ブラウザで動作し、回帰・分類のモデル比較ができること
- **結果**: 全検証パス
  - 回帰AutoML: 7モデル全学習成功 (Ridge R²=0.98 best)
  - 分類AutoML: 7モデル全学習成功 (NB F1=0.67 best)
  - EDA: 概要/分布/相関/欠損値 全タブ正常
  - 前処理: 欠損値/スケーリング/エンコーディング/外れ値 全タブ正常
  - コンソールエラー: 0 (favicon除く)
- **判定**: 採用

## Iteration 2 (2026-03-01)
- **目標**: RF回帰のmaxFeatures修正 + モデルパラメータ最適化
- **変更**:
  - regression.js: RF params にmaxFeatures:null追加, GBM nEstimators 50→100
  - classification.js: RF/GBM nEstimators 50→100
  - random_forest.js: maxFeatures の ?? 演算子を 'in' チェックに修正
- **結果**: RF R² 0.40→0.55-0.60改善, GBM R² 0.88改善
- **判定**: 採用 (RF は線形データのため低めだが正常動作)

## Iteration 3 (2026-03-01)
- **目標**: PyCaret トレース率向上 (~40% → ~70%) - CV接続 + tune_model + predict_model
- **変更**:
  - model_selection.js: `_cloneModel` を `getParams()` 対応に修正 (CV/GridSearchの前提条件)
  - regression.js: 全面改修 (330行 → 380行)
    - crossValidate() 接続: CV R² (mean±std) をテーブルに追加、CVスコアでソート
    - tune_model: PARAM_GRIDS + gridSearch() で Before/After 比較UI
    - predict_model: 特徴量入力フォーム + scaler.transform() 対応
    - ステップインジケータ: 3段→4段 (Setup/Compare/Tune/Predict)
    - CV Fold数セレクタ (3/5/10) 追加
  - classification.js: 同等の全面改修 (369行 → 430行)
    - CV F1 (mean±std) でソート
    - tune_model: 分類用PARAM_GRIDS
    - predict_model: クラスラベル + クラス別確率バー表示
- **成功基準**: CV/tune/predict が全てブラウザで動作すること
- **結果**: 全検証パス
  - 回帰AutoML: 7モデル全学習+CV成功 (Linear CV R²=0.9858 best)
  - 回帰tune_model: Ridge GridSearch完了 (alpha=0.01 best, CV R²=0.9858)
  - 回帰predict_model: 入力→予測値表示 正常 (80㎡/築10年→3810万円)
  - 分類AutoML: 7モデル全学習+CV成功 (KNN CV F1=0.5943 best)
  - 分類tune_model: KNN GridSearch完了 (nNeighbors=7 best, CV F1=0.6166)
  - 分類predict_model: クラスラベル+確率バー表示 正常
  - コンソールエラー: 0 (favicon除く)
- **PyCaret トレース率**: ~40% → ~65-70%
  - setup(): ✅ (target/features/test_size/CV設定)
  - compare_models(): ✅ (CV付きモデル比較+ソート)
  - tune_model(): ✅ (GridSearch CV)
  - evaluate_model(): ✅ (混同行列/ROC/残差/特徴量重要度)
  - predict_model(): ✅ (新規データ予測)
  - 未実装: blend/stack_models, SHAP, create_model個別
- **判定**: 採用
