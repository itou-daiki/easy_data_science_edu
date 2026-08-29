// ==========================================
// easyDataScience - Japanese / English UI
// ==========================================

const LANGUAGE_STORAGE_KEY = 'easyDataScience.language';
const SUPPORTED_LANGUAGES = new Set(['ja', 'en']);
const TRANSLATABLE_ATTRIBUTES = ['placeholder', 'title', 'aria-label'];
const LANGUAGE_CHANGE_EVENT = 'easyDataScience:languagechange';

const JA_TO_EN = new Map(Object.entries({
    // Application shell
    'easyDataScience - ブラウザ機械学習アプリ': 'easyDataScience - Browser-based Machine Learning',
    'easyDataScience を読み込み中...': 'Loading easyDataScience...',
    '機械学習エンジンの初期化をしています': 'Initializing the machine learning engine',
    '数秒で起動します': 'This should only take a few seconds',
    '起動エラー:': 'Startup error:',
    'デモデータを選択': 'Choose a demo dataset',
    '分析目的に合わせたデータセットを選択してください。': 'Choose a dataset that matches your analysis goal.',
    'クリックするとデータが読み込まれます。': 'Click a dataset to load it.',
    '回帰（数値予測）': 'Regression (numeric prediction)',
    '住宅価格データ (推奨)': 'Housing price data (recommended)',
    '150件・面積や築年数から価格を予測': '150 rows - predict price from floor area, building age, and more',
    'チップデータ': 'Restaurant tips data',
    '244件・食事の合計金額からチップを予測': '244 rows - predict tips from total bill and related variables',
    '分類（カテゴリ予測）': 'Classification (category prediction)',
    '顧客離反データ': 'Customer churn data',
    '200件・顧客の解約を予測（2クラス）': '200 rows - predict customer churn (2 classes)',
    'Iris（アヤメ）データ': 'Iris flower data',
    '150件・花の計測値から品種を分類（3クラス）': '150 rows - classify species from flower measurements (3 classes)',
    'ワイン品種データ': 'Wine cultivar data',
    '178件・化学分析13変数から品種を判別（3クラス）': '178 rows - classify cultivars from 13 chemical measurements (3 classes)',
    'ペンギンデータ': 'Penguin data',
    '144件・体の計測値からペンギンの種類を分類（3クラス）': '144 rows - classify penguin species from body measurements (3 classes)',
    '生成AI支援': 'Generative AI support',
    '生成AI支援の設定': 'Generative AI support settings',
    '分析結果ページでは、APIキーがなくても他の生成AIに貼り付けるためのAI用テキストをコピーできます。 Gemini APIキーを入力すると、ブラウザ内の補助パネルから直接「解釈を生成」し、結果について追加質問できます。 APIキーはこのブラウザタブ内にのみ保持されます。「解釈を生成」または追加質問を押した場合のみ、 データプレビュー先頭10件・要約統計量・データ構造・分析手法・主要指標・注意点が Google Gemini API へ送信されます。': 'On result pages, you can copy analysis text for another generative AI without an API key. Enter a Gemini API key to generate an interpretation and ask follow-up questions from the in-browser panel. The key is kept only in this browser tab. Data is sent to the Google Gemini API only when you generate an interpretation or ask a follow-up, and is limited to the first 10 preview rows, summary statistics, data structure, method, key metrics, and cautions.',
    'Gemini APIキー': 'Gemini API key',
    '使用モデル': 'Model',
    '指定モデルが利用できない場合は gemini-2.5-flash へ自動で切り替えます。': 'If the selected model is unavailable, the app automatically falls back to gemini-2.5-flash.',
    '有効化': 'Enable',
    '無効化': 'Disable',
    '有効': 'Enabled',
    '未設定': 'Not configured',
    '使い方マニュアル': 'User manual',
    '表示言語': 'Display language',
    'ブラウザ上で簡単に機械学習　- PyCaret ライクな自動 ML を手軽に -': 'Learn and run machine learning in your browser with a PyCaret-like AutoML workflow',
    'データファイルをアップロード': 'Upload a data file',
    'ここにファイルをドラッグ＆ドロップ': 'Drag and drop a file here',
    'ファイルを選択': 'Choose file',
    'または': 'or',
    'デモデータを試す': 'Try demo data',
    'データプレビュー': 'Data preview',
    '要約統計量': 'Summary statistics',
    'データ情報': 'Data information',
    'ファイル名': 'File name',
    '行数': 'Rows',
    '列数': 'Columns',
    '分析機能を選択': 'Choose an analysis',
    '目的に合ったカテゴリから分析手法を選んでください': 'Select an analysis method that matches your goal',
    '機械学習を学ぶ': 'Learn machine learning',
    '機械学習の基本概念をインタラクティブに学べる初学者向けガイドです': 'An interactive beginner guide to core machine learning concepts',
    '機械学習入門ガイド': 'Introduction to machine learning',
    '回帰・分類・評価指標・アンサンブル学習などを図解付きで学べます。データ不要です': 'Learn regression, classification, metrics, ensemble learning, and more with visual explanations. No data required.',
    'データの探索・前処理': 'Explore and prepare data',
    'データの傾向を把握し、機械学習の前準備を行います': 'Understand your data and prepare it for machine learning',
    '探索的データ分析 (EDA)': 'Exploratory data analysis (EDA)',
    '探索的データ分析（EDA）': 'Exploratory data analysis (EDA)',
    'データの分布・相関・欠損値を可視化し、全体像を把握します': 'Visualize distributions, correlations, and missing values to understand the dataset',
    'データの分布・相関・欠損値を可視化し、機械学習の前にデータの全体像を把握します。': 'Visualize distributions, correlations, and missing values to understand the dataset before modeling.',
    '数値変数が1つ以上必要です': 'Requires at least one numeric variable',
    'データ前処理': 'Data preprocessing',
    '欠損値補完・スケーリング・エンコーディングを自動で行います': 'Review missing-value handling, scaling, and encoding',
    '回帰（連続値の予測）': 'Regression (predict continuous values)',
    '数値の予測を行います。PyCaret のように複数モデルを自動比較できます': 'Predict numeric values and automatically compare multiple models in a PyCaret-like workflow',
    '回帰モデル比較 (AutoML)': 'Regression model comparison (AutoML)',
    '回帰モデル比較（AutoML）': 'Regression model comparison (AutoML)',
    '複数の回帰モデルを一括で学習・比較し、最適なモデルを見つけます': 'Train and compare multiple regression models to identify the strongest candidate',
    '数値変数が2つ以上必要です': 'Requires at least two numeric variables',
    '分類（カテゴリの予測）': 'Classification (predict categories)',
    'カテゴリ分類を行います。PyCaret のように複数モデルを自動比較できます': 'Predict categories and automatically compare multiple models in a PyCaret-like workflow',
    '分類モデル比較 (AutoML)': 'Classification model comparison (AutoML)',
    '分類モデル比較（AutoML）': 'Classification model comparison (AutoML)',
    '複数の分類モデルを一括で学習・比較し、最適なモデルを見つけます': 'Train and compare multiple classification models to identify the strongest candidate',
    '画像分類（ディープラーニング）': 'Image classification (deep learning)',
    'MobileNet 転移学習を用いて、ブラウザ上で画像分類モデルを構築します': 'Build an image classifier in the browser with MobileNet transfer learning',
    '画像分類 (MobileNet)': 'Image classification (MobileNet)',
    '画像分類（MobileNet）': 'Image classification (MobileNet)',
    '画像をアップロードしてカテゴリを学習・予測します。CSVデータ不要で利用できます': 'Upload images to train and predict categories. No CSV data required.',
    '音声分類（ディープラーニング）': 'Audio classification (deep learning)',
    'マイクで録音した音声をスペクトル特徴量に変換し、ブラウザ上で分類モデルを構築します': 'Convert recorded audio into spectral features and build a classifier in the browser',
    '音声分類 (Web Audio)': 'Audio classification (Web Audio)',
    '音声分類（Web Audio）': 'Audio classification (Web Audio)',
    'マイクで録音した音声からカテゴリを学習・予測します。CSVデータ不要で利用できます': 'Record audio to train and predict categories. No CSV data required.',
    '動画分析（Splyzaライク）': 'Video analysis (Splyza-inspired)',
    'ローカル動画を読み込み、コマ送り・描画・タグ付け・2動画比較・ポーズ推定をブラウザ上で行います': 'Load local videos for frame stepping, drawing, tagging, side-by-side comparison, and pose estimation',
    '動画分析ツール': 'Video analysis tool',
    'スポーツや行動動画の解析向け。再生速度切替・描画アノテーション・イベント記録・MoveNetによるポーズ推定が可能です': 'Analyze sports and behavior videos with playback controls, annotations, event logging, and MoveNet pose estimation.',
    '予測モード': 'Prediction mode',
    'easyDataScienceで作成したモデル（JSON）を読み込み、新しいデータで予測を実行します': 'Load a JSON model created in easyDataScience and predict new observations',
    'モデル読み込み＆予測': 'Load model and predict',
    'エクスポートしたJSONモデルファイルを読み込んで予測できます。CSVデータ不要で利用できます': 'Load an exported JSON model and make predictions. No CSV data required.',
    '機能選択に戻る': 'Back to analyses',
    'このアプリケーションについて': 'About this application',
    'easyDataScience - ブラウザ機械学習アプリ': 'easyDataScience - browser-based machine learning',
    'は、ブラウザ上で動作する無料の機械学習Webアプリケーションです。 PyCaret のように複数のモデルを自動比較し、最適なモデルを見つけることができます。 すべての処理はブラウザ内で完結し、データが外部に送信されることはありません。': 'is a free machine learning web application that runs in your browser. It automatically compares multiple models in a PyCaret-like workflow. All processing stays in the browser, so your data is not sent to an external server.',
    '主な特徴': 'Key features',
    '完全ブラウザベース:': 'Runs entirely in the browser:',
    'インストール不要、Webブラウザだけで利用可能': 'No installation required',
    'データプライバシー:': 'Data privacy:',
    'データはブラウザ内で処理され、外部サーバーに送信されません': 'Data is processed in your browser and is not sent to an external server',
    'AutoML機能:': 'AutoML workflow:',
    'PyCaret のように複数モデルを自動で学習・比較': 'Automatically train and compare multiple models in a PyCaret-like workflow',
    '回帰・分類対応:': 'Regression and classification:',
    '連続値予測とカテゴリ分類の両方に対応': 'Supports both continuous-value prediction and category classification',
    '画像・音声分類:': 'Image and audio classification:',
    'TensorFlow.js による転移学習で画像分類、Web Audio API で音声分類が可能': 'Use TensorFlow.js transfer learning for images and the Web Audio API for audio',
    '日本語対応:': 'Bilingual interface:',
    'UIと分析結果の解釈がすべて日本語': 'Switch the interface and result guidance between Japanese and English',
    '教育的:': 'Learning focused:',
    '機械学習手法の説明と結果の解釈を提供': 'Provides explanations of methods and guidance for interpreting results',
    '搭載アルゴリズム': 'Included algorithms',
    '線形回帰': 'Linear regression',
    'Ridge回帰': 'Ridge regression',
    'Lasso回帰': 'Lasso regression',
    'Ridge / Lasso 回帰': 'Ridge / Lasso regression',
    '決定木': 'Decision tree',
    'ランダムフォレスト': 'Random forest',
    'K近傍法': 'K-nearest neighbors',
    'K近傍法 (KNN)': 'K-nearest neighbors (KNN)',
    'サポートベクターマシン': 'Support vector machine',
    'ロジスティック回帰': 'Logistic regression',
    'ナイーブベイズ': 'Naive Bayes',
    '勾配ブースティング': 'Gradient boosting',
    'MobileNet 画像分類': 'MobileNet image classification',
    '音声スペクトル分類': 'Audio spectrum classification',
    '使い方・操作方法': 'How to use the app',
    '基本的な使い方': 'Basic workflow',
    'データをアップロード': 'Upload data',
    'Excel（.xlsx, .xls）またはCSVファイルを選択してください。ドラッグ＆ドロップにも対応しています。': 'Choose an Excel (.xlsx, .xls) or CSV file. Drag and drop is also supported.',
    '分析タスクを選択': 'Choose an analysis task',
    '回帰（連続値予測）または分類（カテゴリ予測）を選択します。EDAで事前にデータを把握することもできます。': 'Choose regression (continuous values) or classification (categories). You can inspect the data with EDA first.',
    '目的変数を選択': 'Choose the target variable',
    '選択してください': 'Choose a variable',
    '予測したい変数（目的変数）を選択すると、自動で前処理とモデル学習が行われます。': 'Choose the variable you want to predict, then run automated preprocessing and model training.',
    'モデル比較結果を確認': 'Review model comparison results',
    'PyCaret のように複数モデルの性能比較表が表示されます。最適なモデルの詳細評価やグラフも確認できます。': 'Compare model performance in a PyCaret-like table, then inspect detailed metrics and charts.',

    // Common analysis UI
    '概要': 'Overview',
    '分布': 'Distribution',
    '相関': 'Correlation',
    '欠損値': 'Missing values',
    '変数を選択:': 'Choose a variable:',
    'サンプル数': 'Samples',
    '特徴量数': 'Features',
    '数値変数': 'Numeric variables',
    'カテゴリ変数': 'Categorical variables',
    '欠損率': 'Missing rate',
    '重複行': 'Duplicate rows',
    '変数の型': 'Variable types',
    '変数名': 'Variable',
    '型': 'Type',
    'ユニーク数': 'Unique values',
    '欠損数': 'Missing',
    'サンプル値': 'Example values',
    '数値': 'Numeric',
    'カテゴリ': 'Categorical',
    'テキスト': 'Text',
    '頻度': 'Frequency',
    '平均': 'Mean',
    '標準偏差': 'Standard deviation',
    '中央値': 'Median',
    '歪度': 'Skewness',
    '尖度': 'Kurtosis',
    '相関行列': 'Correlation matrix',
    '欠損値はありません': 'No missing values',
    'すべての変数にデータが揃っています。': 'Every variable has complete data.',
    '変数': 'Variable',
    '件数': 'Count',
    '最小': 'Min',
    '最大': 'Max',
    '欠損': 'Missing',
    '状況': 'Status',
    '要注意': 'High concern',
    '注意': 'Caution',
    '軽微': 'Minor',
    '欠損値の状況': 'Missing-value profile',
    '欠損率 (%)': 'Missing rate (%)',
    '欠損値処理': 'Missing-value handling',
    'スケーリング': 'Scaling',
    'エンコーディング': 'Encoding',
    '外れ値検出': 'Outlier detection',
    '機械学習の前にデータを整えます。欠損値補完・スケーリング・エンコーディングの効果を確認できます。': 'Prepare data for machine learning and review the effects of imputation, scaling, and encoding.',
    'すべてのセルにデータが入っています。前処理は不要です。': 'Every cell contains data, so missing-value preprocessing is not needed.',
    '総欠損セル数': 'Total missing cells',
    '欠損のある変数': 'Variables with missing values',
    '全体欠損率': 'Overall missing rate',
    '補完方法の推奨': 'Recommended imputation',
    '推奨補完方法': 'Recommended method',
    '理由': 'Reason',
    '変数の除外': 'Exclude the variable',
    '欠損率50%超で信頼性が低い': 'More than 50% is missing, so reliability is low',
    '外れ値の影響を受けにくい': 'Less sensitive to outliers',
    'カテゴリ変数の標準的な方法': 'A standard approach for categorical variables',
    'AutoML 機能を使う際、欠損値は自動的に補完されます。': 'AutoML automatically imputes missing values during model training.',
    'スケーリング推奨': 'Scaling recommended',
    'スケールは概ね均一': 'Scales are broadly similar',
    '変数間でスケールに大きな差があります。KNN や SVM などの距離ベースのアルゴリズムではスケーリングが重要です。': 'Variable scales differ substantially. Scaling is important for distance-based algorithms such as KNN and SVM.',
    '変数間のスケール差は比較的小さいです。ただし、スケーリングは一般的に推奨されます。': 'Variable scales are relatively similar, although scaling is still generally recommended.',
    '各変数のスケール': 'Scale of each variable',
    '最小値': 'Minimum',
    '最大値': 'Maximum',
    '範囲': 'Range',
    'スケーリング手法の比較': 'Comparison of scaling methods',
    '手法': 'Method',
    '変換式': 'Transformation',
    '特徴': 'Result',
    '推奨場面': 'Recommended for',
    '平均0、標準偏差1': 'Mean 0 and standard deviation 1',
    '線形回帰、SVM、PCA': 'Linear regression, SVM, and PCA',
    '[0, 1]に変換': 'Maps values to [0, 1]',
    'ニューラルネット、KNN': 'Neural networks and KNN',
    'AutoML 機能では StandardScaler が自動適用されます。': 'AutoML applies StandardScaler automatically.',
    'カテゴリ変数がありません。エンコーディングは不要です。': 'There are no categorical variables, so encoding is not needed.',
    'カテゴリ変数のエンコーディング': 'Encoding categorical variables',
    '現在の型': 'Current type',
    '推奨エンコーディング': 'Recommended encoding',
    'そのまま使用可能（数値コード）': 'Can be used as-is (numeric code)',
    'Label Encoding (2値)': 'Label encoding (binary)',
    'Label Encoding（高カーディナリティ）': 'Label encoding (high cardinality)',
    '文字列': 'String',
    'AutoML 機能ではカテゴリ変数は自動的にエンコードされます。': 'AutoML automatically encodes categorical variables.',
    'Q1 (25%点)': 'Q1 (25th percentile)',
    'Q3 (75%点)': 'Q3 (75th percentile)',
    '外れ値の数': 'Number of outliers',
    '外れ値の判定基準: IQR法 (Q1 - 1.5*IQR 未満 または Q3 + 1.5*IQR 超)': 'Outlier rule: below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR',
    '処理方法:': 'Method:',
    '平均値で補完': 'Impute with the mean',
    '中央値で補完': 'Impute with the median',
    '最頻値で補完': 'Impute with the mode',
    '補完後のデータ': 'Data after imputation',
    '元のデータ': 'Original data',
    '標準化 (平均0, 標準偏差1)': 'Standardize (mean 0, standard deviation 1)',
    '正規化 (0〜1)': 'Normalize (0 to 1)',
    '変換前': 'Before',
    '変換後': 'After',
    'カテゴリ変数がありません。': 'There are no categorical variables.',
    '外れ値は検出されませんでした。': 'No outliers were detected.',
    'IQR法による外れ値検出': 'Outlier detection using the IQR method',
    '外れ値数': 'Outliers',
    '下限': 'Lower bound',
    '上限': 'Upper bound',
    '線形回帰': 'Linear regression',
    'セットアップ': 'Setup',
    'Step 1: セットアップ': 'Step 1: Setup',
    '目的変数（予測したい数値変数）:': 'Target variable (numeric value to predict):',
    '目的変数（予測したいカテゴリ変数）:': 'Target variable (category to predict):',
    '目的変数を選択': 'Choose the target variable',
    'テストデータの割合:': 'Test-data proportion:',
    '交差検証 Fold数:': 'Cross-validation folds:',
    '使用する特徴量（チェックを外すと除外）:': 'Features to use (clear a feature to exclude it):',
    'モデル比較を開始': 'Compare models',
    'モデルを学習・比較しています...': 'Training and comparing models...',
    'PyCaret のように複数の回帰モデルを一括学習・比較し、最適なモデルを見つけます。': 'Train and compare multiple regression models in a PyCaret-like workflow to identify the strongest candidate.',
    'PyCaret のように複数の分類モデルを一括学習・比較し、最適なモデルを見つけます。': 'Train and compare multiple classification models in a PyCaret-like workflow to identify the strongest candidate.',
    'Step 2: 前処理 (自動完了)': 'Step 2: Preprocessing (completed automatically)',
    '目的変数:': 'Target:',
    '目的変数': 'Target variable',
    '特徴量:': 'Features:',
    '欠損値処理:': 'Missing values:',
    'カテゴリ変数:': 'Categorical variables:',
    '外れ値除去:': 'Outlier removal:',
    '特徴量変換:': 'Feature transformation:',
    '多重共線性:': 'Multicollinearity:',
    'スケーリング:': 'Scaling:',
    'データ分割:': 'Data split:',
    '欠損値なし': 'No missing values',
    'エンコード不要': 'No encoding required',
    '外れ値なし': 'No outliers removed',
    '変換不要': 'No transformation required',
    '問題なし': 'No issues detected',
    'なし': 'None',
    'StandardScaler (平均0, 分散1)': 'StandardScaler (mean 0, variance 1)',
    '分析前の信頼性チェック': 'Pre-analysis reliability check',
    'モデル': 'Model',
    '操作': 'Action',
    '詳細': 'Details',
    '詳細を見る': 'View details',
    '最良': 'Best',
    'モデル比較結果': 'Model comparison results',
    '順位': 'Rank',
    '比較結果をCSVダウンロード': 'Download comparison as CSV',
    'CV R² 参考 (mean)': 'Reference CV R² (mean)',
    'CV R² 参考 (std)': 'Reference CV R² (std)',
    'CV R² 参考': 'Reference CV R²',
    'CV R² 参考 STD': 'Reference CV R² SD',
    'CV R² 参考 std': 'Reference CV R² SD',
    '前処理込みCVの標準偏差': 'Cross-validation standard deviation with fold-specific preprocessing',
    'テストデータ決定係数': 'Coefficient of determination on the test set',
    '特徴量数を考慮したR²': 'R² adjusted for the number of features',
    '平均絶対誤差': 'Mean absolute error',
    '二乗平均平方根誤差': 'Root mean squared error',
    '評価信頼性チェック': 'Evaluation reliability check',
    'モデル評価': 'Model evaluation',
    '評価指標': 'Evaluation metrics',
    '実測値 vs 予測値': 'Actual vs predicted',
    '実測値': 'Actual',
    '予測値': 'Predicted',
    '残差': 'Residual',
    '残差プロット': 'Residual plot',
    '特徴量重要度': 'Feature importance',
    '重要度': 'Importance',
    'データ点': 'Observations',
    '理想線 (y=x)': 'Ideal line (y=x)',
    'ゼロライン': 'Zero line',
    '解釈': 'Interpretation',
    '信頼性・妥当性チェック': 'Reliability and validity check',
    '性能の妥当性チェック': 'Performance validity check',
    'ベースライン': 'Baseline',
    '差分': 'Difference',
    '見方': 'How to read it',
    '指標': 'Metric',
    '訓練データ平均': 'Training-set mean',
    '改善': 'Improvement',
    '目的変数の分散をどれだけ説明できるか。高いほどよい。': 'Proportion of target variance explained; higher is better.',
    '平均的な予測誤差。目的変数と同じ単位で読める。': 'Average prediction error in the same units as the target.',
    '大きな外れ誤差を重く見る。MAEより大きく離れるほど外れ誤差に注意。': 'Weights large errors more heavily. A widening gap from MAE suggests large misses.',
    '残差平均': 'Mean residual',
    '0から大きく離れる場合、全体的な過大/過小予測の偏りがある。': 'A value far from zero indicates systematic over- or under-prediction.',
    '残差標準偏差': 'Residual standard deviation',
    '予測誤差のばらつき。小さいほど安定。': 'Spread of prediction errors; smaller values indicate greater stability.',
    '項目': 'Item',
    '値': 'Value',
    '方向/解釈': 'Direction / interpretation',
    '係数': 'Coefficient',
    '切片': 'Intercept',
    '前処理後スケールでの切片': 'Intercept on the preprocessed scale',
    '非ゼロ係数': 'Non-zero coefficients',
    '0でない係数数': 'Number of coefficients that are not zero',
    'Lassoの変数選択の目安': 'Indicator of feature selection by Lasso',
    'このモデルに線形係数はありません。分岐による誤差低下から得た特徴量重要度を見ます。相関した特徴量があると重要度が分散する点に注意してください。': 'This model has no linear coefficients. Review feature importance derived from reductions in split error; correlated features can divide the importance among themselves.',
    '大きいほど分岐や予測に使われた度合いが高い': 'Larger values indicate greater use in splits and predictions',
    '方向': 'Direction',
    '増えると予測値が上がる傾向': 'Higher values tend to increase the prediction',
    '増えると予測値が下がる傾向': 'Higher values tend to decrease the prediction',
    '絶対値': 'Absolute value',
    '重み': 'Weight',
    '特徴量': 'Feature',
    'クラス': 'Class',
    'クラス別指標': 'Per-class metrics',
    'Test件数': 'Test samples',
    'CV F1 参考 (mean)': 'Reference CV F1 (mean)',
    'CV F1 参考 (std)': 'Reference CV F1 (std)',
    'CV F1 参考': 'Reference CV F1',
    'CV F1 参考 STD': 'Reference CV F1 SD',
    'CV F1 参考 std': 'Reference CV F1 SD',
    '正解率': 'Accuracy on the test set',
    'クラス平均の適合率': 'Class-average precision',
    'クラス平均の再現率': 'Class-average recall',
    'クラス平均F1スコア': 'Class-average F1 score',
    'ROC曲線下面積': 'Area under the ROC curve',
    '小さいほど確率予測が良い': 'Lower values indicate better probability estimates',
    '学習曲線': 'Learning curve',
    '訓練スコア': 'Training score',
    '検証スコア': 'Validation score',
    '訓練サンプル数': 'Training samples',
    '特徴量値': 'Feature value',
    '低': 'Low',
    '高': 'High',
    'ROC曲線': 'ROC curve',
    '偽陽性率 (FPR)': 'False positive rate (FPR)',
    '真陽性率 (TPR)': 'True positive rate (TPR)',
    '混同行列': 'Confusion matrix',
    'ランダム': 'Random',
    '予測結果': 'Prediction result',
    '値を入力': 'Enter a value',
    '予測を実行': 'Run prediction',
    'モデルを作成': 'Create model',
    'create_model - モデル個別作成': 'create_model - Create an individual model',
    '特定のアルゴリズムとパラメータを指定してモデルを作成します。': 'Create a model with a selected algorithm and parameter values.',
    'アルゴリズム:': 'Algorithm:',
    'パラメータ:': 'Parameters:',
    'create_model を実行': 'Run create_model',
    'tune_model - ハイパーパラメータチューニング': 'tune_model - Hyperparameter tuning',
    'GridSearch CV（foldごとに前処理をfitする参考値）でパラメータを最適化します。': 'Optimize parameters with GridSearch CV using preprocessing fitted separately in each fold.',
    '探索範囲:': 'Search space:',
    'tune_model を実行': 'Run tune_model',
    'interpret_model - モデル解釈': 'interpret_model - Model interpretation',
    'Permutation Feature Importance、PDP、Learning Curve、SHAP でモデルを深く理解します。': 'Use permutation importance, PDP, learning curves, and SHAP to examine the model.',
    'interpret_model を実行': 'Run interpret_model',
    'blend_models - モデルアンサンブル': 'blend_models - Model ensemble',
    '上位モデルの予測値を平均して、より安定した予測を実現します。': 'Average predictions from top models to improve stability.',
    'ブレンドするモデル数:': 'Models to blend:',
    '上位3モデル': 'Top 3 models',
    '上位5モデル': 'Top 5 models',
    '全モデル (7)': 'All models (7)',
    'blend_models を実行': 'Run blend_models',
    'stack_models - スタッキングアンサンブル': 'stack_models - Stacking ensemble',
    '上位モデルの予測値を特徴量として、メタモデル（線形回帰）で最終予測を行います。ブレンド（平均）より高度なアンサンブル手法です。': 'Use top-model predictions as features for a linear meta-model. This is a more advanced ensemble than simple averaging.',
    'ベースモデル数:': 'Base models:',
    'stack_models を実行': 'Run stack_models',
    'finalize_model - モデル確定': 'finalize_model - Finalize model',
    '全データ（訓練+テスト）で再学習し、本番用モデルとして確定します。': 'Retrain on all available data and finalize the model for later use.',
    'finalize_model を実行': 'Run finalize_model',
    'predict_model - 新しいデータで予測': 'predict_model - Predict new data',
    '各特徴量の値を入力して予測を実行します。': 'Enter values for each feature and run a prediction.',
    'predict_model を実行': 'Run predict_model',
    'チューニング': 'Tune',
    'チューニングを実行': 'Run tuning',
    'モデル解釈': 'Model interpretation',
    '解釈を実行': 'Run interpretation',
    'モデルをブレンド': 'Blend models',
    'モデルをスタッキング': 'Stack models',
    'モデルを確定': 'Finalize model',
    '確定モデル': 'Finalized model',
    'モデルを保存': 'Save model',
    'CSVダウンロード': 'Download CSV',
    '結果をCSVダウンロード': 'Download results as CSV',
    '予測結果をCSVダウンロード': 'Download prediction as CSV',
    'エラー:': 'Error:',
    'モデル作成エラー:': 'Model creation error:',
    'チューニングエラー:': 'Tuning error:',
    '解釈エラー:': 'Interpretation error:',
    'ブレンドエラー:': 'Blending error:',
    'スタッキングエラー:': 'Stacking error:',
    '確定エラー:': 'Finalization error:',
    '予測エラー:': 'Prediction error:',
    'すべての特徴量に値を入力してください。': 'Enter a value for every feature.',
    '数値を正しく入力してください。': 'Enter valid numeric values.',
    'このモデルにはパラメータがありません。': 'This model has no configurable parameters.',
    'このモデルにはデフォルトパラメータがありません。': 'This model has no default parameters.',
    '分類モデル比較': 'Classification model comparison',
    '回帰モデル比較': 'Regression model comparison',
    'クラス数': 'Classes',
    'クラス分布': 'Class distribution',
    'クラス別確率:': 'Probability by class:',
    '予測クラス': 'Predicted class',
    '多数派クラス': 'Majority class',
    '多数派クラスベースラインよりMacro F1が改善しています。': 'Macro F1 is better than the majority-class baseline.',
    '多数派クラスだけを予測する方法とMacro F1が近いため、特徴量の情報量を見直してください。': 'Macro F1 is close to the majority-class baseline, so review whether the features contain enough predictive information.',
    'Accuracyだけでなく、クラス平均のPrecision/Recall/F1と多数派クラスだけを予測するベースラインとの差を確認します。 CVはfoldごとの訓練データだけで前処理をfitする参考値です。Test指標、混同行列、クラス別指標も重視してください。': 'Review class-average precision, recall, and F1 together with accuracy, and compare them with a majority-class baseline. CV is a reference estimate with preprocessing fitted only on each fold’s training data. Also prioritize the test metrics, confusion matrix, and per-class results.',
    '全体の正解率。クラス不均衡では過信しない。': 'Overall accuracy. Do not rely on it alone when classes are imbalanced.',
    '予測したクラスがどれだけ当たるかのクラス平均。': 'Class-average precision: how often predicted classes are correct.',
    '各クラスをどれだけ取りこぼさないかのクラス平均。': 'Class-average recall: how well each class is detected.',
    'PrecisionとRecallのバランス。比較の主指標。': 'Balances precision and recall; the primary comparison metric.',
    '取りこぼしに注意': 'Check missed cases',
    '誤検出に注意': 'Check false positives',
    '概ね安定': 'Generally stable',
    '大きいほど分類に使われた度合いが高い': 'Larger values indicate greater use by the classifier',
    '大きいほど予測に使われた度合いが高い': 'Larger values indicate greater use by the model',
    'モデル固有の見方: 係数': 'Model-specific interpretation: coefficients',
    'モデル固有の見方: 木構造・重要度': 'Model-specific interpretation: tree structure and importance',
    'モデル固有の見方: 近傍法': 'Model-specific interpretation: nearest neighbors',
    'モデル固有の見方: Naive Bayes': 'Model-specific interpretation: Naive Bayes',
    'モデル固有の見方: マージン重み': 'Model-specific interpretation: margin weights',
    'SVMに表示される値は分類境界のマージン重みです。確率はsigmoid近似で未キャリブレーションのため、確率値やLog Lossは参考扱いにしてください。': 'The displayed SVM values are margin weights for the decision boundary. Its probabilities use an uncalibrated sigmoid approximation, so treat probability values and log loss as reference only.',
    'ロジスティック回帰の係数は標準化後特徴量に対するlog-oddsの変化です。符号はクラス方向、絶対値は影響の目安です。': 'Logistic-regression coefficients represent changes in log odds for standardized features. The sign indicates class direction and the absolute value indicates relative influence.',
    'Label Encodingされたカテゴリ特徴量の係数はカテゴリ順序を意味しないため、強く解釈しないでください。': 'Do not strongly interpret coefficients for label-encoded categorical features because their numeric codes do not represent an ordered scale.',
    'SVMに表示される値は分類境界のマージン重みです。確率はsigmoid近似で未キャリブレーションのため、確率値やLog Lossは参考扱いにしてください。 Label Encodingされたカテゴリ特徴量の係数はカテゴリ順序を意味しないため、強く解釈しないでください。': 'The displayed SVM values are margin weights for the decision boundary. Its probabilities use an uncalibrated sigmoid approximation, so treat probability values and log loss as reference only. Do not strongly interpret weights for label-encoded categorical features because their numeric codes do not represent an ordered scale.',
    'ロジスティック回帰の係数は標準化後特徴量に対するlog-oddsの変化です。符号はクラス方向、絶対値は影響の目安です。 Label Encodingされたカテゴリ特徴量の係数はカテゴリ順序を意味しないため、強く解釈しないでください。': 'Logistic-regression coefficients represent changes in log odds for standardized features. The sign indicates class direction and the absolute value indicates relative influence. Do not strongly interpret coefficients for label-encoded categorical features because their numeric codes do not represent an ordered scale.',
    '表示方向:': 'Displayed direction:',
    'One-vs-Restの平均絶対値': 'Mean absolute value across one-vs-rest models',
    '最大深さ': 'Maximum depth',
    '木の数': 'Number of trees',
    '学習率': 'Learning rate',
    '特徴量サブサンプル': 'Feature subsampling',
    'サブサンプル': 'Subsampling',
    'このモデルに係数はありません。分岐の不純度低下などから得た特徴量重要度を確認します。相関した特徴量があると重要度は分散します。': 'This model has no coefficients. Review feature importance derived from reductions in split impurity; correlated features can divide the importance among themselves.',
    '近傍数 k': 'Number of neighbors (k)',
    '重み付け': 'Weighting',
    'Naive Bayesに係数はありません。各クラスの事前確率と、特徴量がクラスごとに正規分布に従うという仮定で分類します。 特徴量同士が強く相関する場合、仮定が崩れる点に注意してください。': 'Naive Bayes has no coefficients. It classifies using class priors and the assumption that features follow a normal distribution within each class. Strongly correlated features can violate its independence assumption.',
    '非常に高い分類精度': 'Very high classification performance',
    '良好な分類精度': 'Good classification performance',
    '中程度の分類精度': 'Moderate classification performance',
    '分類精度が低い': 'Low classification performance',
    'テストデータにやや弱い': 'weaker performance on the test set',
    '特定のモデルをパラメータ指定で作成・学習します。Compare の結果を踏まえ、詳細にモデルを構築できます。': 'Create and train a selected model with explicit parameters, using the comparison results to guide the configuration.',
    'モデルを選択:': 'Choose a model:',
    '上位モデルの予測確率を平均して、より安定した分類を実現します。': 'Average predicted probabilities from the top models to improve classification stability.',
    '上位モデルの予測をメタ学習器（ロジスティック回帰）の入力として使い、より高精度な予測を目指します。': 'Feed top-model predictions into a logistic-regression meta-learner to seek better classification performance.',
    'スタックするモデル数:': 'Models to stack:',
    'create_model - 個別モデル作成': 'create_model - Create an individual model',
    'の評価結果:': ' evaluation results:',
    'です。': '.',
    'です。特徴量の改善を検討してください。': '. Consider improving the feature set.',
    'です。データやモデルの見直しが必要です。': '. Review the data and model.',
    'です。実用的に十分な精度と言えます。': '. This may be practically useful, subject to the data and evaluation design.',
    'です。特徴量の追加や前処理の改善を検討してください。': '. Consider adding informative features or improving preprocessing.',
    'です。データの品質や特徴量の選択を見直してください。': '. Review data quality and feature selection.',
    '可能性があります。': '.',
    'CVよりテストF1が高い結果です。': 'The test F1 is higher than the cross-validation estimate.',
    'CVよりテストR²が高い結果です。': 'The test R² is higher than the cross-validation estimate.',

    // AI assistance
    '生成AI解釈補助': 'AI interpretation support',
    '分析結果': 'Analysis results',
    '読み取り対象': 'Information used',
    'AI用テキストをコピー': 'Copy text for AI',
    '解釈を生成': 'Generate interpretation',
    '質問': 'Ask',
    '折りたたむ': 'Collapse',
    '閉じる': 'Close',
    'コピーしました': 'Copied',
    '例: この結果をレポート用に短く書くと？': 'Example: How can I summarize these results in a report?',
    '分析に使う変数を選択し、結果を生成すると利用できます。': 'Choose variables and generate results to enable this feature.',
    '目的変数未選択': 'No target selected',
    '目的変数を選択し、モデル比較を実行してください。': 'Choose a target variable and run model comparison.',
    '目的変数を選択し、モデル比較を実行するとAI用テキストをコピーできます。': 'Choose a target variable and run model comparison to enable copying text for AI.',
    '結果の解釈': 'Interpreting the results',
    '非常に高い予測精度': 'Very high predictive accuracy',
    '良好な予測精度': 'Good predictive accuracy',
    '中程度の予測精度': 'Moderate predictive accuracy',
    '予測精度が低い': 'Low predictive accuracy',
    '安定したモデル': 'a stable model',
    'テストデータでやや性能低下': 'Some performance loss on the test set',
    '正則化なしのOLS。係数は前処理後、主に標準化後特徴量に対する値です。数値特徴量では符号が方向、絶対値が影響の大きさの目安です。 Label Encodingされたカテゴリ特徴量の係数はカテゴリ順序を意味しないため、符号や大小を強く解釈しないでください。': 'Ordinary least squares without regularization. Coefficients are measured on preprocessed, usually standardized features: the sign indicates direction and the absolute value indicates relative effect size. Do not strongly interpret coefficients for label-encoded categories because their numeric codes do not represent an ordered scale.',

    // Reliability and validity guidance
    '要確認': 'Needs review',
    '参考': 'Reference',
    '良好': 'Good',
    '分析信頼性チェック': 'Analysis reliability check',
    '重要な確認点があります': 'Important issues need review',
    '注意点があります': 'Some cautions need review',
    '大きな問題は見つかりません': 'No major issues were detected',
    '結果を読む前に、データと評価条件を確認します。': 'Review the data and evaluation design before interpreting results.',
    '重要な確認点があります。結果を読む前に、データと評価条件を確認します。': 'Important issues need review before interpreting the results.',
    '注意点があります。結果を読む前に、データと評価条件を確認します。': 'Review the listed cautions before interpreting the results.',
    '大きな問題は見つかりません。結果を読む前に、データと評価条件を確認します。': 'No major issues were detected, but review the data and evaluation design before interpreting the results.',
    'サンプル数が少ない': 'Small sample size',
    'サンプル数がやや少ない': 'Somewhat small sample size',
    'テスト件数が少ない': 'Small test set',
    '訓練件数が少ない': 'Small training set',
    '特徴量がありません': 'No features selected',
    '特徴量が多すぎる可能性': 'Possibly too many features',
    '特徴量数に注意': 'Check the feature count',
    '特徴量数': 'Feature count',
    '欠損率が高い列があります': 'Some columns have high missingness',
    '欠損率に注意': 'Check missingness',
    '欠損値があります': 'Missing values are present',
    '目的変数が空です': 'The target is empty',
    '回帰の目的変数に非数値があります': 'The regression target contains non-numeric values',
    '目的変数に変動がありません': 'The target has no variation',
    '目的変数の値の種類が少ない': 'The target has few unique values',
    '分類クラスが1種類だけです': 'Only one class is present',
    'クラス不均衡が非常に大きい': 'Severe class imbalance',
    'クラス不均衡に注意': 'Check class imbalance',
    '少数クラスの件数が少ない': 'The minority class has few observations',
    '訓練データ内の少数クラスが少ない': 'The minority class is small in the training split',
    'リークまたはID列の候補': 'Possible leakage or ID column',
    '変化しない特徴量があります': 'Some features are constant',
    '目的変数に近すぎる特徴量があります': 'Some features are suspiciously close to the target',
    '強い相関の特徴量があります': 'Some features are strongly correlated with the target',
    '目的変数とほぼ同じ特徴量があります': 'A feature is nearly identical to the target',
    'CV Fold数を自動調整しました': 'Cross-validation folds were adjusted automatically',
    'CV Fold数': 'Cross-validation folds',
    '外れ値除去の影響が大きい': 'Outlier removal has a large impact',
    '外れ値除去': 'Outlier removal',
    '多重共線性の自動除去': 'Automatic multicollinearity removal',
    'CVスコアのばらつきが大きい': 'Cross-validation scores vary substantially',
    'CVとTestの差が大きい': 'Large gap between CV and test performance',
    'CVとTestの差に注意': 'Check the gap between CV and test performance',
    'Test R²が負です': 'Test R² is negative',
    '単純ベースラインに十分勝っていません': 'The model does not clearly beat the simple baseline',
    'ベースライン比較': 'Baseline comparison',
    '多数派ベースラインに十分勝っていません': 'The model does not clearly beat the majority-class baseline',
    'CVはfold内で前処理をfit': 'Preprocessing is fitted within each CV fold',
    '比較・チューニングのCVでは、欠損補完・変換・エンコード・標準化をfoldごとの訓練データだけでfitします。最終判断では独立テスト指標もあわせて確認してください。': 'During comparison and tuning, imputation, transformation, encoding, and scaling are fitted only on the training portion of each fold. Use the independent test metrics for the final judgment as well.',
    'Test指標を主に見ます。CVはfoldごとの訓練データだけで前処理をfitする参考値ですが、最終判断では独立テスト指標も確認します。': 'Prioritize the test metrics. CV uses preprocessing fitted only on each fold’s training data and is a useful reference, but the independent test results should guide the final judgment.',
    '訓練平均ベースラインより誤差が小さくなっています。': 'The model has lower error than the training-mean baseline.',
    '訓練平均を予測するだけの方法と比べて誤差改善が小さいため、モデルの有用性を慎重に見てください。': 'The error improvement over predicting the training mean is small, so assess the model’s usefulness cautiously.',

    // Prediction, image, audio, video and learning modules
    'MobileNet の転移学習を使って、ブラウザ上で画像分類モデルを構築します。 データのアップロードから学習・予測まですべてブラウザ内で完結します。': 'Build an image classifier in the browser using MobileNet transfer learning. Uploading data, training, and prediction all stay in your browser.',
    'Step 1: 学習データの準備': 'Step 1: Prepare training data',
    '各クラスに最低2枚ずつ画像をアップロードしてください。クラスは2つ以上必要です。': 'Upload at least two images for each class. At least two classes are required.',
    '画像をドラッグ＆ドロップまたはクリック': 'Drag and drop images or click to choose files',
    'データ準備に戻る': 'Back to data preparation',
    'Step 2: モデル学習': 'Step 2: Train the model',
    'エポック数:': 'Epochs:',
    '検証データ割合:': 'Validation-data proportion:',
    '準備中...': 'Preparing...',
    '学習設定に戻る': 'Back to training settings',
    'Step 3: 評価結果': 'Step 3: Evaluation results',
    '精度曲線': 'Accuracy curve',
    'クラス別精度': 'Accuracy by class',
    'モデルの学習内容を理解する': 'Understand what the model learned',
    'モデルがどのような特徴を捉えて分類しているかを、3つの視点から分析します。': 'Examine the features used by the classifier from three perspectives.',
    '分析中...': 'Analyzing...',
    '特徴空間の可視化 (PCA)': 'Feature-space visualization (PCA)',
    'MobileNetが抽出した1280次元の特徴を2次元に圧縮したものです。 同じクラスの画像が近くに集まっているほど、モデルはクラスをよく区別できています。': 'This view compresses MobileNet’s 1,280-dimensional features into two dimensions. Tighter within-class clusters indicate that the model separates the classes more clearly.',
    '信頼度分析': 'Confidence analysis',
    'モデルが最も自信を持って正しく分類した画像と、判断に迷った画像を表示します。 誤分類された画像がある場合、何と間違えたかも確認できます。': 'Compare the most confidently correct images with uncertain examples, including what the model confused when it made an error.',
    '注目領域の可視化（オクルージョン感度）': 'Important-region visualization (occlusion sensitivity)',
    '画像の各領域を隠したときの予測変化を調べ、モデルが分類に重要視している部分を ヒートマップで表示します。': 'Hide parts of each image and measure the prediction change to visualize influential regions as a heatmap.',
    '赤い領域': 'Red regions',
    'ほど分類に重要な部分です。': ' indicate areas that matter more to the classification.',
    '予測に進む': 'Continue to prediction',
    '評価結果に戻る': 'Back to evaluation results',
    'Step 4: 新しい画像を分類': 'Step 4: Classify a new image',
    '分類したい画像をアップロードしてください。学習済みモデルで予測を行います。': 'Upload an image to classify with the trained model.',
    '画像をドラッグ＆ドロップ、またはクリックしてファイルを選択': 'Drag and drop an image or click to choose a file',
    'マイクまたはファイルから音声サンプルを収集し、TensorFlow.js で分類モデルを学習します。': 'Collect audio samples from a microphone or files and train a classifier with TensorFlow.js.',
    'Step 1: データ準備': 'Step 1: Prepare data',
    '各クラスごとに2つ以上の音声サンプルを録音またはアップロードしてください（3秒クリップ）。': 'Record or upload at least two audio samples for each class (three-second clips).',
    'クラスを追加:': 'Add a class:',
    'クラス名（例: 拍手、声）': 'Class name (for example, applause or voice)',
    '追加': 'Add',
    '録音': 'Record',
    'アップロード': 'Upload',
    '録音中... (3秒)': 'Recording... (3 seconds)',
    '学習を開始（Step 2へ）': 'Start training (continue to Step 2)',
    'Step 2: 特徴抽出と学習': 'Step 2: Extract features and train',
    'TensorFlow.js を読み込み中...': 'Loading TensorFlow.js...',
    'インターネット接続を確認してページを再読み込みしてください。': 'Check your internet connection and reload the page.',
    '特徴量を抽出しています...': 'Extracting features...',
    'モデルを学習中...': 'Training the model...',
    'Step 3: 評価': 'Step 3: Evaluation',
    '全体正解率': 'Overall accuracy',
    'サンプル数': 'Samples',
    'クラス別正解率': 'Accuracy by class',
    '予測へ進む（Step 4）': 'Continue to prediction (Step 4)',
    '損失 (Loss)': 'Loss',
    '正解率 (Accuracy)': 'Accuracy',
    '学習履歴': 'Training history',
    'Step 4: 予測': 'Step 4: Prediction',
    '新しい音声を録音またはアップロードして分類結果を確認しましょう。': 'Record or upload new audio to review the classification result.',
    '録音して予測': 'Record and predict',
    'ファイルで予測': 'Predict from a file',
    '信頼度:': 'Confidence:',
    '波形': 'Waveform',
    'クラス別確率': 'Probability by class',
    'クラス別予測確率': 'Predicted probability by class',
    '確率': 'Probability',
    'ローカル動画・画像をブラウザで読み込み、コマ送り再生・描画アノテーション・タグ記録・2メディア比較・MoveNet ポーズ推定をブラウザ内で完結します。 ファイルは外部に送信されません。Splyza のような行動・スポーツ動画解析を、教育用途で手軽に試せます。': 'Load local videos or images for frame stepping, drawing annotations, event tags, side-by-side comparison, and MoveNet pose estimation. Files stay in the browser, making sports and behavior analysis easy to explore for educational use.',
    '動画 (mp4 / webm / mov / ogg) または画像 (jpg / png / webp / gif / bmp) を読み込みます。': 'Load a video (mp4, webm, mov, or ogg) or image (jpg, png, webp, gif, or bmp).',
    'に切り替えると、左右並べて同時に分析できます。': ' to analyze two media files side by side.',
    'メディアA': 'Media A',
    'メディアB': 'Media B',
    '動画 / 画像をドラッグ＆ドロップ': 'Drag and drop a video or image',
    'Step 2: 解析・記録': 'Step 2: Analyze and record',
    '描画でフォームを示したり、タグでイベントを打刻したり、ポーズ推定で骨格を観察します。 ツールはタブで切り替えできます。': 'Use drawings to mark form, tags to timestamp events, and pose estimation to inspect body position. Switch tools with the tabs.',
    'タグ / イベント': 'Tags / events',
    'Step 1 でメディアを読み込むと描画を開始できます。': 'Load media in Step 1 to begin drawing.',
    'Step 3: サマリー・出力': 'Step 3: Summary and export',
    '記録した描画・タグ・ポーズサンプルをファイルに書き出します。動画ファイルとは別に保管できます。': 'Export recorded drawings, tags, and pose samples as separate files from the source video.',
    '描画件数': 'Drawings',
    'タグ件数': 'Tags',
    'ポーズサンプル': 'Pose samples',
    'タグを CSV に保存': 'Save tags as CSV',
    '描画 (メディアA) を JSON に保存': 'Save drawings for Media A as JSON',
    '関節角度 (メディアA) を CSV に保存': 'Save joint angles for Media A as CSV',
    'キーボードショートカット': 'Keyboard shortcuts',
    '再生/停止 ・': 'Play/pause:',
    'コマ送り ・': 'Step frame:',
    'タグ打刻 ・': 'Add tag:',
    'ツール対象切替': 'Switch target media',
    'easyDataScienceで作成したモデル（JSONファイル）を読み込んで、新しいデータで予測を実行できます': 'Load a model JSON created by easyDataScience and predict new observations.',
    'モデルJSONファイルをアップロード': 'Upload a model JSON file',
    'ドラッグ＆ドロップ、またはクリックしてファイルを選択': 'Drag and drop a file or click to choose one',
    'モデル読み込み完了': 'Model loaded',
    'データセット': 'Dataset',
    'エクスポート日': 'Exported',
    '不明': 'Unknown',
    '特徴量を入力して予測': 'Enter feature values and predict',
    '(数値)': '(numeric)',
    '(カテゴリ)': '(categorical)',
    'モデルファイルを読み込む': 'Load a model file',
    'JSONモデルファイルを選択': 'Choose a JSON model file',
    'モデル情報': 'Model information',
    'タスク': 'Task',
    '回帰': 'Regression',
    '分類': 'Classification',
    '画像分類': 'Image classification',
    '音声分類': 'Audio classification',
    'クラスを追加': 'Add class',
    'クラス名': 'Class name',
    '学習を実行': 'Train model',
    '学習を開始': 'Start training',
    '評価': 'Evaluation',
    '学習': 'Training',
    '予測': 'Prediction',
    'データ準備': 'Prepare data',
    'メディア準備': 'Prepare media',
    '解析・記録': 'Analyze and record',
    '出力': 'Export',
    'Step 1: メディアを開く': 'Step 1: Open media',
    '単一メディア': 'Single media',
    '2メディア比較': 'Compare two media files',
    '同期再生': 'Synchronized playback',
    '比較モード': 'Comparison mode',
    '動画': 'Video',
    '画像': 'Image',
    '未読込': 'Not loaded',
    '動画・画像を読み込む': 'Load video or image',
    'クリックまたはドラッグ＆ドロップ': 'Click or drag and drop',
    '再生': 'Play',
    '一時停止': 'Pause',
    '前のフレーム': 'Previous frame',
    '次のフレーム': 'Next frame',
    '描画': 'Draw',
    'タグ': 'Tags',
    'ポーズ推定': 'Pose estimation',
    'フリーハンド': 'Freehand',
    '直線': 'Line',
    '矢印': 'Arrow',
    '円': 'Circle',
    '四角': 'Rectangle',
    '元に戻す': 'Undo',
    '全消去': 'Clear all',
    'エクスポート': 'Export',
    'インポート': 'Import',
    'タグを追加': 'Add tag',
    'コメント': 'Comment',
    '削除': 'Delete',
    'ジャンプ': 'Jump',
    '現在フレームを推定': 'Estimate current frame',
    '連続推定': 'Continuous estimation',
    '停止': 'Stop',
    'CSVを書き出す': 'Export CSV',
    '機械学習入門': 'Machine learning fundamentals',
    '前へ': 'Previous',
    '次へ': 'Next',
    '目次': 'Contents',

    // Manual
    'easyDataScience - 使い方マニュアル': 'easyDataScience - User manual',
    'アプリへ戻る': 'Back to app',
    'easyDataScience 使い方マニュアル': 'easyDataScience user manual',
    'はじめに': 'Getting started',
    'できること': 'What you can do',
    '起動方法': 'Starting the app',
    '基本操作': 'Basic workflow',
    'データ準備': 'Preparing data',
    '表の作り方': 'Structuring a table',
    'データ型': 'Data types',
    'デモデータ': 'Demo datasets',
    ': 住宅価格データ。面積や築年数から価格を予測します。': ': housing data for predicting prices from floor area, building age, and related variables.',
    ': 食事の合計金額などからチップを予測します。': ': restaurant data for predicting tips from the total bill and related variables.',
    ': 顧客の解約を予測します。': ': customer data for predicting churn.',
    ': 花の計測値から品種を分類します。': ': flower measurements for classifying species.',
    ': 化学分析値からワイン品種を分類します。': ': chemical measurements for classifying wine cultivars.',
    ': 体の計測値から種類を分類します。': ': body measurements for classifying penguin species.',
    '表データ分析': 'Tabular data analysis',
    'EDA・前処理': 'EDA and preprocessing',
    '回帰AutoML': 'Regression AutoML',
    '分類AutoML': 'Classification AutoML',
    'AutoML手順': 'AutoML workflow',
    'モデル活用': 'Using trained models',
    '困ったとき': 'Troubleshooting',
    '結果の見方': 'Reading results',
    'チェックリスト': 'Checklist',
    'CSV / Excel の表データによる回帰・分類 AutoML、画像分類、音声分類、保存済みモデルによる予測までを、 初学者でも迷わず進められるように整理したマニュアルです。': 'This manual guides beginners through regression and classification AutoML with CSV or Excel data, image and audio classification, and predictions with saved models.',
    '1. できること': '1. What you can do',
    'easyDataScience は、PyCaret のような機械学習ワークフローをブラウザ上で体験できる教育用アプリです。': 'easyDataScience is an educational browser app for exploring a PyCaret-like machine learning workflow.',
    '学ぶ': 'Learn',
    '機械学習入門ガイドで、回帰・分類・評価指標・前処理を図解付きで確認できます。': 'Use visual lessons to learn regression, classification, evaluation metrics, and preprocessing.',
    '表データAutoML': 'Tabular AutoML',
    'CSV / Excel を読み込み、回帰・分類モデルを比較、チューニング、解釈、予測できます。': 'Load CSV or Excel data to compare, tune, interpret, and use regression or classification models.',
    '画像・音声分類': 'Image and audio classification',
    'MobileNet 転移学習による画像分類と、Web Audio API による音声分類を体験できます。': 'Explore image classification with MobileNet transfer learning and audio classification with the Web Audio API.',
    'ファイナライズしたモデルを JSON で保存し、予測モードで再利用できます。': 'Save a finalized model as JSON and reuse it in prediction mode.',
    'プライバシー:': 'Privacy:',
    'データ処理はブラウザ内で行われ、アップロードしたデータを外部サーバーへ送信しません。': 'Data is processed in your browser. Uploaded data is not sent to an external server.',
    '2. 起動方法': '2. Starting the app',
    '公開ページで使う': 'Use the hosted app',
    '表示言語を切り替える': 'Switch display language',
    '公開されている easyDataScience のページをブラウザで開きます。インストールは不要です。': 'Open the published easyDataScience page in a browser. No installation is required.',
    'ローカルで使う': 'Run locally',
    'リポジトリを手元に置いて使う場合は、プロジェクトルートでローカルサーバーを起動します。': 'To run a local copy, start a local server from the project root.',
    'ブラウザで': 'Open',
    'を開きます。': 'in your browser.',
    '注意:': 'Note:',
    'を直接開くと、ES Modules やデモデータの読み込みが失敗することがあります。': 'directly may prevent ES modules or demo datasets from loading.',
    '3. 基本操作': '3. Basic workflow',
    'データをアップロードする、またはデモデータを読み込む': 'Upload data or load a demo dataset',
    'データプレビューと要約統計量を確認する': 'Review the data preview and summary statistics',
    '必要に応じて EDA で分布、相関、欠損値を確認する': 'Use EDA to inspect distributions, correlations, and missing values',
    '目的変数の種類に合わせて「回帰」または「分類」を選ぶ': 'Choose regression or classification based on the target variable',
    '目的変数、テストデータ割合、交差検証 Fold 数、特徴量を設定する': 'Set the target, test-data proportion, cross-validation folds, and features',
    'モデル比較を開始する': 'Start model comparison',
    '比較表、評価指標、グラフ、解釈結果を見る': 'Review the comparison table, metrics, charts, and interpretation',
    '必要に応じてチューニング、ブレンド、スタッキング、ファイナライズを行う': 'Tune, blend, stack, or finalize models as needed',
    'predict_model または予測モードで新しいデータを予測する': 'Predict new data with predict_model or prediction mode',
    '4. 正しいデータ表の作り方': '4. How to structure a data table',
    '機械学習では、データの形が結果の信頼性に直結します。読み込み前に、機械が解釈しやすい表へ整えてください。': 'Table structure directly affects the reliability of machine learning results. Prepare a machine-readable table before loading it.',
    '良い例': 'Good example',
    '1行目は列名だけにする': 'Use only column names in the first row',
    '2行目以降にデータを入れる': 'Place observations from the second row onward',
    '1件のデータを1行にする': 'Use one row per observation',
    'セル結合を使わない': 'Do not merge cells',
    '数値列に単位や「約」などの文字を混ぜない': 'Do not mix units or words such as “about” into numeric columns',
    '目的変数を1つ決めておく': 'Choose one target variable',
    '5. データ型の考え方': '5. Understanding data types',
    '価格、売上、年齢、面積、学習時間、テスト点数など。数値を予測したい場合は回帰を使います。': 'Examples include price, sales, age, area, study time, and test scores. Use regression to predict a number.',
    '解約する / しない、品種、クラス、合格 / 不合格など。カテゴリを予測したい場合は分類を使います。': 'Examples include churn/no churn, species, class, and pass/fail. Use classification to predict a category.',
    '画像・音声': 'Images and audio',
    '画像分類と音声分類では、CSVではなく画像ファイルや音声サンプルを直接使います。': 'Image and audio classification use image files or audio samples directly instead of CSV data.',
    '6. デモデータ': '6. Demo datasets',
    '手元にデータがない場合は、トップ画面の「デモデータを試す」から読み込めます。': 'If you do not have data, choose “Try demo data” on the home screen.',
    '回帰向け': 'For regression',
    '住宅価格データ。面積や築年数から価格を予測します。': 'Housing data for predicting price from floor area, building age, and other variables.',
    '食事の合計金額などからチップを予測します。': 'Predict tips from the total bill and related variables.',
    '分類向け': 'For classification',
    '顧客の解約を予測します。': 'Predict customer churn.',
    '花の計測値から品種を分類します。': 'Classify species from flower measurements.',
    '化学分析値からワイン品種を分類します。': 'Classify wine cultivars from chemical measurements.',
    '体の計測値から種類を分類します。': 'Classify species from body measurements.',
    '7. EDA・前処理': '7. EDA and preprocessing',
    'データの分布、相関、欠損値を可視化します。回帰や分類に進む前に、外れ値、欠損、目的変数との関係を確認します。': 'Visualize distributions, correlations, and missing values. Before regression or classification, check outliers, missingness, and relationships with the target.',
    '数値変数 1つ以上': 'At least 1 numeric variable',
    '欠損値補完、スケーリング、エンコーディングなど、機械学習に必要な前処理を確認します。AutoML 画面では学習時にも自動前処理が実行されます。': 'Review missing-value imputation, scaling, encoding, and other preprocessing. AutoML also applies preprocessing during training.',
    '8. 回帰モデル比較（AutoML）': '8. Regression model comparison (AutoML)',
    '住宅価格、売上、テスト点数、チップ金額など、数値を予測したいときに使います。': 'Use regression to predict numeric outcomes such as prices, sales, test scores, or tips.',
    '目的変数: 数値': 'Target: numeric',
    '数値変数 2つ以上': 'At least 2 numeric variables',
    '操作手順': 'Steps',
    '目的変数を選ぶ': 'Choose the target variable',
    'テストデータ割合を選ぶ': 'Choose the test-data proportion',
    '交差検証 Fold 数を選ぶ': 'Choose the number of cross-validation folds',
    '使用する特徴量を確認する': 'Review the selected features',
    '「モデル比較を開始」を押す': 'Select “Compare models”',
    'R2、MAE、RMSE、CV R2 を確認する': 'Review R2, MAE, RMSE, and CV R2',
    '搭載モデル': 'Included models',
    '線形回帰、Ridge回帰、Lasso回帰、決定木、ランダムフォレスト、K近傍法、勾配ブースティング。': 'Linear regression, Ridge, Lasso, decision tree, random forest, K-nearest neighbors, and gradient boosting.',
    '9. 分類モデル比較（AutoML）': '9. Classification model comparison (AutoML)',
    '解約する / しない、品種、合格 / 不合格など、カテゴリを予測したいときに使います。': 'Use classification to predict categories such as churn/no churn, species, or pass/fail.',
    '目的変数: カテゴリ': 'Target: categorical',
    '「モデル比較を開始」を押す': 'Select “Compare models”',
    'F1、Accuracy、Precision、Recall、AUC、CV F1 を確認する': 'Review F1, accuracy, precision, recall, AUC, and CV F1',
    'ロジスティック回帰、決定木、ランダムフォレスト、K近傍法、ナイーブベイズ、SVM、勾配ブースティング。': 'Logistic regression, decision tree, random forest, K-nearest neighbors, Naive Bayes, SVM, and gradient boosting.',
    'クラス不均衡がある場合は、Accuracy だけでなく F1、Precision、Recall、混同行列を確認してください。': 'When classes are imbalanced, review F1, precision, recall, and the confusion matrix instead of relying on accuracy alone.',
    '10. AutoML ワークフロー': '10. AutoML workflow',
    '目的変数、テスト割合、Fold数、特徴量を決めます。': 'Choose the target, test proportion, folds, and features.',
    'データ分割、欠損処理、エンコーディング、スケーリングを行います。': 'Split the data and apply missing-value handling, encoding, and scaling.',
    '複数モデルを一括学習し、テストデータと交差検証で比較します。': 'Train multiple models and compare test and cross-validation performance.',
    'ハイパーパラメータを探索します。': 'Search hyperparameters.',
    '特徴量重要度、PDP、学習曲線、SHAP 系の可視化を確認します。': 'Review feature importance, PDP, learning curves, and SHAP-based visualizations.',
    '最終モデルを確定し、新しいデータを予測します。': 'Finalize a model and predict new data.',
    '11. 生成AI支援': '11. Generative AI support',
    '分析結果ページのフローティングパネルから、結果の読み取りを補助するAI用テキストを作成できます。': 'Use the floating panel on result pages to create context that helps an AI explain the analysis.',
    'APIキーなしで使う': 'Use without an API key',
    '「AI用テキストをコピー」を押すと、データプレビュー、要約統計量、分析手法、主要指標、注意点を含む文脈をコピーできます。ChatGPT、Gemini、Claudeなどに貼り付けて利用してください。': 'Select “Copy text for AI” to copy the data preview, summary statistics, method, key metrics, and cautions for use in ChatGPT, Gemini, Claude, or another AI.',
    'Geminiで直接生成する': 'Generate directly with Gemini',
    'ページ上部の「生成AI支援」からGemini APIキーを入力すると、「解釈を生成」と追加質問を使ってブラウザ内で解釈補助を生成できます。': 'Enter a Gemini API key under “Generative AI support” to generate an interpretation and ask follow-up questions in the browser.',
    '送信範囲:': 'Data sent:',
    'Google Gemini APIへ送信されるのは「解釈を生成」または追加質問を押した場合だけです。アップロードファイル全体ではなく、先頭10件のプレビュー、要約統計量、分析手法、主要指標、注意点などの要約文脈を送信します。': 'Data is sent to the Google Gemini API only when you generate an interpretation or ask a follow-up. The app sends a summarized context, including the first 10 rows, summary statistics, method, key metrics, and cautions, rather than the entire uploaded file.',
    '回帰・分類では、目的変数を選択してモデル比較結果が表示されるまでコピーできません': 'For regression and classification, copying is disabled until you choose a target and generate model-comparison results',
    'APIキー有効時は、生成した解釈や分析文脈をもとに「この結果をレポート用に短く書くと？」のような追加質問ができます': 'With an API key, you can ask follow-up questions such as “How can I summarize these results in a report?”',
    'コピー内容には、信頼性と妥当性チェック、解釈の注意点、レポート例を求める指示が含まれます': 'The copied prompt requests reliability and validity checks, interpretation cautions, and report examples',
    '指定したGeminiモデルが利用できない場合は、既定モデルへ自動で切り替えます': 'If the selected Gemini model is unavailable, the app automatically uses the default model',
    'APIキーはブラウザタブ内にのみ保持され、ページを閉じると消えます': 'The API key is kept only in the current browser tab and is cleared when the page closes',
    '12. モデル保存と予測モード': '12. Saving models and prediction mode',
    '回帰・分類でファイナライズしたモデルは JSON ファイルとして保存できます。保存済みモデルはトップ画面の「モデル読み込み＆予測」から利用します。': 'Finalized regression and classification models can be saved as JSON and used from “Load model and predict” on the home screen.',
    'JSON モデルファイルをアップロードする': 'Upload a JSON model file',
    'モデル名、タスク、目的変数、特徴量数を確認する': 'Review the model name, task, target, and number of features',
    '表示された入力フォームに新しい特徴量を入力する': 'Enter new feature values in the form',
    '「予測を実行」を押す': 'Select “Run prediction”',
    '13. 画像分類（MobileNet）': '13. Image classification (MobileNet)',
    'CSVデータを使わず、画像ファイルを直接アップロードして分類モデルを作ります。': 'Build a classifier by uploading image files directly, without CSV data.',
    '2クラス以上': 'At least 2 classes',
    '各クラス最低2枚': 'At least 2 images per class',
    'クラスを追加する': 'Add classes',
    '各クラスに画像をアップロードする': 'Upload images for each class',
    'エポック数と検証データ割合を選ぶ': 'Choose the epochs and validation proportion',
    '学習を実行する': 'Train the model',
    '評価結果、学習曲線、クラス別精度、混同行列を確認する': 'Review evaluation results, learning curves, per-class accuracy, and the confusion matrix',
    '新しい画像をアップロードして分類する': 'Upload a new image for classification',
    '14. 音声分類（Web Audio）': '14. Audio classification (Web Audio)',
    'マイク録音または音声ファイルからサンプルを集め、スペクトル特徴量を使って分類モデルを学習します。': 'Collect samples from the microphone or audio files and train a classifier with spectral features.',
    '各クラス2サンプル以上': 'At least 2 samples per class',
    '3秒程度のクリップ': 'Clips of about 3 seconds',
    'クラス名を入力して追加する': 'Enter and add class names',
    '各クラスで録音または音声サンプルをアップロードする': 'Record or upload audio samples for each class',
    '学習を開始する': 'Start training',
    '評価結果、クラス別精度、混同行列を確認する': 'Review evaluation results, per-class accuracy, and the confusion matrix',
    '新しい音声を録音またはアップロードして分類する': 'Record or upload new audio for classification',
    'マイク利用時はブラウザのマイク許可が必要です。周囲の雑音が多いと精度が落ちます。': 'Microphone access requires browser permission. Background noise can reduce accuracy.',
    '15. 結果の見方': '15. Reading results',
    'テストスコアと交差検証スコアを両方見る': 'Review both test and cross-validation scores',
    '回帰では R2 だけでなく MAE と RMSE も確認する': 'For regression, review MAE and RMSE as well as R2',
    '分類では Accuracy だけでなく F1、Precision、Recall、混同行列を確認する': 'For classification, review F1, precision, recall, and the confusion matrix as well as accuracy',
    '特徴量重要度は因果関係の証明ではなく、モデル解釈のヒントとして使う': 'Treat feature importance as a model-interpretation clue, not evidence of causation',
    '欠損、外れ値、カテゴリ表記ゆれ、クラス不均衡を確認する': 'Check missing values, outliers, inconsistent category labels, and class imbalance',
    '16. よくあるトラブル': '16. Troubleshooting',
    'ファイルが読み込めない': 'The file will not load',
    '拡張子、1行目の列名、空行、セル結合、装飾用の見出し行を確認してください。': 'Check the extension, first-row column names, blank rows, merged cells, and decorative heading rows.',
    '分析カードが押せない': 'An analysis card is disabled',
    'カードごとに必要なデータ条件があります。回帰は数値変数2つ以上、分類は数値変数1つ以上が必要です。': 'Each card has data requirements. Regression needs at least two numeric variables; classification needs at least one.',
    '数値列が特徴量として使えない': 'A numeric column cannot be used as a feature',
    '、': ',',
    '、全角数字などが混ざっていないか確認してください。': ', or full-width digits are mixed into the column.',
    '画像・音声分類がうまく学習しない': 'Image or audio training performs poorly',
    '各クラスのサンプル数、クラス間の偏り、撮影・録音条件、背景や雑音の違いを確認してください。': 'Check the sample count per class, class balance, capture conditions, backgrounds, and noise.',
    '17. 分析前チェックリスト': '17. Pre-analysis checklist',
    '目的変数を決めた': 'I chose the target variable',
    '目的変数が数値なら回帰、カテゴリなら分類を選ぶと決めた': 'I will use regression for a numeric target or classification for a categorical target',
    '1行目が列名だけになっている': 'The first row contains only column names',
    'セル結合や装飾用の見出し行がない': 'There are no merged cells or decorative heading rows',
    '数値列に単位や文字が混ざっていない': 'Numeric columns do not contain units or text',
    '欠損値と外れ値を確認した': 'I checked missing values and outliers',
    'クラス不均衡を確認した': 'I checked class imbalance',
    'モデル保存が必要な場合はファイナライズ後に JSON をダウンロードする': 'If I need to save the model, I will download the JSON after finalization',
    '最終更新: 2026-08-29': 'Last updated: 2026-08-29'
}).map(([key, value]) => [key.replace(/\s+/g, ' ').trim(), value]));

const SCOPED_JA_TO_EN = new Map([
    ['learning', new Map(Object.entries({
        '機械学習 学習ガイド': 'Machine learning guide',
        'インタラクティブに機械学習の基礎を学びましょう。各タブをクリックして進めてください。': 'Learn the foundations of machine learning interactively. Use the tabs to move through the guide.',
        '機械学習とは': 'What is machine learning?',
        '回帰と分類': 'Regression and classification',
        'モデルの評価': 'Model evaluation',
        '特徴量と前処理': 'Features and preprocessing',
        'アンサンブル学習': 'Ensemble learning',
        'このアプリの使い方': 'Using this app',
        '機械学習とは？': 'What is machine learning?',
        '機械学習とは、': 'Machine learning is a technology that ',
        'データからパターンを自動的に学習し、未知のデータに対して予測や判断を行う': 'automatically learns patterns from data and makes predictions or decisions for unseen data',
        '技術です。 人間がルールを一つ一つプログラムするのではなく、コンピュータがデータの中にある規則性を見つけ出します。': '. Instead of having a person program every rule, the computer discovers regularities in the data.',
        'データ': 'Data',
        '学習アルゴリズム': 'Learning algorithm',
        '機械学習の種類': 'Types of machine learning',
        '教師あり学習': 'Supervised learning',
        '正解ラベル付きデータで学習します。': 'Learns from data with known target labels.',
        ': 連続値を予測（例: 住宅価格）': ': predicts continuous values (for example, house prices)',
        ': カテゴリを予測（例: スパム判定）': ': predicts categories (for example, spam detection)',
        '教師なし学習': 'Unsupervised learning',
        '正解ラベルなしでデータの構造を発見します。': 'Discovers structure in data without target labels.',
        'クラスタリング': 'Clustering',
        ': データのグループ分け': ': groups similar observations',
        '次元削減': 'Dimensionality reduction',
        ': 情報を圧縮': ': compresses information',
        '強化学習': 'Reinforcement learning',
        '試行錯誤で最適な行動を学習します。': 'Learns effective actions through trial and error.',
        'ゲームAI': 'Game AI',
        'ロボット制御': 'Robot control',
        '自動運転': 'Autonomous driving',
        'などに使用': 'Common applications',
        'インタラクティブ体験: パターンを見つけよう': 'Interactive exercise: find the pattern',
        '下のボタンをクリックすると、ランダムなデータが生成されます。データにどのようなパターンがあるか観察してみましょう。': 'Generate random data with the buttons below and observe the pattern in each dataset.',
        '直線パターンを生成': 'Generate a linear pattern',
        'クラスターパターンを生成': 'Generate a cluster pattern',
        '学習した直線': 'Learned line',
        '教師あり学習 (回帰)': 'Supervised learning (regression)',
        '特徴量 X': 'Feature X',
        '目的変数 Y': 'Target Y',
        'クラスター A': 'Cluster A',
        'クラスター B': 'Cluster B',
        'クラスター C': 'Cluster C',
        '教師なし学習 (クラスタリング)': 'Unsupervised learning (clustering)',
        'ポイント': 'Key point',
        '機械学習は「データの中に隠れたパターンを自動で発見する技術」です。 予測したい値（目的変数）の種類に応じて、回帰・分類・クラスタリングなどの手法を使い分けます。': 'Machine learning automatically discovers patterns hidden in data. Choose regression, classification, clustering, or another method according to the kind of target or structure you want to study.',

        '回帰と分類の違い': 'Regression vs classification',
        '教師あり学習は大きく「回帰」と「分類」に分かれます。 目的変数（予測したい値）が': 'Supervised learning is broadly divided into regression and classification. Use regression when the target is continuous and classification when it is categorical.',
        '連続値なら回帰': 'Regression for continuous values',
        'カテゴリなら分類': 'Classification for categories',
        '連続的な数値を予測します。': 'Predicts a continuous numeric value.',
        '例': 'Examples',
        ': 住宅価格、気温、売上高': ': house price, temperature, sales',
        ': 実数値 (例: 3500万円)': ': a numeric value (for example, 35 million yen)',
        '離散的なカテゴリを予測します。': 'Predicts a discrete category.',
        ': スパム/非スパム、犬/猫、病気の有無': ': spam/not spam, dog/cat, disease/no disease',
        ': クラスラベル (例: 陽性)': ': a class label (for example, positive)',
        'インタラクティブ: 回帰 vs 分類を切り替えよう': 'Interactive exercise: switch between regression and classification',
        '同じデータセットでも、目的に応じて回帰と分類を切り替えられます。ボタンで表示を切り替えてみましょう。': 'The same dataset can support regression or classification depending on the question. Use the buttons to switch views.',
        '回帰モード': 'Regression mode',
        '分類モード': 'Classification mode',
        'ノイズ量:': 'Noise level:',
        '回帰直線': 'Regression line',
        '回帰: 連続値を予測': 'Regression: predict a continuous value',
        '決定境界': 'Decision boundary',
        '分類: カテゴリを予測': 'Classification: predict a category',
        '特徴量 Y': 'Feature Y',
        '回帰は「どれくらい？」、分類は「どちらに属する？」を予測します。 適切な問題設定が良いモデルの第一歩です。': 'Regression predicts how much; classification predicts which group. Framing the problem correctly is the first step toward a useful model.',

        '訓練データとテストデータの分割': 'Splitting training and test data',
        'モデルの本当の実力を測るために、データを': 'To estimate performance on unseen data, split the dataset into ',
        '訓練用': 'training data',
        'と': ' and ',
        'テスト用': 'test data',
        'に分けます。 訓練データで学習し、テストデータで評価します。': '. Fit the model on the training data and evaluate it on the test data.',
        '訓練データの割合:': 'Training-data proportion:',
        'データ分割': 'Data split',
        'サンプル番号': 'Sample index',
        '回帰の評価指標': 'Regression metrics',
        'スライダーで予測の精度を変化させ、評価指標がどう変わるか観察しましょう。': 'Change prediction quality with the slider and observe how the metrics respond.',
        '予測のノイズ:': 'Prediction noise:',
        '予測 vs 実測': 'Predicted vs actual',
        'R² (決定係数)': 'R² (coefficient of determination)',
        '1に近いほど良い': 'Closer to 1 is better',
        'MAE (平均絶対誤差)': 'MAE (mean absolute error)',
        '0に近いほど良い': 'Closer to 0 is better',
        'RMSE (二乗平均平方根誤差)': 'RMSE (root mean squared error)',
        '分類の評価指標: 混同行列': 'Classification evaluation: confusion matrix',
        '分類の評価は': 'A ',
        'が基本です。予測と実際の組み合わせを4つのセルで整理します。': ' is a basic classification tool that organizes predicted and actual classes into four cells.',
        '分類の精度:': 'Classification accuracy:',
        '予測: 陽性': 'Predicted: positive',
        '予測: 陰性': 'Predicted: negative',
        '実際: 陽性': 'Actual: positive',
        '実際: 陰性': 'Actual: negative',
        'Accuracy (正解率)': 'Accuracy',
        'Precision (適合率)': 'Precision',
        '陽性と予測した中の正解率': 'Fraction of positive predictions that are correct',
        'Recall (再現率)': 'Recall',
        '実際の陽性を正しく検出した率': 'Fraction of actual positives detected',
        '過学習と未学習': 'Overfitting and underfitting',
        'モデルの複雑さと汎化性能の関係を学習曲線で見てみましょう。': 'Use learning curves to examine the relationship between model complexity and generalization.',
        '未学習 (Underfitting)': 'Underfitting',
        '適切なモデル': 'Appropriate model',
        '過学習 (Overfitting)': 'Overfitting',
        'テストスコア': 'Test score',
        '学習曲線 - 適切': 'Learning curve - appropriate fit',
        '学習曲線 - 未学習': 'Learning curve - underfitting',
        '学習曲線 - 過学習': 'Learning curve - overfitting',
        'スコア': 'Score',
        'モデルの評価は「未知のデータに対する予測精度」が重要です。訓練データだけでの評価は過学習を見逃します。 複数の指標を総合的に判断し、過学習と未学習のバランスを取りましょう。': 'Evaluate performance on unseen data. Training performance alone can hide overfitting. Consider multiple metrics and balance underfitting against overfitting.',

        '特徴量とは？': 'What is a feature?',
        'とは、モデルが予測に使用するデータの各属性のことです。 例えば住宅価格予測なら「面積」「築年数」「駅からの距離」などが特徴量です。': ' is an attribute the model uses for prediction. For house-price prediction, features might include floor area, building age, and distance to a station.',
        '面積': 'Floor area',
        '築年数': 'Building age',
        '駅距離': 'Distance to station',
        '価格予測': 'Price prediction',
        'データ前処理の重要性': 'Why preprocessing matters',
        'データに穴がある場合の対処法:': 'Ways to handle missing values:',
        '平均値で埋める': 'Impute with the mean',
        '中央値で埋める': 'Impute with the median',
        '行を削除': 'Remove rows',
        '特徴量の値の範囲を揃えます:': 'Put feature values on comparable scales:',
        '標準化': 'Standardization',
        ': 平均0、分散1に変換': ': transform to mean 0 and variance 1',
        '正規化': 'Normalization',
        ': 0~1の範囲に変換': ': transform to the 0-to-1 range',
        'カテゴリを数値に変換:': 'Convert categories to numbers:',
        'ラベル': 'Label',
        'すべての特徴量が等しく重要とは限りません。下のチャートは、住宅価格予測における各特徴量の重要度のシミュレーションです。 ボタンをクリックして異なるシナリオを確認しましょう。': 'Features do not contribute equally. The chart simulates feature importance for several prediction scenarios. Use the buttons to compare them.',
        '住宅価格': 'House price',
        '健康診断': 'Health screening',
        '売上予測': 'Sales forecast',
        '周辺施設数': 'Nearby amenities',
        '日当たり': 'Sun exposure',
        '治安スコア': 'Safety score',
        '部屋数': 'Rooms',
        '階数': 'Floor number',
        '駅距離(分)': 'Distance to station (min)',
        '面積(m2)': 'Floor area (m²)',
        '飲酒量': 'Alcohol consumption',
        '喫煙歴': 'Smoking history',
        '運動頻度': 'Exercise frequency',
        'コレステロール': 'Cholesterol',
        '血糖値': 'Blood glucose',
        '年齢': 'Age',
        '血圧': 'Blood pressure',
        '曜日': 'Day of week',
        '天気': 'Weather',
        'SNSフォロワー': 'Social-media followers',
        '在庫量': 'Inventory',
        '競合数': 'Competitors',
        '季節': 'Season',
        '価格': 'Price',
        '広告費': 'Advertising spend',
        '良い特徴量はモデルの精度を大きく向上させます。前処理で欠損値を適切に扱い、 スケーリングやエンコーディングでデータを整えることが、成功への近道です。': 'Informative features can greatly improve performance. Handle missing values appropriately and prepare the data with scaling and encoding.',

        'アンサンブル学習とは？': 'What is ensemble learning?',
        '複数のモデルを組み合わせることで、単一モデルよりも高い予測精度を達成する手法です。 「三人寄れば文殊の知恵」と同じ考え方です。': 'Ensemble learning combines multiple models and can outperform a single model by pooling their strengths.',
        '主なアンサンブル手法': 'Main ensemble methods',
        'データをランダムに抽出して複数のモデルを': 'Randomly sample the data and train several models in ',
        '並列': 'parallel',
        'に学習し、結果を平均（回帰）または多数決（分類）で統合します。': ', then combine them by averaging for regression or voting for classification.',
        'モデル1': 'Model 1',
        'モデル2': 'Model 2',
        'モデル3': 'Model 3',
        'モデル4': 'Model 4',
        'モデル5': 'Model 5',
        '多数決 / 平均': 'Vote / average',
        ': 分散を減らし、過学習を抑制': ': reduces variance and limits overfitting',
        '前のモデルの': 'Train each new model to ',
        '誤差を修正する': 'correct errors',
        'ように次のモデルを': ' made by the previous model, working ',
        '逐次的': 'sequentially',
        'に学習します。': '.',
        '誤差修正': 'Error correction',
        ': バイアスを減らし、高精度を実現': ': reduces bias and can improve accuracy',
        'スタッキング': 'Stacking',
        '複数のモデルの予測を': 'Use predictions from several models as inputs to ',
        '別のモデル（メタモデル）': 'another model (the meta-model)',
        'への入力として使い、最終予測を行います。 異なる種類のモデルの強みを組み合わせられます。': ' to make the final prediction and combine strengths from different model families.',
        'ブレンディング': 'Blending',
        'Stackingに似ていますが、ホールドアウトデータの予測値を使ってメタモデルを訓練する、 よりシンプルな手法です。データリークのリスクが低いのが利点です。': 'Blending is a simpler relative of stacking. It trains the meta-model on predictions for a holdout set, which helps limit leakage.',
        'インタラクティブ: アンサンブルの効果': 'Interactive exercise: the effect of ensembling',
        '個別モデルとアンサンブルの精度を比較してみましょう。ボタンでモデル数を変更できます。': 'Compare individual models with their ensemble and change the number of models with the buttons.',
        'モデル数:': 'Number of models:',
        '個別モデルの予測': 'Individual model predictions',
        'アンサンブル平均': 'Ensemble average',
        '真の値': 'True value',
        'アンサンブルによる予測の安定化': 'Prediction stabilization through ensembling',
        '個別モデルの平均誤差': 'Mean error of individual models',
        'アンサンブルの誤差': 'Ensemble error',
        'アンサンブル学習は多くのコンペティションで上位入賞の鍵となる手法です。 Baggingは過学習防止に、Boostingは精度向上に強く、 Stackingで異なるモデルの長所を組み合わせることで更なる改善が期待できます。': 'Ensembles are widely used in high-performing machine-learning systems. Bagging can reduce overfitting, boosting can improve accuracy, and stacking can combine strengths from different model families.',

        'easyDataScience の使い方': 'Using easyDataScience',
        'このアプリは、CSVファイルをアップロードするだけで、データ分析から機械学習モデルの構築まで、 すべてブラウザ上で完結するツールです。': 'Upload a CSV file to move from data exploration to model building entirely in the browser.',
        'ステップバイステップ ワークフロー': 'Step-by-step workflow',
        'データのアップロード': 'Upload data',
        'CSVファイルをドラッグ＆ドロップ、またはクリックしてアップロードします。 数値変数とカテゴリ変数は自動判別されます。': 'Drag and drop a CSV file or click to upload it. Numeric and categorical variables are detected automatically.',
        'データの概要、分布、相関、欠損値を確認します。 この段階でデータの全体像を把握することが重要です。': 'Review the overview, distributions, correlations, and missing values to understand the dataset before modeling.',
        '欠損値の補完、スケーリング、エンコーディングなどを行います。 適切な前処理がモデルの精度を大きく左右します。': 'Handle missing values, scaling, and encoding. Preprocessing choices can strongly affect model performance.',
        '分析タイプの選択': 'Choose the analysis type',
        '目的変数の種類に応じて「回帰」か「分類」を選択します。 連続値なら回帰、カテゴリなら分類です。': 'Choose regression for a continuous target or classification for a categorical target.',
        'モデルの学習と比較': 'Train and compare models',
        '複数のアルゴリズムが自動的に比較され、最適なモデルが推薦されます。 交差検証により信頼性の高い評価が行われます。': 'Several algorithms are compared automatically. Cross-validation provides a more reliable comparison estimate.',
        '評価指標、特徴量重要度、SHAP値などでモデルの振る舞いを理解します。 予測結果はCSVでダウンロードできます。': 'Use metrics, feature importance, and SHAP-style explanations to inspect model behavior. Prediction results can be downloaded as CSV.',
        'どの分析を選べばいい？': 'Which analysis should I choose?',
        'いつ使う？': 'When to use it',
        'まずはこれから始めましょう。データの全体像を把握するために必ず最初に行います。': 'Start here to understand the overall dataset before modeling.',
        'わかること:': 'What it shows:',
        '分布、相関、欠損値の状況': 'Distributions, correlations, and missingness',
        'EDAで問題が見つかった場合（欠損値、外れ値、スケールの違いなど）。': 'Use this when EDA reveals missing values, outliers, or incompatible scales.',
        'できること:': 'What you can do:',
        '欠損値処理、スケーリング、エンコーディング': 'Missing-value handling, scaling, and encoding',
        '回帰分析': 'Regression analysis',
        '目的変数が': 'Use it when the target is ',
        '連続値': 'continuous',
        '（価格、温度、売上など）の場合。': ', such as price, temperature, or sales.',
        '線形回帰、Ridge、Lasso、決定木、ランダムフォレスト、勾配ブースティングなど': 'Linear regression, ridge, lasso, decision trees, random forests, gradient boosting, and more',
        '分類分析': 'Classification analysis',
        'カテゴリ': 'categorical',
        '（Yes/No、A/B/C など）の場合。': ', such as Yes/No or A/B/C.',
        'ロジスティック回帰、SVM、決定木、ランダムフォレスト、勾配ブースティングなど': 'Logistic regression, SVM, decision trees, random forests, gradient boosting, and more',
        '結果の読み方': 'How to read the results',
        '回帰の場合': 'For regression',
        ': 1に近いほど良い予測。0.7以上なら良好。': ': closer to 1 indicates better fit, though usefulness depends on the problem and evaluation design.',
        ': 予測誤差の大きさ。値が小さいほど良い。': ': prediction-error magnitude; smaller values are better.',
        ': ランダムに散らばっていれば良い。パターンがあれば改善の余地あり。': ': random scatter is desirable; visible patterns suggest model misspecification.',
        '分類の場合': 'For classification',
        ': 全体の正解率。クラスが不均衡な場合は注意。': ': overall accuracy; interpret cautiously with class imbalance.',
        'F1スコア': 'F1 score',
        ': PrecisionとRecallのバランス指標。': ': balances precision and recall.',
        ': どのクラスを間違えやすいかがわかる。': ': shows which classes are confused.',
        '分析は「EDA → 前処理 → モデル構築 → 評価」の順で進めましょう。 急いでモデルを作るより、データをしっかり理解することが成功の鍵です。 このアプリはすべてブラウザ上で動作し、データがサーバーに送信されることはありません。': 'Work through EDA, preprocessing, model building, and evaluation in that order. Understanding the data matters more than rushing to fit a model. The app runs in the browser and does not send your data to a server.'
    }).map(([key, value]) => [key.replace(/\s+/g, ' ').trim(), value]))],
    ['video', new Map(Object.entries({
        'ファイルを開く': 'Open file',
        'ツール対象': 'Tool target',
        'アクティブ': 'Active',
        '静止画': 'Still image',
        'オフセット (B - A)': 'Offset (B - A)',
        '秒': 'seconds',
        '同期停止': 'Pause both',
        '両方0秒へ': 'Reset both to 0 seconds',
        '同期再生は両側が動画のときだけ有効です。': 'Synchronized playback is available only when both sides contain videos.',
        '描画ツール': 'Drawing tool',
        '色': 'Color',
        '太さ': 'Width',
        '表示秒数': 'Display duration',
        '1つ戻す': 'Undo one',
        'JSON保存': 'Save JSON',
        'JSON読込': 'Load JSON',
        '使い方': 'How to use',
        ': 「ツール対象」で動画/画像を選んでから上のメディアをドラッグして描画します。 動画では現在時刻から指定秒数だけ表示され、再生中に自動でフェードアウト。画像ではずっと表示されたままになります。': ': Choose a video or image with Tool target, then drag over the media to draw. On video, drawings appear for the selected duration from the current time and fade during playback. On images, drawings remain visible.',
        'Step 1 で動画を読み込むとタグ／イベント記録が利用できます。': 'Load a video in Step 1 to record tags and events.',
        '画像にはタグ付けできません。動画ファイルを読み込んでください。': 'Tags are available for video only. Load a video file to record events.',
        'タグ種別': 'Tag types',
        '先頭9件のタグはホットキー': 'The first nine tags can use hotkeys ',
        'で打鍵できます（テキスト入力中は無効）。': ' (disabled while typing in a text field).',
        '記録': 'Record',
        'グッドプレー': 'Good play',
        'ミス': 'Mistake',
        'シュート': 'Shot',
        'パス': 'Pass',
        'CSV出力': 'Export CSV',
        '全削除': 'Delete all',
        'タグ一覧': 'Tag list',
        '時刻': 'Time',
        'ラベル': 'Label',
        'Step 1 でメディアを読み込んでから推定してください。': 'Load media in Step 1 before running pose estimation.',
        '画像モードでは「現在フレームを推定」のみ利用できます。連続推定は動画のみ対応です。': 'For images, only current-frame estimation is available. Continuous estimation requires video.',
        '連続推定 開始': 'Start continuous estimation',
        '連続推定 停止': 'Stop continuous estimation',
        'サンプリング': 'Sampling',
        '骨格オーバーレイ': 'Skeleton overlay',
        '未推定': 'Not estimated',
        'サンプル削除': 'Delete samples',
        '関節角度CSV': 'Joint-angle CSV',
        '関節角度の現在値': 'Current joint angles',
        '左肘': 'Left elbow',
        '右肘': 'Right elbow',
        '左膝': 'Left knee',
        '右膝': 'Right knee',
        '左肩': 'Left shoulder',
        '右肩': 'Right shoulder',
        '関節角度の時系列': 'Joint angles over time',
        'まだサンプルがありません。「現在フレームを推定」または「連続推定」を実行してください。': 'No samples yet. Estimate the current frame or start continuous estimation.',
        '連続推定で蓄積されたサンプル（時刻×関節角度）をプロットします。': 'Plot samples collected through continuous estimation as time by joint angle.',
        'MoveNetについて': 'About MoveNet',
        ': 人物を1人検出し、17キーポイント（頭・肩・肘・手首・腰・膝・足首など）の2D座標を返します。 背景や衣服、視点、解像度の影響を受けるため、ぶれる場合は信頼度が低くなり描画されません。教育用の目安としてご利用ください。': ': Detects one person and returns 2D coordinates for 17 keypoints, including the head, shoulders, elbows, wrists, hips, knees, and ankles. Background, clothing, viewpoint, and resolution can reduce confidence, so unstable points may not be drawn. Treat the result as an educational aid.',
        'このメディアを描画/タグ/ポーズ推定の対象にする': 'Use this media for drawing, tags, and pose estimation',
        'メディアと描画・タグを破棄': 'Remove the media, drawings, and tags',
        '再生/一時停止 (Space)': 'Play/pause (Space)',
        '前フレーム (,)': 'Previous frame (,)',
        '次フレーム (.)': 'Next frame (.)',
        '再生速度': 'Playback speed',
        '音量': 'Volume',
        '動画モードでのみ有効。画像はずっと表示されます。': 'Available for video only. Drawings remain visible on images.',
        '直前の描画を取り消し': 'Undo the most recent drawing',
        '全描画を削除': 'Delete all drawings',
        '新しいタグ名': 'New tag name'
    }).map(([key, value]) => [key.replace(/\s+/g, ' ').trim(), value]))]
]);

const SCOPED_PATTERN_TRANSLATIONS = new Map([
    ['learning', [
        [/^クラス (\d+)$/, (_, index) => `Class ${index}`],
        [/^: Y = (.+) \| 正解率 = ([\d.]+)%$/, (_, rule, accuracy) => `: Y = ${rule} | accuracy = ${accuracy}%`],
        [/^モデル(\d+)$/, (_, index) => `Model ${index}`]
    ]],
    ['video', [
        [/^対象: メディア([AB]) \((未読込|動画|画像|静止画)\)$/, (_, target, status) => `Target: Media ${target} (${({ '未読込': 'not loaded', '動画': 'video', '画像': 'image', '静止画': 'still image' })[status]})`],
        [/^対象: メディア([AB])$/, (_, target) => `Target: Media ${target}`],
        [/^描画 \(メディア([AB])\) を JSON に保存$/, (_, target) => `Save drawings for Media ${target} as JSON`],
        [/^関節角度 \(メディア([AB])\) を CSV に保存$/, (_, target) => `Save joint angles for Media ${target} as CSV`]
    ]]
]);

const PATTERN_TRANSLATIONS = [
    [/^デモデータ \((.+)\) を読み込み中\.\.\.$/, (_, name) => `Loading demo data (${name})...`],
    [/^デモデータ \((.+)\) の読み込みに失敗しました。$/, (_, name) => `Could not load demo data (${name}).`],
    [/^(\d+) 枚$/, (_, count) => `${count} images`],
    [/^\((\d+) サンプル\)$/, (_, count) => `(${count} samples)`],
    [/^(\d+) \/ (\d+) 正解$/, (_, correct, total) => `${correct} / ${total} correct`],
    [/^信頼度: ([\d.]+)%$/, (_, confidence) => `Confidence: ${confidence}%`],
    [/^テストデータ \((\d+)件\)$/, (_, count) => `Test data (${count} samples)`],
    [/^訓練データ \((\d+)件\)$/, (_, count) => `Training data (${count} samples)`],
    [/^先頭 (\d+) 行を表示（全 (\d+) 行）$/, (_, shown, total) => `Showing the first ${shown} of ${total} rows`],
    [/^(.+) を学習中\.\.\. \((\d+)\/(\d+)\)$/, (_, model, current, total) => `Training ${translateJapanese(model)}... (${current}/${total})`],
    [/^(.+) の分布$/, (_, variable) => `Distribution of ${variable}`],
    [/^特徴量: (\d+)個$/, (_, count) => `Features: ${count}`],
    [/^(\d+)個を平均値で補完$/, (_, count) => `${count} values imputed with the mean`],
    [/^(\d+)列をLabel Encoding$/, (_, count) => `${count} columns label encoded`],
    [/^(\d+)行をIQR法で除去$/, (_, count) => `${count} rows removed using the IQR rule`],
    [/^(.+) にlog変換$/, (_, variables) => `Log transform: ${variables}`],
    [/^(.+) を除去$/, (_, variables) => `Removed: ${variables}`],
    [/^訓練(\d+)件 \/ テスト(\d+)件$/, (_, train, test) => `Train: ${train} / Test: ${test}`],
    [/^欠損値処理: (.+)$/, (_, detail) => `Missing values: ${translateJapanese(detail)}`],
    [/^カテゴリ変数: (.+)$/, (_, detail) => `Categorical variables: ${translateJapanese(detail)}`],
    [/^外れ値除去: (.+)$/, (_, detail) => `Outlier removal: ${translateJapanese(detail)}`],
    [/^特徴量変換: (.+)$/, (_, detail) => `Feature transformation: ${translateJapanese(detail)}`],
    [/^多重共線性: (.+)$/, (_, detail) => `Multicollinearity: ${translateJapanese(detail)}`],
    [/^スケーリング: (.+)$/, (_, detail) => `Scaling: ${translateJapanese(detail)}`],
    [/^データ分割: 訓練(\d+)件 \/ テスト(\d+)件$/, (_, train, test) => `Data split: ${train} training / ${test} test rows`],
    [/^(要確認|注意|参考|良好) (\d+)$/, (_, label, count) => `${translateJapanese(label)} ${count}`],
    [/^(\d+)行あり、学習用デモとしては比較しやすい規模です。$/, (_, rows) => `${rows} rows are available, a practical size for a learning demonstration.`],
    [/^(\d+)行です。機械学習の評価は大きく揺れやすいため、学習用の実験結果として扱ってください。$/, (_, rows) => `There are only ${rows} rows, so performance estimates may vary substantially. Treat this as a learning experiment.`],
    [/^(\d+)行です。テスト分割やCVのfoldによって順位が変わる可能性があります。$/, (_, rows) => `There are ${rows} rows, so rankings may change with the test split or CV folds.`],
    [/^特徴量(\d+)個です。データ件数に対して極端に多い状態ではありません。$/, (_, count) => `${count} features are selected, which is not excessive relative to the data size.`],
    [/^\((\d+)クラス\)$/, (_, count) => `(${count} classes)`],
    [/^(\d+)クラスで、極端な多数派偏りは見つかりません。$/, (_, count) => `${count} classes are present, with no extreme majority-class imbalance detected.`],
    [/^訓練(\d+)件に対して特徴量(\d+)個です。偶然の当たりや過学習が起きやすい状態です。$/, (_, rows, features) => `${features} features are used with ${rows} training rows, increasing the risk of chance fit and overfitting.`],
    [/^訓練(\d+)件に対して特徴量(\d+)個です。モデル解釈と過学習を確認してください。$/, (_, rows, features) => `${features} features are used with ${rows} training rows. Check model interpretation and overfitting.`],
    [/^選択した目的変数・特徴量には欠損が見つかりません。$/, () => 'No missing values were found in the selected target and features.'],
    [/^回帰対象として使える数値のばらつきがあります。$/, () => 'The numeric target has enough variation for regression.'],
    [/^(\d+)-Foldで比較します。$/, (_, folds) => `Models are compared with ${folds}-fold cross-validation.`],
    [/^(\d+)-Foldです。小規模データでは安定性と計算量のバランスを見て調整してください。$/, (_, folds) => `${folds}-fold cross-validation is used. For small datasets, balance stability against computation.`],
    [/^(\d+)-Fold 交差検証スコア（foldごとに前処理をfit、参考値）でソートしています。テストデータ \((\d+) サンプル\) での評価結果も併記。$/, (_, folds, samples) => `Sorted by reference ${folds}-fold CV scores with preprocessing fitted inside each fold. Test-set results for ${samples} samples are also shown.`],
    [/^(\d+)-Fold 前処理込みCV平均$/, (_, folds) => `Mean ${folds}-fold CV score with fold-specific preprocessing`],
    [/^(.+) の詳細評価$/, (_, model) => `Detailed evaluation: ${translateJapanese(model)}`],
    [/^(.+) \((LR|Tree|RF|KNN|NB|SVM|GBM|Linear|Ridge|Lasso)\)( ★1位)?$/, (_, model, badge, top) => `${translateJapanese(model)} (${badge})${top ? ' #1' : ''}`],
    [/^予測結果 \((.+)\)$/, (_, model) => `Prediction result (${translateJapanese(model)})`],
    [/^(.+) の評価結果:$/, (_, model) => `Evaluation results for ${translateJapanese(model)}:`],
    [/^MAE = ([\d.\-]+) : 予測値と実測値の平均的なずれは ([\d.\-]+) です。$/, (_, value, repeated) => `MAE = ${value}: the average absolute difference between predictions and actual values is ${repeated}.`],
    [/^RMSE = ([\d.\-]+) : 大きな誤差をより重く評価した指標で ([\d.\-]+) です。$/, (_, value, repeated) => `RMSE = ${value}: this metric gives greater weight to large errors.`],
    [/^ROC曲線 \(AUC = ([\d.\-]+)\)$/, (_, value) => `ROC curve (AUC = ${value})`],
    [/^KNNに係数はありません。近いサンプル (\d+) 件の多数決または距離重みで分類します。 距離ベースなのでスケーリング、不要特徴量、外れ値の影響を確認してください。$/, (_, neighbors) => `KNN has no coefficients. It classifies using a majority vote or distance weighting among the ${neighbors} nearest samples. Because it is distance-based, check the effects of scaling, irrelevant features, and outliers.`],
    [/^KNNに係数はありません。予測は近いサンプル (\d+) 件の目的変数から決まります。 距離に基づくため、スケーリング済みであることと、外れ値・不要特徴量の影響に注意してください。$/, (_, neighbors) => `KNN has no coefficients. Its prediction is determined by the target values of the ${neighbors} nearest samples. Because it is distance-based, verify scaling and check sensitivity to outliers and irrelevant features.`],
    [/^L1正則化 alpha=([\d.\-]+) \/ 非ゼロ係数 (\d+)\/(\d+)。係数は前処理後、主に標準化後特徴量に対する値です。数値特徴量では符号が方向、絶対値が影響の大きさの目安です。 Label Encodingされたカテゴリ特徴量の係数はカテゴリ順序を意味しないため、符号や大小を強く解釈しないでください。$/, (_, alpha, nonZero, total) => `L1 regularization with alpha=${alpha}; ${nonZero}/${total} coefficients are non-zero. Coefficients are measured on preprocessed, usually standardized features: the sign indicates direction and the absolute value indicates relative influence. Do not strongly interpret coefficients for label-encoded categories because their numeric codes do not represent an ordered scale.`],
    [/^L2正則化 alpha=([\d.\-]+)。係数は前処理後、主に標準化後特徴量に対する値です。数値特徴量では符号が方向、絶対値が影響の大きさの目安です。 Label Encodingされたカテゴリ特徴量の係数はカテゴリ順序を意味しないため、符号や大小を強く解釈しないでください。$/, (_, alpha) => `L2 regularization with alpha=${alpha}. Coefficients are measured on preprocessed, usually standardized features: the sign indicates direction and the absolute value indicates relative influence. Do not strongly interpret coefficients for label-encoded categories because their numeric codes do not represent an ordered scale.`],
    [/^です。モデルがデータの([\d.]+)%の分散を説明しています。$/, (_, percent) => `. The model explains ${percent}% of the target variance in the test set.`],
    [/^CV R² = ([\d.\-]+) ± ([\d.\-]+) : 交差検証とテストの差が小さく、$/, (_, mean, std) => `CV R² = ${mean} ± ${std}: the cross-validation and test scores are close, indicating `],
    [/^CV R² = ([\d.\-]+) ± ([\d.\-]+) : CVがテストR²より高く、$/, (_, mean, std) => `CV R² = ${mean} ± ${std}: cross-validation is higher than test R², suggesting `],
    [/^GridSearch CV（foldごとに前処理をfitする参考値）でパラメータを最適化します。 探索範囲: (.+)$/, (_, searchSpace) => `Optimize parameters with GridSearch CV using preprocessing fitted separately in each fold. Search space: ${searchSpace}`],
    [/^CV F1 = ([\d.\-]+) ± ([\d.\-]+) : 交差検証とテストの差が小さく、$/, (_, mean, std) => `CV F1 = ${mean} ± ${std}: the cross-validation and test scores are close, indicating `],
    [/^CV F1 = ([\d.\-]+) ± ([\d.\-]+) : CVがテストF1より高く、$/, (_, mean, std) => `CV F1 = ${mean} ± ${std}: cross-validation is higher than test F1, suggesting `],
    [/^Accuracy = ([\d.\-]+) : 全体の ([\d.\-]+)% を正しく分類できました。$/, (_, value, percent) => `Accuracy = ${value}: ${percent}% of test observations were classified correctly.`],
    [/^Precision = ([\d.\-]+) : 正と予測したものの ([\d.\-]+)% が実際に正でした。$/, (_, value, percent) => `Macro precision = ${value}: ${percent}% of class-positive predictions were correct on average across classes.`],
    [/^Recall = ([\d.\-]+) : 実際に正のものの ([\d.\-]+)% を検出できました。$/, (_, value, percent) => `Macro recall = ${value}: ${percent}% of class-positive observations were detected on average across classes.`],
    [/^AUC = ([\d.\-]+) : ROC曲線下面積で、1\.0に近いほどランダムより優れた分類です。$/, (_, value) => `AUC = ${value}: area under the ROC curve; values closer to 1.0 indicate better ranking than random.`],
    [/^事前確率: (.+)$/, (_, label) => `Prior probability: ${label}`],
    [/^表示方向: (.+) 方向$/, (_, label) => `Displayed direction: ${label}`],
    [/^(.+) に寄る$/, (_, label) => `Toward ${label}`],
    [/^(.+) の識別に強い$/, (_, label) => `Strong for identifying ${label}`],
    [/^回帰モデル比較 \(AutoML\) - (.+)$/, (_, model) => `Regression model comparison (AutoML) - ${translateJapanese(model)}`],
    [/^分類モデル比較 \(AutoML\) - (.+)$/, (_, model) => `Classification model comparison (AutoML) - ${translateJapanese(model)}`],
    [/^質問: (.+)$/, (_, question) => `Question: ${question}`],
    [/^最終更新: (.+)$/, (_, date) => `Last updated: ${date}`]
];

let currentLanguage = readStoredLanguage();
let observer = null;
let initialized = false;
let applying = false;
let originalTitle = null;
const originalText = new WeakMap();
const originalAttributes = new WeakMap();

export function getLanguage() {
    return currentLanguage;
}

export function isEnglish() {
    return currentLanguage === 'en';
}

export function tr(value, language = currentLanguage) {
    if (language !== 'en' || typeof value !== 'string') return value;
    return translateJapanese(value);
}

export function setLanguage(language, { persist = true } = {}) {
    const nextLanguage = SUPPORTED_LANGUAGES.has(language) ? language : 'ja';
    currentLanguage = nextLanguage;

    if (persist) {
        try {
            localStorage.setItem(LANGUAGE_STORAGE_KEY, nextLanguage);
        } catch (error) {
            console.warn('Could not save language preference:', error);
        }
    }

    applyLanguage(document);
    updateLanguageControls();
    window.dispatchEvent(new CustomEvent(LANGUAGE_CHANGE_EVENT, {
        detail: { language: nextLanguage }
    }));
}

export function initializeI18n() {
    if (initialized) {
        applyLanguage(document);
        updateLanguageControls();
        return;
    }

    initialized = true;
    originalTitle = document.title;

    document.addEventListener('click', (event) => {
        const button = event.target.closest('[data-language-option]');
        if (!button) return;
        setLanguage(button.dataset.languageOption);
    });

    startObserver();
    applyLanguage(document);
    updateLanguageControls();
}

export function applyLanguage(root = document) {
    if (!root) return;
    applying = true;
    stopObserver();

    document.documentElement.lang = currentLanguage;
    if (document.body) document.body.dataset.language = currentLanguage;

    if (originalTitle == null) originalTitle = document.title;
    document.title = currentLanguage === 'en'
        ? translateJapanese(originalTitle)
        : originalTitle;

    translateSubtree(root);
    applying = false;
    startObserver();
}

function readStoredLanguage() {
    try {
        const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
        return SUPPORTED_LANGUAGES.has(stored) ? stored : 'ja';
    } catch (error) {
        return 'ja';
    }
}

function translateJapanese(value, element = null) {
    const normalized = normalizeText(value);
    if (!normalized) return value;

    const scope = element?.closest?.('[data-i18n-scope]')?.dataset.i18nScope;
    const scoped = scope ? SCOPED_JA_TO_EN.get(scope)?.get(normalized) : null;
    if (scoped != null) return preserveOuterWhitespace(value, scoped);

    for (const [pattern, replacement] of SCOPED_PATTERN_TRANSLATIONS.get(scope) || []) {
        const match = normalized.match(pattern);
        if (match) {
            const translated = typeof replacement === 'function'
                ? replacement(...match)
                : normalized.replace(pattern, replacement);
            return preserveOuterWhitespace(value, translated);
        }
    }

    const exact = JA_TO_EN.get(normalized);
    if (exact != null) return preserveOuterWhitespace(value, exact);

    for (const [pattern, replacement] of PATTERN_TRANSLATIONS) {
        const match = normalized.match(pattern);
        if (match) {
            const translated = typeof replacement === 'function'
                ? replacement(...match)
                : normalized.replace(pattern, replacement);
            return preserveOuterWhitespace(value, translated);
        }
    }

    return value;
}

function normalizeText(value) {
    return String(value).replace(/\s+/g, ' ').trim();
}

function preserveOuterWhitespace(source, translated) {
    const leading = source.match(/^\s*/)?.[0] || '';
    const trailing = source.match(/\s*$/)?.[0] || '';
    return `${leading}${translated}${trailing}`;
}

function translateSubtree(root) {
    if (root.nodeType === Node.TEXT_NODE) {
        translateTextNode(root);
        return;
    }

    if (root.nodeType !== Node.DOCUMENT_NODE && root.nodeType !== Node.ELEMENT_NODE) return;
    if (root.nodeType === Node.ELEMENT_NODE) {
        translateExplicitElement(root);
        translateAttributes(root);
    }

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
        if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
        else {
            translateExplicitElement(node);
            translateAttributes(node);
        }
        node = walker.nextNode();
    }
}

function translateTextNode(node) {
    if (!node.parentElement || shouldIgnore(node.parentElement)) return;
    if (node.parentElement.closest('[data-i18n-en]')) return;

    const liveValue = node.nodeValue || '';
    if (!originalText.has(node) || (currentLanguage === 'en' && containsJapanese(liveValue))) {
        originalText.set(node, liveValue);
    }

    const source = originalText.get(node) ?? liveValue;
    const nextValue = currentLanguage === 'en' ? translateJapanese(source, node.parentElement) : source;
    if (node.nodeValue !== nextValue) node.nodeValue = nextValue;
}

function translateExplicitElement(element) {
    if (!element.hasAttribute('data-i18n-en') || element.children.length > 0) return;
    const textNode = Array.from(element.childNodes).find(node => node.nodeType === Node.TEXT_NODE);
    if (!textNode) return;

    if (!originalText.has(textNode)) originalText.set(textNode, textNode.nodeValue || '');
    const source = originalText.get(textNode) || '';
    const nextValue = currentLanguage === 'en' ? element.dataset.i18nEn : source;
    if (textNode.nodeValue !== nextValue) textNode.nodeValue = nextValue;
}

function translateAttributes(element) {
    if (shouldIgnore(element)) return;

    let sources = originalAttributes.get(element);
    if (!sources) {
        sources = new Map();
        originalAttributes.set(element, sources);
    }

    TRANSLATABLE_ATTRIBUTES.forEach(attribute => {
        if (!element.hasAttribute(attribute)) return;
        const liveValue = element.getAttribute(attribute) || '';
        if (!sources.has(attribute) || (currentLanguage === 'en' && containsJapanese(liveValue))) {
            sources.set(attribute, liveValue);
        }
        const source = sources.get(attribute) || '';
        const nextValue = currentLanguage === 'en' ? translateJapanese(source, element) : source;
        if (liveValue !== nextValue) element.setAttribute(attribute, nextValue);
    });

    if (element.hasAttribute('data-i18n-value-en') && 'value' in element) {
        const liveValue = element.value || '';
        if (!sources.has('value')) sources.set('value', liveValue);
        const source = sources.get('value') || '';
        const nextValue = currentLanguage === 'en' ? element.dataset.i18nValueEn : source;
        if (liveValue !== nextValue) element.value = nextValue;
    }
}

function shouldIgnore(element) {
    if (!element) return true;
    if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'CODE', 'PRE'].includes(element.tagName)) return true;
    return Boolean(element.closest('[data-i18n-ignore]'));
}

function containsJapanese(value) {
    return /[ぁ-んァ-ン一-龯]/.test(value);
}

function updateLanguageControls() {
    document.querySelectorAll('[data-language-option]').forEach(button => {
        const active = button.dataset.languageOption === currentLanguage;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });

    document.querySelectorAll('[data-language-switch]').forEach(control => {
        control.setAttribute('aria-label', currentLanguage === 'en'
            ? 'Display language'
            : '表示言語');
    });
}

function startObserver() {
    if (applying || !document.documentElement) return;
    if (!observer) observer = new MutationObserver(handleMutations);
    observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true,
        attributes: true,
        attributeFilter: TRANSLATABLE_ATTRIBUTES
    });
}

function stopObserver() {
    if (!observer) return;
    observer.disconnect();
    observer.takeRecords();
}

function handleMutations(mutations) {
    if (applying) return;
    applying = true;
    stopObserver();

    mutations.forEach(mutation => {
        if (mutation.type === 'characterData') {
            translateTextNode(mutation.target);
            return;
        }
        if (mutation.type === 'attributes') {
            translateAttributes(mutation.target);
            return;
        }
        mutation.addedNodes.forEach(node => translateSubtree(node));
    });

    applying = false;
    startObserver();
}

export { LANGUAGE_CHANGE_EVENT, LANGUAGE_STORAGE_KEY };
