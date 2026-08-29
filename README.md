# easy_data_science_edu

ブラウザだけで動作する、教育用の PyCaret ライクな AutoML アプリです。

使い方は [manual.html](manual.html) を参照してください。Markdown版は [USAGE_MANUAL.md](USAGE_MANUAL.md) にあります。

画面上部の `JP / EN` で、操作画面、分析結果の説明、学習ガイド、マニュアル、生成AI支援を日本語または英語へ切り替えられます。選択はブラウザに保存され、列名・ファイル名・クラス名・入力値などのユーザーデータは翻訳されません。

回帰・分類のモデル比較は、前処理を各fold内で学習する交差検証で行います。未使用テストデータは候補モデルを決めるまで封印し、開示後の追加調整やアンサンブル比較は探索的な結果として表示します。

カテゴリ特徴量は、保持データや検証foldを見ずに訓練側でOne-Hot Encodingします。検証・予測時の未知カテゴリは全0で表し、線形モデルなどへ人工的な順序を持ち込みません。

分割方法は、独立な行の無作為分割、同一人物・学校などをまたがせないグループ分割、過去から未来を評価する時系列分割（任意のgap付き）から選べます。データの生成過程に合う方法を選んでください。

通常のデータ処理と学習はブラウザ内で完結します。生成AI支援を実行した場合だけ分析文脈が Google Gemini API へ送信されます。送信前に実際の文脈を画面で確認し、個人情報・機密情報を含まないことへのチェックが必要です。先頭10件のプレビューは利用者が明示的に許可した場合だけ含まれますが、列名・クラス名・要約統計・分析結果はプレビュー設定にかかわらず送信対象です。APIキーと追加質問の履歴はページメモリ内だけに保持し、Interactions APIには常に`store=false`を指定します。Googleの[データ保持案内](https://ai.google.dev/gemini-api/docs/zdr)も確認してください。

Gemini連携の既定モデルは、2026年8月時点の安定版[`gemini-3.7-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash)です。推奨される[Interactions API](https://ai.google.dev/gemini-api/docs/migrate-to-interactions)を使用し、解釈は構造化JSONとして受信後に型検証します。Google AI Studioで作成したAuth keyを使用してください。Googleは[2026年9月からStandard keyを拒否する](https://ai.google.dev/gemini-api/docs/api-key)と案内しています。
