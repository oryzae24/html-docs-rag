# 検索・回答品質改善戦略

## 目的と原則

この文書は、Python 3.13日本語公式ドキュメントRAGについて、現在のDense検索を
起点に、検索品質と回答品質をどの順序で改善し、どの評価値で採否を判断するかを
固定する。実装担当者は、一度に複数の改善を混ぜず、同一の質問セット、コーパス、
正解ラベルで各段階を個別に比較する。

基本原則は次のとおり。

- 本体はlocal-firstとし、有料APIや外部providerを必須にしない。
- Retrieverの改善とGeneratorの改善を分離して評価する。
- URLは取得済みmetadataを使用し、モデルに生成させない。
- 評価値を高くするために正解URLを不自然に広げない。
- 変更ごとに再現条件、結果、採用・不採用理由を実験summaryへ残す。
- 現在の26問はRetriever比較用の固定セットとして維持する。

## 1. 現在のDense baseline

保存済みFAISS `IndexFlatIP`と
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`を使用した実測値は
次のとおり。

| 指標 | 値 |
| --- | ---: |
| 質問数 | 26 |
| Hit@1 | 0.3846 |
| Hit@3 | 0.5385 |
| Hit@5 | 0.6923 |
| Hit@10 | 0.8077 |
| MRR@10 | 0.4972 |
| 平均検索時間 | 0.0168秒 |
| 中央検索時間 | 0.0092秒 |

質問種類別の傾向は次のとおり。

| query_type | Hit@1 | Hit@10 | MRR@10 | 傾向 |
| --- | ---: | ---: | ---: | --- |
| exact_identifier | 0.5000 | 0.8000 | 0.5976 | 全体では強いが希少識別子に失敗する |
| conceptual | 0.2500 | 0.7500 | 0.3710 | 正解が上位へ集まりにくい |
| operational | 0.3750 | 0.8750 | 0.4979 | 操作語とAPI名が本文にある質問に強い |

完全な数値、質問別結果、main関数とisattyのTop-10は
[`evaluation/dense_baseline_summary.md`](../evaluation/dense_baseline_summary.md)を参照する。

## 2. 確認された失敗パターン

- main関数の質問では、「配置」「位置」という一般語に引かれ、heapqの優先度
  キュー、GUI layout、演算子優先順位などを取得した。
- isattyでは、短い識別子の字面をDense Embeddingが十分に保持できず、AST、errno、
  IMAP4などのAPI節を取得した。
- 同一または類似セクションの複数チャンクがTop-10を占有し、候補の多様性が下がった。
- Dense検索はsubprocess、pathlib、shlexなど、API名と操作語が本文に現れる質問には
  比較的強い。
- 概念的な言い換え、慣習的な設計質問、希少識別子、識別子を自然文へ言い換えた
  質問に弱い。

この結果から、最初の改善対象は語彙一致を補うSparse検索であり、重複抑制や
Generator変更を先に導入しない。

## 3. 評価データセットの分離

### Retriever評価

現在の`evaluation/questions.jsonl`はRetriever専用の固定評価セットとする。
Hit@KやMRRの比較中は、質問文、query_type、topic、正解URLを変更しない。ラベルの
誤りを発見した場合は、改善方式の比較とは別の変更として根拠と影響を記録する。

### RAG品質評価

将来、`evaluation/rag_quality_questions.jsonl`を追加する。用途は次の指標の評価。

- Contextual Precision
- Contextual Recall
- Contextual Relevancy
- Faithfulness
- Answer Relevancy
- 引用妥当性

想定フィールド:

- `id`
- `question`
- `expected_answer`
- `expected_url_keywords`
- `query_type`
- `topic`

### 回答可能性評価

将来、`evaluation/answerability_questions.jsonl`を追加する。用途は回答可能・回答不能
判定、関連性閾値の調整、無関係な文書を引用する挙動の抑制である。

想定フィールド:

- `id`
- `question`
- `answerable`
- `expected_answer`
- `reason`

現在の`evaluation/unanswerable_candidates.jsonl`は、この評価セットを作成するための
候補置き場とする。候補をそのまま正解データとして扱わず、人手で根拠と期待挙動を
確定してから移す。

## 4. 改善順序

改善は次の順序で行う。

1. Dense baseline
2. 日本語・コード識別子対応BM25
3. Dense＋BM25の順位統合
4. child chunk検索＋parent section返却
5. Rerank
6. 回答不能判定
7. 同一URL・同一節の重複除去とMMR
8. 元質問を維持したQuery Rewrite
9. 合成質問、生成タイトル、field別Embedding
10. Embeddingモデル比較
11. QwenとOpenAI AnswerGenerator比較

各段階は同じ評価セットで個別に比較する。例えば、BM25とRRFを同時に実装して
Dense baselineと比較してはならない。前段の採否を確定してから次段へ進む。

### 進捗チェックポイント（2026-07-29）

Dense baseline、日本語文字2-gramのBM25単独評価、Dense＋BM25のRRF評価まで完了した。
RRFは`rrf_k=10`、`candidate_k=30`、DenseとBM25を同一重みとした。HybridはDense比で
Hit@1、Hit@10、MRR@10を改善したが、Hit@5は同値だった。

実Qwen評価では、引用形式検証を通過しても出典が回答を意味的に支持しない例を確認した。
このためCLI既定はDenseを維持し、Hybridは`--retriever hybrid`によるopt-inとする。
次段階へ進む前に、現在の26問と重複しない独立holdoutとgrounding評価を実施し、その
結果を確認してからHybridの既定化を再検討する。

## 5. 各段階の設計方針

### BM25

- `isatty`、`__name__`、`shlex.split`、`argv[0]`などのPython識別子を壊さない。
- 日本語本文には形態素分割と文字n-gramを候補として比較する。
- Notebookにある空白分割例を、そのまま日本語コーパスへ適用しない。
- Denseと同じ26問で、tokenizer方式ごとの単独BM25結果を保存する。
- 新しい依存を導入する場合は、標準ライブラリまたは小さなローカル実装との
  複雑度・再現性も比較する。

### Hybrid Retrieval

- Dense上位候補とBM25上位候補をそれぞれ取得する。
- 第一候補はReciprocal Rank Fusionなどの順位ベース統合とする。
- 尺度の異なるDense scoreとBM25 scoreを正規化なしで直接加算しない。
- 各Retrieverの取得数、RRF定数、重みは固定値を推測せず評価結果から決定する。
- Dense単独、BM25単独、Hybridの3結果を同じsummaryで比較する。

### Parent Document

- 小さいchild chunkを検索単位に使用する。
- v1では既存のcitation-ready `SearchChunk`をparentとして回答生成へ渡す。
- v1を見出し全体の`DocumentSection` parentとは呼ばない。
- 既存の`source_url`、ページタイトル、節タイトルmetadataを保持する。
- childとparentの対応、生成条件、件数をmanifestへ記録する。
- parent化による文脈量増加がQwenのtoken上限内か測定する。
- 完全な`DocumentSection`は一般HTML ingestionで永続化した後、同じ`ParentStore`
  境界へ差し替える。

### Rerank

- Hybridで取得した10〜20件を、回答へ渡す上位5件程度へ絞る。
- 標準経路はlocal-firstとし、ローカルrerankerを第一候補として検討する。
- OpenAIを使うrerankは、公開・匿名化データだけを対象にした比較オプションとする。
- RetrieverのRecallを損なわず、MRRと文脈精度が改善するか確認する。

### 回答不能判定

第一段階として`answer-or-abstain-v1`を実装し、ローカルQwenへ`answer`または`abstain`を
明示選択させ、Python側でstrict JSONとcitationを検証する。abstainは正常結果であり、
回答本文・出典を一切表示しない。OpenAI APIはproduction判定へ使用しない。

この第一段階はmodel-based abstentionであり、Retriever score閾値による生成前gateではない。
評価でfalse answerが残る場合は、次段階として次を独立比較する。

- 関連性が不十分な場合はGeneratorを呼ばない。
- 閾値は開発用answerability setのPrecision / Recallから決定する。
- 引用検証およびmodel-based abstentionとは別責務の生成前判定とする。
- 固定holdoutを閾値調整へ使用しない。

### 重複除去とMMR

- 同一URL・同一節の占有数を測定し、明らかな重複候補を抑制する。
- MMRは関連候補が取得できた後の多様性調整として使う。
- BM25やRerankより先に導入しない。
- 多様性向上によってfirst relevant rankやHit@Kが悪化しないか確認する。

### Query Rewrite

- 元質問を捨てず、元質問と書き換え質問の両方で検索して順位統合する。
- Python識別子、添字、記号を保持し、`isatty`や`argv[0]`の破壊を防ぐ。
- 外部providerを使う場合、質問文が外部送信されることを明示する。
- Rewriteなし、ローカルRewrite、任意の外部Rewriteを分けて比較する。

### 合成質問・生成タイトル・field別Embedding

- 原文を合成文で置き換えない。
- ページタイトル、節タイトル、本文、合成質問を別フィールドとして比較する。
- 全8,677チャンクへ適用する前に、main、isatty、EOFErrorなどの失敗領域で
  小規模実験する。
- 生成物のモデル、prompt、revisionと原文との対応をmanifestへ記録する。

### Embeddingモデル比較

- 同じchunk、metadata、質問、Top-Kで比較する。
- 日本語意味類似度だけでなく、コード識別子保持、速度、モデルサイズを評価する。
- indexをモデルごとに分離し、異なるEmbeddingのindexを混用しない。

### AnswerGenerator比較

- Retrieverを固定してからQwenと任意のOpenAI AnswerGeneratorを比較する。
- Faithfulness、Answer Relevancy、引用妥当性、不要引用、失敗率を比較する。
- Generator比較をRetriever改善の代用にしない。

Final quality sprint v2の固定context比較では、Qwen3-8B non-thinkingが
Qwen3-4B baselineよりfalse answerとfalse abstentionを減らし、必須事実の
coverageを改善した。一方で、`argparse`の誤った補足、`main()`配置の誤解、
`aclosing()`やdescriptor precedenceの不足は残った。したがって大きいGenerator
だけで品質問題が解決したとはせず、pinned Qwen3-8BをPhase B winnerとして
出力制約・組合せ評価へ固定する。既存recommended-v1とproduction defaultは
最終untouched評価まで変更しない。

同スプリントの`two-stage-answerability-v1`では、回答可否をtoken trieで
`answer` / `abstain`へ厳密制約し、回答文をJSON stringから分離した結果、
`ensure_ascii=False`を含む最終contract failureを0へ減らせた。ただし、
正しい候補がTop-5にないargparse質問へ資料外の設定を生成したため、構造制約を
semantic correctnessの代替にしない。exact choice、回答生成、引用finalizationを
独立して測り、最終profileはuntouched setまで凍結する。

`evidence-first-v1`ではsource本文とのexact substring一致まで要求したが、
paraphraseを大量に拒否してcontract failureが増えた一方、質問との関連性は保証できず、
資料外の社内規約・alias回答も生じた。この方式はGate不合格とし、extractive一致を
groundingやanswerabilityの十分条件として扱わない。将来再検討する場合も、検索候補の
妥当性、evidence relevance、answer completenessを独立に評価する。

## 6. 評価指標

### Retriever

- Hit@1 / Hit@3 / Hit@5 / Hit@10
- MRR@10
- 質問ごとのfirst relevant rank
- 平均・中央検索時間
- query_type別指標
- topic別指標
- Top-10失敗質問と取得結果

### 重複・多様性

将来、次を追加する。

- unique URL count @10
- unique section count @10
- 同一URLの最大占有数 @10

### RAG

将来、次を追加する。

- Contextual Precision
- Contextual Recall
- Contextual Relevancy
- Faithfulness
- Answer Relevancy
- 引用妥当性
- 不要引用数
- 本文中URL数
- 回答不能判定のPrecision / Recall

LLM-as-a-Judgeの値は絶対的な正解とみなさず、固定rubricによる人手確認と併用する。
Judgeのモデル、revision、prompt、温度、実行日時も保存する。

## 7. 採否基準

各改善は同一評価セット・同一コーパスでbaselineまたは直前の採用方式と比較する。
最低限、次を確認する。

- Hit@5またはMRR@10が改善する。
- exact_identifierだけを最適化せず、conceptualを大きく悪化させない。
- Top-10失敗質問が改善するか、新しい重大な失敗を生まない。
- 平均・中央・必要に応じて上位percentileの検索時間がCLI利用上許容できる。
- 出典URL、ページタイトル、節タイトルのmetadata安全性を壊さない。
- Qwenへ渡す文脈量が入力token上限に収まる。
- 実装・依存・運用の複雑度に対して改善効果がある。

単一の総合値だけで自動採用しない。全体値、query_type別、topic別、主要失敗質問、
検索時間を併記し、採用・不採用・保留の理由を実験summaryへ残す。評価セットへ
過適合していないか確認するため、将来は固定開発セットと保留セットの分離も検討する。

## 8. OpenAI APIの境界

- 本体はlocal-firstとし、OpenAI APIは任意providerとして分離する。
- Python公式文書や公開HTMLでは、比較実験に利用できる。
- 社内秘文書、個人情報、契約上送信できないデータは組織の承認なしに送信しない。
- Query Rewriteでは質問文が外部へ送信される。
- Rerankでは質問と候補本文が外部へ送信される。
- Embeddingでは対象文書全文が外部へ送信される。
- Generatorでは質問と選択チャンクが外部へ送信される。
- 評価では質問、回答、期待回答、根拠文書が外部へ送信される。

外部providerの有効化は明示設定とし、未設定時に暗黙送信しない。送信データ、保存、
retention、地域、費用、rate limitを実験前に確認する。

## 9. 再現性

今後の評価JSONには、可能な限り次を記録する。

- Git commit
- questions JSONLのSHA-256
- processed JSONLのSHA-256
- index manifestとindexのSHA-256
- Embeddingモデル名とrevision
- chunk size / overlap
- 実行日時
- Python、Sentence Transformers、FAISSのバージョン
- Retriever方式とパラメータ
- Top-K、RRF、MMR、rerankerなどの設定
- 実行deviceと検索時間

入力SHAが異なる結果を同一条件の比較として扱わない。評価JSONは大きな実データと
同じデータルートへ保存し、Gitには数値とprovenanceを含む小さなsummaryだけを置く。

## 10. 対象外

現段階では次を実装対象にしない。

- PDF・Word対応
- JavaScriptレンダリング
- 認証付きクロール
- 複数サイトの権限管理
- 量子化の再検討
- GUI
- Web API
- LangChain Communityへの全面移行

これらが必要になった場合も、現在のRetriever評価と回答品質評価を混ぜず、別の要件と
評価計画を作成する。

## 11. Holdout・grounding・answerability checkpoint

`evaluation/questions.jsonl`の26問はBM25 tokenizer、`rrf_k`、`candidate_k`の選択に
使ったdevelopment benchmarkであり、独立test setではない。
`evaluation/holdout_questions.jsonl`は未使用質問として固定し、Retrieverの
parameter調整や正解ラベル変更には使わない。

評価は次の3層へ分ける。

1. Retriever ranking: 正解URLのHit@KとMRR
2. Citation形式: marker、番号、URL信頼境界、fail-closed
3. Semantic grounding: 引用チャンクによる主張支持とrequired factsの完全性

Citation形式の成功だけでsemantic groundingを成功とみなさない。answerability setの
false answer、false abstention、unanswerable時の出典表示を、将来の生成前回答不能
判定の設計に使う。

OpenAI judgeは任意で、公開Python文書だけを対象とする。Retriever、Embedding、
AnswerGenerator、Query Rewrite、Rerankには使わない。judge値は絶対的正解ではなく、
同一model/prompt/schemaでの比較と人手確認を必要とする。

holdoutではHybridのHit@10が改善したがHit@5はDenseと同値で、RAG品質では正解source
Top-5がDenseを下回った。このためCLI既定はDense、Hybridはopt-inを維持する。

Answerabilityではunanswerable 6問中、strict abstentionはDense 1問、Hybrid 0問で、
残りはfalse answerだった。次のproduction課題は明示的な`answer`／`abstain`契約とする。
Parent Documentはその後の独立Retriever実験とし、Rerank、MMR、Query Rewriteはまだ
実装しない。

## 12. Judge v3のevidenceと導出責務

Semantic groundingを次の独立した責務へ分ける。

1. Judge raw evidence: answer relevance、faithfulness、citation support、completeness、
   unsupported claims、missing required facts
2. Local groundedness: faithfulnessとcitation supportが各3以上かつunsupported claimsなし
3. Local coverage: required factsとcompletenessによるcomplete／partial／insufficient
4. Local answerability: datasetのanswerable、実際のabstained、導出groundedによるlabel

Groundednessは回答中の主張が資料で支持されるかを表し、coverageはrequired factsの
充足度を表す。両者を同じbooleanへ畳み込まない。Coverageは正しさではないため、
`complete`かつfalse answer、または`partial`／`insufficient`かつgroundedも成立し得る。
Groundednessはfaithfulness、citation support、unsupported claims、coverageはcompletenessと
missing required facts、answerability labelはanswerable、abstained、groundedから導出する。
Judgeモデルはraw evidenceだけを返し、3つのderived値はPython側で決定的に導出する。

Revisionは`rag-grounding-v3`、`rag-judge-evidence-v3`、
`rag-derived-evaluation-v1`とする。v1、v2、v3は責務とrubricが異なるため直接混在
させない。v1はscore方向が曖昧で、v2は方向を固定したがgroundednessとcompletenessを
モデル出力で混同した。v3を今後の正式方式とし、v1、v2結果は改善履歴としてGit管理外に
保持する。Structured Outputの形式適合と意味的妥当性は別であり、OpenAI Judgeは
ground truthではないため人手評価と併用する。

Judge v3実測はDense grounded 5/10、Hybrid 6/10、人手groundingは両方5/10だった。
一方、正解source Top-5と引用はDense 8/10、Hybrid 7/10である。Hybridを既定化する
一貫した根拠はなく、Dense既定とHybrid opt-inを維持する。全12新規APIケースは最終的に
成功したが、retry上限2に対するケース別retry回数は未記録で算出不能である。この点は
必要になった場合のinstrumentation対象であり、rubric変更理由にはしない。

Judgeへ送信する対象は公開Python公式文書に限定した。社内秘密文書へ同じ方式をそのまま
適用せず、データ取扱い、retention、権限を別途設計する。

固定holdoutはこのcheckpoint以後もparameter tuning、閾値調整、rubric選択、正解ラベル
変更へ使用しない。新しい調整にはdevelopment setを使い、holdoutは最終確認に限定する。
将来のrelease判断前には、さらに別の未使用blind setによる確認が望ましい。
Retriever既定、Qwen、生成prompt、citation検証、URL信頼境界はJudge責務分離では変更
しない。

## 13. Answerability Contract checkpoint（2026-07-30）

`legacy`と`answer-or-abstain`をask、chat、評価runnerで明示切替できるようにした。
現時点の既定値は`legacy`を維持する。新方式はtop-level 3 keyだけのJSON、固定status、
固定reason、既存citation finalizerを組み合わせ、abstainを`AbstainedAnswer`として
`CitedAnswer`から型で分離する。検索0件もGeneratorを呼ばない正常abstainである。

RunPod RTX 4090で同一Qwen revisionによるreference evaluationを実施した。品質・機能評価は
有効だが、速度、load時間、VRAMは4090固有値として扱う。L4未実測はhard blockerでも
merge条件でもなく、別GPUでの再確認は任意のcross-device compatibility checkとする。

新方式はfalse answerを大幅に減らした一方、Denseのfalse abstentionとHybridのcontract
failureが残る。このためexperimental opt-inとして採用し、品質・安定性を理由に既定化は
保留する。`legacy`は後方互換、比較、rollback用に維持するが、安全なproduction既定と
認定したわけではない。Retriever既定はDense、Hybridはopt-inを維持する。次はParent
Documentを比較し、false abstentionとcoverageを改善できるか評価する。score-based生成前
gate、Rerank、MMR、Query Rewriteはこのcheckpointでは実装しない。

## 14. Existing-chunk Parent Retrieval checkpoint（2026-07-30）

`existing-chunk-parent-v1`として、既存8,677 `SearchChunk`をparent、400文字・overlap
100文字の23,116 childを検索専用単位にした。child candidateはdevelopment 26問で
30と60を比較し、全問でTop-10に必要なunique parentが確保され、rankingも同値だった
ため30へ固定した。child IDやoffsetはGeneratorへ渡さず、citationはparent metadata
だけから構築する。

HoldoutではParent Dense/HybridともHit@5とMRRを改善し、Dense Answerabilityのfalse
abstentionも2/6から1/6へ改善した。一方、Parent DenseのRAG contract failureが0から
1件へ増え、Hybrid Answerabilityのfalse abstentionと正解source引用は悪化した。
全Parent RAGで同じ`ensure_ascii=False`ケースが最終contract failureになり、
answerable質問の意味的誤回答も残った。

このため実験実装とrunnerは保持するが、production CLIへ`context-mode`は追加しない。
baseline artifactは上書きせず、child artifactと詳細JSONはdata-root配下へ分離した。
OpenAI APIは使用していない。完全な見出し単位`DocumentSection` parentは、一般HTML
ingestionがsection正本を永続化した後の別評価とする。完全な数値とprovenanceは
[`evaluation/parent_document_retrieval_summary.md`](../evaluation/parent_document_retrieval_summary.md)
を参照する。今回の不採用をParent Document概念全体の否定とはせず、直ちにRerankへ
進まずにPython専用版をこのcheckpointで固定し、次は一般HTML版へ移行する。

## 15. Final quality sprint checkpoint（2026-07-31）

復元したraw Python HTMLが保護済8,677 chunk baselineをbyte-identicalに再生成する
ことを確認した上で、Local Reranker、完全な`DocumentSection` Parent、固定設定同士の
組合せを連続評価した。OpenAI APIとOpenAI Judgeは使用していない。

Local Rerankerは2モデル、Dense/Hybrid候補、candidate k 20/30のDevelopment限定比較で、
Apache-2.0のmMARCO MiniLM、Hybrid、k 30を選定した。HoldoutはHit@5 0.8333、MRR@10
0.8519、Answerabilityはfalse answer 0・false abstention 0、RAGはvalid answer 8/10、
正解source引用8/10となった。Hybrid input baselineからcontract failureを増やさず、
Gateを満たした。

完全Section Parentは2,766 `DocumentSection`を正本とし、全文がpromptへ収まらない場合は
matched child中心の段落windowを決定的に使う`section-parent-v1`とした。400/100 +
HybridをDevelopmentだけで固定し、Holdout Hit@5は0.9444まで改善したが、Development
MRRはHybrid baselineを下回り、Answerability false abstentionが1/6へ増え、正解source
引用も4/6へ低下したため不採用とした。組合せもfalse abstention 2/6、RAG valid answer
7/10となり不採用である。Retrieval hitとsemantic correctnessを同一視しない。

トーナメント勝者はRerank onlyとし、production既定を変更せず
`python -m python_doc_rag chat --profile recommended`で1コマンド起動できる明示profile
として提供する。profileなしのDense、legacy、rerankerなしは後方互換のまま維持する。
完全Section Parentはproduction CLIへ接続しない。Python固有parserからsection正本、
child mapping、token-aware resolverまでの境界は研究資産として固定できたため、次は
一般HTML ingestionで同じ不変条件を満たすsection生成へ進める。ただしPythonでの
Section Parent不採用を覆すものと仮定せず、新しいcorpusとblind setで再評価する。

全数値、既知難問、実行環境、性能、artifact SHA-256は
[`evaluation/final_quality_sprint_summary.md`](../evaluation/final_quality_sprint_summary.md)
を正本とする。

## 16. Final quality sprint v2 Candidate Recall checkpoint（2026-07-31）

recommended-v1でreranker候補外だった`isatty()`を含むcandidate recallを、rerank後の
Hitだけと分離して計測した。parser由来chunkのidentifier、section title、page title、
bodyを独立rankとしてRRF統合するsidecar方式と、MITのBGE-M3／multilingual E5-baseを
Developmentだけで比較した。探索はfield 4設定・Embedding 2モデルで打ち切った。

Developmentではequal field + BGE-M3がRecall@30 0.9615、rerank後Hit@5 0.9615、
MRR@10 0.7179で選定された。`isatty()`はcandidate rank 6へ入り、optional multi-query
の実施条件は成立しなかった。固定HoldoutではRecall@30はrecommended-v1と同じ0.9444、
Hit@5は0.8333から0.8889、MRR@10は0.8519から0.8611へ改善したためGate Aを通過した。
ただしRecall@10は低下しており、Retriever単独の勝利を最終profile採用とは扱わない。

baseline artifact、既存recommended-v1、CLI既定、Qwen、answer contract、citation URL
境界は変更していない。次のGenerator比較では質問ごとのselected context tupleを一度
だけ固定し、Retriever差を混ぜない。完全な数値とartifact SHAは
[`evaluation/final_quality_sprint_v2_summary.md`](../evaluation/final_quality_sprint_v2_summary.md)
を正本とする。OpenAI APIとOpenAI Judgeは使用していない。

## 17. Final quality sprint v2の凍結判断（2026-07-31）

Candidate Recall winnerのcontextを固定してQwen3-4B、Qwen3-8B、Qwen2.5-7Bを比較し、
false answer 0、contract failure非増加、fact coverage改善を満たしたQwen3-8Bを選定した。
自由文JSONの不安定性にはexact token-trieで`answer`／`abstain`だけを選ぶtwo-stageも
実装し、既存RAG setのcontract failureを1から0へ減らした。ただし根拠sourceが候補外の
argparseでunsupported answerが残るため、構造成功を意味成功とは扱わない。
Evidence-firstはfalse answer 2と大量のcontract failureを生じ、明確に不採用とした。

設定commit後に、既存質問・既知難問と重ならないFinal Untouched RAG 12問および
Answerability 8問を作成・commitし、recommended-v1と新finalist 2件を一度だけ評価した。
3候補ともコーパス外`InterpreterPoolExecutor`へ1/4のfalse answerを出したことは重要な
未解決制約である。ここを最終結果後にhard-codeせず、負結果として固定する。

安全性上位3条件は同率だった。その次の優先順位で、equal technical-field + BGE-M3、
mMARCO、Qwen3-8B、answer-or-abstain-v1の直接方式が、Answerability false abstention
0/4、RAG valid answer 12/12、正解source引用11/12、fact coverage 24/29で勝った。
two-stageはfinal contract上の追加利益がなく、statistics.kdeをabstainして10/12引用へ
低下したため採用しない。

この勝者を`recommended-v2`とし、`recommended` aliasをv2へ更新する。
`recommended-v1`はrollback用に不変で残し、profileなしのDense・legacy既定も維持する。
v2のBGE-M3 indexとsymbol sidecarはdata-root相対pathで明示し、`check --profile`が
モデルロードなしに件数、次元、revision、SHA-256を検査する。Python専用品質改善は、
上記false answerを既知の制約として凍結し、次は一般HTML ingestionへ移行できる。

## 18. Configurable HTML ingestion checkpoint（2026-08-01）

Python固有のRAG後半を再調整せず、SiteConfig、Loader、SourceDocument、Parser、
DatasetArtifactLayoutを分離した。PythonSphinx adapter経由の全量出力は8,677 chunk、
SHA-256 `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869`
で保護baselineとbyte-identicalである。recommended-v1/v2、profileなしの既定、model pin、
answer contract、citation URL境界は変更していない。

generic datasetではalgorithm profileのmodel/revision/dimension/prefixと、corpus固有の
metadata、symbol、FAISS SHAをdataset-local manifestへ分離した。uv文書の2 start URLを
40ページ上限で取得すると31ページ、184 section、270 chunkとなり、同じ固定BGE-M3、
equal field RRF、mMARCO、Qwen3-8B、answer-or-abstain-v1をparameter tuningなしで適用
できた。site固有selectorはTOMLだけに閉じている。

8問のportability smokeはblind benchmarkではない。false answer 0、false abstention 0、
contract failure 1で、成功5回答は全件正解sourceを引用した。1件は正解sourceがTop-5に
あっても契約失敗したため、別corpusへ移植できることと品質最適性を同一視しない。
OpenAI APIとOpenAI Judgeは使用していない。今後別サイトへ広げる場合も、site configを
固定してから質問を作り、質問結果によるselector/retrieval tuningを避ける。
