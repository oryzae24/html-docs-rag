# RAG品質評価

## 条件

- 実行日: 2026-07-29
- 質問数: 10
- 質問: holdoutからgrounding確認に適した質問を選定
- Generator: `Qwen/Qwen3-4B-Instruct-2507`
- device/dtype: CUDA / bfloat16
- Top-5、入力上限8,192 token、最大生成512 token
- DenseとHybridでGenerator、prompt、citation検証は同一
- 詳細JSON（Git管理外）:
  - `<data-root>/evaluation/rag_dense.json`
  - `<data-root>/evaluation/rag_hybrid.json`

## 決定的なローカル指標

| Metric | Dense | Hybrid |
| --- | ---: | ---: |
| 生成成功 | 10/10 | 10/10 |
| Citation形式成功 | 10/10 | 10/10 |
| Fail-closed | 0/10 | 0/10 |
| 回答本文中URL | 0 | 0 |
| 不正citation番号 | 0 | 0 |
| 表示出典数 | 13 | 13 |
| 正解source Top-5 | 8/10 | 7/10 |
| 正解source表示・引用 | 8/10 | 7/10 |
| 無関係sourceだけを表示 | 2/10 | 3/10 |

HybridはこのセットでDenseより正解source Top-5と正解source引用が1問少なかった。

## 人手Semantic grounding

各回答について次を人手確認した。

1. `required_facts`をすべて含む
2. 実際に引用されたチャンクが各主張を直接支持する
3. 矛盾した主張や資料外の補完がない

この厳格な全条件を満たした回答はDense 5/10、Hybrid 5/10だった。これは再現可能な
自動judge値ではなく、少数ケースの人手checkpointである。

Denseの主な失敗:

- `contextlib.aclosing()`で`await`と早期終了時の保証が欠落
- `dataclasses.replace()`で「いいえ」と「呼ばれる」が同一回答内で矛盾
- loggingで実効level探索と`propagate`によるhandler伝播を混同
- descriptor回答は正しいが、実際の引用チャンクが優先順位を直接支持しない
- argparseの相互排他groupを否定する誤回答

Hybridの主な失敗:

- `dataclasses.replace()`の矛盾
- descriptorで非データデスクリプタ側の必須事実が欠落
- TaskGroupの例外集約を誤り、正解節もTop-5外
- argparseの相互排他groupを否定
- `json.dumps(ensure_ascii=False)`ではなくCLIの`--no-ensure-ascii`を回答

引用形式10/10はsemantic groundingを保証しない。形式検証はcitation markerとURL信頼
境界を守るが、取得節の正しさ、引用による主張支持、回答の完全性は別に評価する。

## OpenAI judgeの改善履歴

評価専用の`OpenAIResponsesJudge`をoptional dependencyとして実装した。Responses API、
strict JSON Schema、`store=False`、model名必須、限定retry/timeoutを使用し、ケース単位
失敗を記録できる。Qwen保存結果へ後から適用できる。

- 後続v1 smoke: 1問を実行し、不整合を検出
- v1 judge model: `gpt-5-mini-2025-08-07`
- v1 prompt revision: `rag-grounding-v1`
- v1 schema revision: `rag-judge-result-v1`

LLM-as-a-Judgeは絶対的な正解ではない。モデル、prompt、schemaで値が変わるため、
同一設定で方式間を比較し、少数ケースを人手確認し、開発用指標として使う。

## 判断

Hybridはholdout検索のTop-10を改善したが、このRAG評価では正解source取得・引用が
Denseを下回り、厳格な人手groundingも改善しなかった。Denseを既定として維持し、
Hybridの既定化は保留する。

## OpenAI judge v2校正

v1 smokeでは、正解回答に対して`grounded=true`、`supported_answer`、未支持・欠落なし
である一方、4品質scoreがすべて0になる不整合を確認した。v1はscore範囲だけをschemaで
指定し、採点方向と各段階をpromptへ明記せず、項目間の意味的整合性も検証していなかった。
このv1ファイルは保持し、v2集計へ混ぜない。

v2では4品質scoreを次に統一した。

- 0: 全く満たさない
- 1: ほとんど満たさない
- 2: 一部満たす
- 3: おおむね満たす
- 4: 完全に満たす

- prompt revision: `rag-grounding-v2`
- schema revision: `rag-judge-result-v2`

Strict JSON Schema通過後、`grounded`と低いfaithfulness／citation support、
`supported_answer`と非grounded、本文ありの`correct_abstention`、materialな未支持主張と
grounded、全required facts欠落と高completenessを不整合として検出する。不整合結果は
`judge_error`へ保存し、judge完了数・平均・grounded率・label件数から除外して処理を継続する。

Structured Outputの形式適合は評価内容の妥当性を保証しない。Judgeは開発用指標であり、
同一model／prompt／schemaで比較し、人手確認と併用する。v1とv2の結果を直接比較・混在
させない。

校正3ケースと同一1問v2 smokeの実API結果は、v1を上書きせずdata-root配下の別JSONへ
保存した。v1はscore方向が曖昧で、v2は方向を固定したがgroundednessとcompletenessを
モデル出力で混同した。v3はraw evidenceとPython側derived判定を分離する。v1、v2、v3を
直接混ぜず、今後の正式な評価方式はv3とする。既存v1、v2結果は改善履歴としてGit管理外
に保持する。

## OpenAI judge v3責務分離

v2全件評価では、回答中の主張が資料で支持されていても、required factsの欠落を理由に
`grounded=false`または`false_answer`となる部分回答があり、groundedness、completeness、
answerability labelの責務混在が確認された。v2の実測結果は方式改善前の履歴として
data-root配下へ保持し、v3値へ書き換えない。

v3のJudgeモデルは次のraw evidenceだけを返す。

- answer relevance、faithfulness、citation support、completeness（各0～4）
- concise reason
- unsupported claims
- missing required facts

`grounded`はfaithfulnessとcitation supportが各3以上でunsupported claimsが空の場合に
Pythonで導出する。completeness、missing required facts、answer relevanceはgroundedの
直接条件にしない。Coverageは必要な論点の充足度であって正しさではないため、
`complete`な誤回答や、groundedかつcoverage `partial`／`insufficient`の回答も成立し得る。
Coverageはcompletenessとmissing required facts、answerability labelはanswerable、
abstained、groundedからPython側の決定規則で導出する。

- prompt revision: `rag-grounding-v3`
- schema revision: `rag-judge-evidence-v3`
- derived result revision: `rag-derived-evaluation-v1`

raw失敗とderived semantic errorは分けて保存し、いずれかがあるケースを完了件数と集計
から除外して後続質問を継続する。OpenAI Judgeは開発用指標であり、人手groundingと併用
する。Structured Outputの形式適合は意味的な妥当性を保証しない。

## OpenAI Judge v3実測

- 実行日: 2026-07-29
- Judge model: `gpt-5-mini-2025-08-07`
- prompt revision: `rag-grounding-v3`
- raw schema revision: `rag-judge-evidence-v3`
- derived revision: `rag-derived-evaluation-v1`
- 全12新規APIケースは最終的に成功
- retry上限: 2
- ケース別retry回数: 未記録のため算出不能。将来必要ならinstrumentation対象

| Raw metric | Dense | Hybrid |
| --- | ---: | ---: |
| Answer relevance平均 | 3.1 | 2.8 |
| Faithfulness平均 | 2.7 | 3.1 |
| Citation support平均 | 2.9 | 3.2 |
| Completeness平均 | 3.0 | 2.7 |
| Unsupported claims | 10 | 5 |
| Missing required facts | 6 | 9 |

| Derived metric | Dense | Hybrid |
| --- | ---: | ---: |
| Grounded | 5/10 | 6/10 |
| Coverage complete | 6 | 5 |
| Coverage partial | 2 | 3 |
| Coverage insufficient | 2 | 2 |
| Supported answer | 5 | 6 |
| False answer | 5 | 4 |
| False abstention | 0 | 0 |
| Correct abstention | 0 | 0 |
| Raw／derived error | 0 | 0 |

| Token usage | Dense | Hybrid |
| --- | ---: | ---: |
| Input tokens | 40,890 | 42,705 |
| Output tokens | 11,272 | 10,781 |
| Total tokens | 52,162 | 53,486 |
| 1問平均total | 5,216.2 | 5,348.6 |

Denseはanswer relevance、completeness、正解source Top-5、正解source引用で優位だった。
Hybridはfaithfulness、citation support、unsupported claimsの少なさで優位だった。
OpenAI JudgeのgroundedはDense 5/10、Hybrid 6/10、人手groundingは両方5/10である。
質問単位の人手labelは保存されていないため、Judgeとの質問単位一致率は算出しない。

Judge v3は`complete + false_answer`、`partial + supported_answer`、
`insufficient + supported_answer`をいずれも有効に扱った。Hybridを既定化する一貫した
根拠はなく、Denseを既定、Hybridをopt-inとして維持する。

## Provenance

- evaluation definition commit:
  `5c2f40dd6fd822f1b65e251441d11469c5e410de`
- source code parent commit:
  `4e608ba6a70413c3db1b027c23ff1e1f2774f4b5`
- holdout questions SHA-256:
  `f71870a240e7a98ca05716e3e1d8f4d09b2ff856cdd93f608cd6ebeca86ffc36`
- RAG quality questions SHA-256:
  `585cc4200f697472fe8dd31d4e5b35ce488f2b04a5202e2937cc88ba4bb82de9`
- answerability questions SHA-256:
  `d14734bc35967482e9d54a0300c929182a08371c03b1e8fe8ccc91893f9788b6`
- Qwen model:
  `Qwen/Qwen3-4B-Instruct-2507`
- Qwen revision:
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- Embedding:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dense: 既存FAISS、Top-5
- Hybrid: Dense＋文字2-gram BM25、`rrf_k=10`、`candidate_k=30`、同一重み
- 実行日: 2026-07-29
- `rag_dense_openai_v3.json` SHA-256:
  `ba2bfa6a86ddcd73250559e2aecb72923d3696887843e553337158e495def9d7`
- `rag_hybrid_openai_v3.json` SHA-256:
  `25897371d328657a22821fd4a4b366ed62fcd32cc76c95f55fee4cf177a62793`
- `rag_openai_v3_comparison.json` SHA-256:
  `b4b54dd11829b98e33c700e38fd3a891a7f69765769e17e519b0ab08a1eadfbc`
- `rag_openai_v3_comparison.md` SHA-256:
  `cf0fa26ea5a522e3f86e4bdcfddd349d15ae6c35aa194472a37701beb170737d`

上記詳細結果はGit管理外であり、APIキーは保存されていない。APIへ送信した対象は
公開Python公式文書だけである。社内秘密文書へ同じ方式をそのまま適用しない。
