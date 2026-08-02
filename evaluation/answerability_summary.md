# Answerability評価

## 条件

- 実行日: 2026-07-29
- 質問数: 12
- answerable: 6
- unanswerable: 6
- Generatorと実行設定はRAG品質評価と同一
- 詳細JSON（Git管理外）:
  - `<data-root>/evaluation/answerability_dense.json`
  - `<data-root>/evaluation/answerability_hybrid.json`

unanswerableはIDE操作、組織固有規約、外部ライブラリ、Python 3.14機能、個別shell
環境、存在しないAPIから構成した。

## 結果

| Metric | Dense | Hybrid |
| --- | ---: | ---: |
| 全体の生成成功 | 11/12 | 12/12 |
| Citation形式成功 | 11/12 | 12/12 |
| Fail-closed | 1/12 | 0/12 |
| Answerable回答成功 | 6/6 | 6/6 |
| False abstention | 0/6 | 0/6 |
| Unanswerableのstrict abstention | 1/6 | 0/6 |
| False answer | 5/6 | 6/6 |
| Unanswerableで出典表示 | 5/6 | 6/6 |

ここでstrict abstentionは、固定の空検索messageまたはcitation検証によるfail-closedを
機械判定した値である。専用の回答不能出力schemaはまだない。

Qwenは「資料に記述がない」「回答できない」と本文で述べながら、無関係な引用番号と
出典を表示する場合があった。現行の決定的指標では、citation契約を通過して通常回答
として返されたものをfalse answerへ数える。自由文から拒否を推測する不安定な
heuristicは追加していない。

DenseのVS Code質問は2回ともcitation検証に失敗しfail-closedしたが、これは専用の
回答不能判定ではない。Hybridでは同じ質問にもcitation付き回答を返した。さらに組織
規約やshell aliasについて、取得資料から根拠のない具体値を回答する例があった。

## 次の回答不能判定への要件

- 生成前にretrieval evidenceの十分性を判定する
- `answer`と`abstain`を機械判定できる明示schemaを定義する
- abstain時は無関係な出典を表示しない
- answerableのfalse abstentionとunanswerableのfalse answerを別々に最適化する
- holdoutで閾値や判定器を評価し、同じデータで調整しない
- fail-closedと意味的な回答不能判定を区別する

現時点でproduction挙動は変更していない。Parent Documentへ進む前に、少なくとも
回答不能判定の設計と評価contractを決める必要がある。

次のproduction課題は、`answer`と`abstain`を明示する契約である。Parent Documentは
その後の独立Retriever実験とし、Rerank、MMR、Query Rewriteはまだ実装しない。
Answerability 12問にはOpenAI Judgeを実行していない。

## Judge v3との責務境界

Judge v3ではモデルにanswerability labelを直接選ばせない。評価データの`answerable`、
実際の`abstained`、raw evidenceからローカル導出した`grounded`を入力として、Pythonで
次のように決定する。

- answerableかつ回答ありかつgrounded: `supported_answer`
- answerableかつ回答ありかつ非grounded: `false_answer`
- answerableかつabstain: `false_abstention`
- unanswerableかつabstain: `correct_abstention`
- unanswerableなのに回答: `false_answer`

Abstainは既存の`generation_attempts == 0`と固定の空検索messageを使用し、自由文から
推測しない。Groundednessとrequired factsのcompletenessは分離し、支持された部分回答は
`grounded=true`、coverage `partial`、`supported_answer`になり得る。OpenAI Judgeは開発用
指標であり、人手評価と併用する。v1、v2、v3結果は直接混在させず、既存実測値も変更
しない。

## Provenance

- evaluation definition commit:
  `5c2f40dd6fd822f1b65e251441d11469c5e410de`
- source code parent commit:
  `4e608ba6a70413c3db1b027c23ff1e1f2774f4b5`
- answerability questions SHA-256:
  `d14734bc35967482e9d54a0300c929182a08371c03b1e8fe8ccc91893f9788b6`
- Qwen model:
  `Qwen/Qwen3-4B-Instruct-2507`
- Qwen revision:
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- Embedding:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dense／Hybrid設定: RAG品質評価と同一
- 実行日: 2026-07-29
