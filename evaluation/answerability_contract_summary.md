# Answerability Contract評価

## 背景と安全境界

従来の`legacy`方式は検索が1件以上なら引用付き自由文を要求するため、回答不能質問にも
無関係な引用を付けたfalse answerを返した。これを直ちに削除せず、比較可能な
`answer-or-abstain` modeを追加した。CLI既定値は`legacy`のままである。

Contract revisionは`answer-or-abstain-v1`。Generatorへ渡すのはURL・環境固有pathを
除去した完成済みpromptと`max_new_tokens`だけであり、token計測した文字列をそのまま
生成へ使用する。出典URLは取得metadataからのみ構築する。初回違反時は、不正出力を
含めず同じselected chunk tupleで1回だけ再生成する。

## Model-visible JSON契約

```json
{"status":"answer","answer_text":"根拠に基づく回答。[S1]","reason":null}
```

```json
{"status":"abstain","answer_text":"","reason":"insufficient_evidence"}
```

Top-level keyは`status`、`answer_text`、`reason`の3件だけで、追加・欠落を許可しない。
`answer`は非空本文、`reason=null`、1件以上の正規引用を必要とする。`abstain`は空文字本文と
固定reasonだけを許し、回答・引用・URL・Markdown link・出典を一切表示しない。
検索0件はGeneratorを呼ばず、`no_retrieval_results`、`generation_attempts=0`の正常な
`AbstainedAnswer`を返す。

Python parserは前後空白だけを許し、invalid JSON、code fence、前置き・後書き、duplicate
key、top-level非object、必須key欠落、追加key、型不一致、unknown status/reason、NaN等の
非標準値をstable reason code付きで拒否する。Answer本文は既存
`finalize_cited_answer`へ接続し、malformed／範囲外citation、URL、Markdown linkを
fail-closedにする。

## 実測上の制約

実験環境はRunPodの`NVIDIA GeForce RTX 4090`（24,564 MiB）だった。新方式4条件は
同一Qwen revisionによる**RTX 4090 reference evaluation**として扱う。品質・機能評価は
有効だが、速度、load時間、VRAMは4090固有値であり、過去のL4・T4性能値と単純比較しない。
L4未実測はhard blockerでも今回のmerge条件でもない。別GPUでの再確認は任意の
cross-device compatibility checkとし、最終利用環境が変わった場合はsmoke testを推奨する。
既定化保留の理由はGPUではなく、後述する品質とcontract安定性である。

Legacy Answerabilityは質問SHA、Qwen revision、Retriever、Top-5、token上限、生成設定が
一致する2026-07-29の既存結果を再利用した。元JSONと複製先のSHA-256一致を確認した。
Legacyの旧schemaには新しいcontract・分離generation timing項目がないため、それらは
推測で補完しない。

- Qwen: `Qwen/Qwen3-4B-Instruct-2507`
- Qwen revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Transformers: `5.14.1`
- Sentence Transformers: `5.6.1`
- dtype: `bfloat16`
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Top-5、入力8,192 token、最大生成512 token、greedy
- Dense: 既存FAISS
- Hybrid: Dense＋文字2-gram BM25、`rrf_k=10`、`candidate_k=30`、同一重み
- OpenAI API／OpenAI Judge: 不使用

## Answerability 12問

| Metric | Dense legacy | Dense contract | Hybrid legacy | Hybrid contract |
| --- | ---: | ---: | ---: | ---: |
| Answerable回答 | 6/6 | 4/6 | 6/6 | 6/6 |
| False abstention | 0/6 | 2/6 | 0/6 | 0/6 |
| Unanswerable strict abstention | 1/6 | 6/6 | 0/6 | 6/6 |
| False answer | 5/6 | 0/6 | 6/6 | 0/6 |
| Unanswerableでsource表示 | 5/6 | 0/6 | 6/6 | 0/6 |
| 初回contract成功 | 旧schema | 12/12 | 旧schema | 12/12 |
| Retry | 旧schema | 0/12 | 旧schema | 0/12 |
| 2回失敗 | 1/12（citation） | 0/12 | 0/12 | 0/12 |

ここでanswer precision相当を「回答したケースのうちdataset上answerableだった割合」、
answer recall相当を「answerable 6問のうち回答した割合」と定義する。Denseはlegacyの
precision 6/11（54.5%）、recall 6/6（100%）から、contractのprecision 4/4（100%）、
recall 4/6（66.7%）となった。Hybridは6/12（50%）、6/6（100%）から6/6（100%）、
6/6（100%）となった。

Dense contractのfalse abstentionは`Counter.total()`と`typing.Protocol`で、どちらも
正解sourceはTop-5に存在した。Hybrid contractはこの12問でanswerable 6問をすべて回答し、
unanswerable 6問をすべてabstainした。

## RAG quality 10問（新方式）

| Metric | Dense contract | Hybrid contract |
| --- | ---: | ---: |
| Valid answer | 7/10 | 7/10 |
| Valid abstain（false abstention） | 3/10 | 2/10 |
| Contract failure | 0/10 | 1/10 |
| Unanswered total | 3/10 | 3/10 |
| Citation形式成功 / valid answer | 7/7 | 7/7 |
| 正解source Top-5 | 8/10 | 7/10 |
| 正解source引用 | 7/10 | 7/10 |
| 無関係sourceのみ | 0/10 | 0/10 |
| 初回contract成功 | 10/10 | 9/10 |
| Retry | 0/10 | 1/10 |
| 2回失敗 | 0/10 | 1/10 |

Hybridのunanswered 3件はfalse abstention 2件と2回contract違反によるcontract failure
1件からなる。
Denseはdescriptor、argparse、`ensure_ascii=False`でabstainした。HybridはTaskGroupと
argparseでabstainし、`ensure_ascii=False`で2回ともcontract検証に失敗した。一方、
`dataclasses.replace()`は両方式で矛盾なく`__post_init__()`が呼ばれると回答し、Hybridの
descriptor回答はデータ／非データdescriptorの優先順位を直接回答した。

この評価ではOpenAI Judgeを実行していないため、回答件数をsemantic正解件数とは呼ばない。
Citation形式成功も主張の完全なgroundingを保証しない。

## Contractと性能

新方式44ケース（Answerability 24＋RAG quality 20）の合計は、初回成功43/44（97.7%）、
retry 1/44（2.3%）、最終contract failure 1/44（2.3%）だった。詳細JSONから確認した
最終内訳はvalid answer 24件、valid abstain 19件、contract failure 1件、valid outcome
43件である。Citation形式成功は24/24 valid answer。19件のvalid abstainではanswer text
漏出0/19、source漏出0/19だった。

| 4090参考値 | Dense Answerability | Hybrid Answerability | Dense RAG | Hybrid RAG |
| --- | ---: | ---: | ---: | ---: |
| 検索 平均／中央 秒 | 0.0120 / 0.0052 | 0.0214 / 0.0154 | 0.0132 / 0.0048 | 0.0230 / 0.0146 |
| 生成 平均／中央 秒 | 0.5624 / 0.3551 | 0.6191 / 0.5646 | 1.3741 / 1.0084 | 1.4877 / 1.2175 |
| 全体 平均／中央 秒 | 0.5917 / 0.3775 | 0.6591 / 0.6004 | 1.4056 / 1.0315 | 1.5306 / 1.2531 |

これらはRTX 4090固有の参考性能値である。モデルload時間は質問別時間へ含めない。

## 採否判断

RTX 4090参考実測では、unanswerable false answerをDense 5/6から0/6、Hybrid 6/6から
0/6へ減らし、abstain時の本文とsourceを完全に遮断した。JSON契約は43/44で最終成功し、
既存のURL・citation信頼境界も維持した。一方、Dense Answerabilityではanswerableの
false abstentionが2/6（33.3%）。RAG qualityではDenseのfalse abstention 3/10、
Hybridのfalse abstention 2/10とcontract failure 1/10により、どちらもunanswered total
3/10だった。唯一のretry 1件は最終成功につながらず、標本数も小さい。実モデル評価GPUも
RTX 4090の1種類だけである。

したがって現時点では次の判断とする。

- 実装はexperimental opt-inとして採用する
- Dense／Hybridとも`answer-or-abstain`の既定値変更は品質・安定性を理由に保留する
- CLI既定は後方互換と比較・rollbackのため`legacy`を維持するが、安全なproduction既定と
  認定したわけではない
- RetrieverのCLI既定はDense、Hybridはopt-inを維持する
- answerableで直接根拠がTop-5にあるのにabstainする例を分析する
- score-based生成前gateはfalse abstentionを悪化させ得るため、今は追加しない
- 次にParent Documentを比較し、false abstentionとcoverageを改善できるか評価する

Parent Document、score-based gate、Rerank、MMR、Query Rewriteは未実装である。
OpenAIを使ったproduction判定も行わない。

## 詳細結果とprovenance

詳細JSONはrepository外のGit管理外data-rootへ保存した。

| File | SHA-256 | 備考 |
| --- | --- | --- |
| `answerability_legacy_dense.json` | `7066a537225b6485f187623a41444a33c94f46c496af2c505cc21f33a263fc45` | 既存`answerability_dense.json`と同一SHAで再利用 |
| `answerability_contract_dense.json` | `faa11214905f8554ee9e3f392e352abb730b3376ec899d0208c0e9858810ce35` | RTX 4090参考実測 |
| `answerability_legacy_hybrid.json` | `782f1d5d09cc24d8e4bddc4f4416ad97060bdf206563f65d164add7a069a0ca7` | 既存`answerability_hybrid.json`と同一SHAで再利用 |
| `answerability_contract_hybrid.json` | `db91fe72b5a38931c01d9277b6363061a64d2e835b57ed4adcfe76e94d34a782` | RTX 4090参考実測 |
| `rag_quality_contract_dense.json` | `9ee65b615b587dd3f0325d5fd268646633c7a88ee1038ce1edcb3edde061ed06` | RTX 4090参考実測 |
| `rag_quality_contract_hybrid.json` | `de5017f02ce6b14b80ed989e974940a102faa57404d46f61b4fb8e947f16fd08` | RTX 4090参考実測 |

- 開始branch: `feat/answerability-contract`
- 開始HEAD: `a4dff165f900ca5175eef4fc23f8cb6649a26884`
- Source code parent: `a4dff165f900ca5175eef4fc23f8cb6649a26884`
- Evaluation definition commit:
  `900ed8a84d4b133800535d0886e693f77da265fd`
- answerability questions SHA-256:
  `d14734bc35967482e9d54a0300c929182a08371c03b1e8fe8ccc91893f9788b6`
- RAG quality questions SHA-256:
  `585cc4200f697472fe8dd31d4e5b35ce488f2b04a5202e2937cc88ba4bb82de9`
- 実行日時（結果保存時刻）: 2026-07-30 19:47:12–19:47:13 UTC
- 実測GPU: `NVIDIA GeForce RTX 4090`
- VRAM: 24,564 MiB
- Driver: `580.159.04`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Transformers: `5.14.1`
- Sentence Transformers: `5.6.1`
- dtype: `bfloat16`
- Qwen: `Qwen/Qwen3-4B-Instruct-2507`
- Qwen revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Dense: 既存FAISS、Top-5
- Hybrid: Dense＋文字2-gram BM25、`rrf_k=10`、`candidate_k=30`、同一重み、Top-5
- OpenAI API／Judge: 未使用
- Retriever、Embedding、FAISS、chunk、corpus、Qwen生成設定: 変更なし
- 詳細JSONはGit管理外で、API keyや秘密値を含まない

上記の評価定義 commit を作成する前の監査では、production source と test の最終更新時刻は
実測結果の保存前であり、評価 runner も最初の詳細 JSON を保存する前に確定していた。
結果保存後の repository 変更は文書だけであり、実験時の code、test、runner は同 commit の
tree に評価定義として固定されている。詳細 JSON 自体には source commit が記録されて
いないため、この対応関係は上記の時刻と Git 差分の監査、および同 tree に基づく。
