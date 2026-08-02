# Existing-chunk Parent Retrieval v1評価

## 背景と範囲

Parent Document Retrievalの第一段階として、小さいchild chunkを検索し、回答生成と
citationには既存のcitation-ready `SearchChunk`を返す方式を実装した。revisionは
`existing-chunk-parent-v1`。

今回のparentは最大およそ1,000文字、overlap 150文字の既存FAISS metadata単位であり、
見出し全体の`DocumentSection`ではない。再クロール、HTML再解析、コーパス再生成は
行っていない。既存parentを正本にすることで、現在のURL信頼境界と評価資産を維持し、
将来は`ParentStore`の正本だけを完全な`DocumentSection`へ置き換えられる。

childは検索専用である。Generatorへ渡すのは解決済みparentの本文・ページタイトル・
節タイトルだけで、child ID、parent ID、child offset、検索診断、URLはpromptへ渡さない。
citation URLは従来どおりparent `SearchChunk.source_url`からPython側で構築する。
OpenAI APIおよびOpenAI Judgeは使用していない。

## Parent IDと検証

parent IDは`source_url`、`chunk_index`、`start_index`をkey順固定・空白なしのcanonical
JSONへ変換し、UTF-8のSHA-256を取った64桁lowercase hexである。

8,677 parentで次を確認した。

- parent ID衝突・duplicate: 0
- unresolved child: 0
- childとparentのURL不一致: 0
- parent text SHA-256不一致: 0
- JSONL再読込後のID変化: 0

`ParentStore`はparent JSONLを起動時に一度だけ読み、duplicate、missing parent、URL、
identity field、parent text hash、child range、child textをfail-closedで検証する。
同じURLの別chunkは別parentとして扱う。

## Child構築と設定選択

共通条件は既存の`RecursiveCharacterTextSplitter`、日本語separator
（空行、改行、句点、読点、空白、文字境界）、同一Embedding、
FAISS `IndexFlatIP`、L2正規化である。検索文字列は既存baselineと同じ
`page_title`、`section_title`、`text`のラベル付き連結。

| child設定 | child数 | 平均/parent | 最大/parent | index | metadata | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 / 50 | 44,378 | 5.114 | 11 | 65.01 MiB | 42.25 MiB | 57.90秒 |
| 300 / 75 | 30,698 | 3.538 | 9 | 44.97 MiB | 32.10 MiB | 37.65秒 |
| 400 / 100 | 23,116 | 2.664 | 8 | 33.86 MiB | 26.42 MiB | 27.472秒 |

baselineは8,677件、index 12.71 MiB、metadata 11.05 MiB。選定設定はbaseline比で
child数2.66倍、index 2.66倍、metadata 2.39倍である。選定buildのprocess peak CPU RSSは
2.13 GB、開始後増加は1.34 GB、PyTorch peak allocated VRAMは619.8 MBだった。
いずれもRTX 4090固有の参考値である。
選定artifactの実サイズはchild index 35,506,221 bytes、child metadata
27,696,108 bytesである。
選定child JSONLを同一入力・設定から別pathへ再生成し、SHA-256
`7d80dcdd56ab1ea2e2717dcb7d93d9c10703391b1af67c63182cd4fb1d65d425`
が一致するbyte-identical出力を確認した。

Development 26問のDense比較は次のとおり。candidate 30と60でranking指標は全設定とも
同値だった。

| child設定 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 / 50 | 0.2692 | 0.5769 | 0.6923 | 0.7692 | 0.4513 |
| 300 / 75 | 0.3462 | 0.5769 | 0.6538 | 0.7308 | 0.4799 |
| 400 / 100 | 0.3462 | 0.6154 | 0.6923 | 0.6923 | 0.4872 |

事前規則に従い、Hit@5をbaselineから悪化させずMRR@10が最も高い400/100を選んだ。
200/50はHit@10が高いがMRRの選択順位が後であるため採用しなかった。400/100の
candidate 30は全質問でunique parentが18件以上、平均23.62件あり、60と結果が同じ
だったため、少ない30を固定した。Holdoutはこの選択へ使用していない。

## Retriever評価

既存baseline JSONは質問SHA、baseline index SHA、Dense/Hybrid設定が一致したため再利用した。
Parentは400/100、child candidate 30へ固定した。Hybridは既存どおり日本語2-gram、
`rrf_k=10`、各Retriever candidate 30、同一重みであり、再調整していない。

### Development 26問

| 方式 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | 平均/中央 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Dense | 0.3846 | 0.5385 | 0.6923 | 0.8077 | 0.4972 | 0.0168 / 0.0092秒 |
| Parent Dense | 0.3462 | 0.6154 | 0.6923 | 0.6923 | 0.4872 | 0.0106 / 0.0040秒 |
| Baseline Hybrid | 0.5769 | 0.6538 | 0.6923 | 0.8846 | 0.6389 | 0.0251 / 0.0179秒 |
| Parent Hybrid | 0.4615 | 0.6154 | 0.6538 | 0.8077 | 0.5737 | 0.0196 / 0.0151秒 |

Denseは改善7、同順位11、悪化8問。Top-5へ`EOFError`が入り、`shell=True`が外れた。
Top-10へ`EOFError`が入り、4問が外れた。Hybridは改善5、同順位14、悪化7問。
Top-5へ`EOFError`が入り、`os.pipe()`と`KeyboardInterrupt`が外れた。conceptualの
Parent Dense Hit@5は0.5000から0.3750へ低下した一方、exact identifierは0.7000から
0.8000、operational MRRは0.4979から0.5729へ改善した。

Parent Denseのchild 30件は平均23.62 parentへ圧縮され、平均圧縮率は0.787、単一parentの
最大child占有数は4。Parent Hybridは平均24.42 parent、圧縮率0.814、最大4だった。

### Holdout 18問

| 方式 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | 平均/中央 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Dense | 0.5556 | 0.6667 | 0.7778 | 0.8333 | 0.6431 | 0.1967 / 0.0088秒 |
| Parent Dense | 0.6111 | 0.7778 | 0.8333 | 0.8889 | 0.7139 | 0.0090 / 0.0042秒 |
| Baseline Hybrid | 0.5556 | 0.6667 | 0.7778 | 0.9444 | 0.6492 | 0.1130 / 0.0208秒 |
| Parent Hybrid | 0.6111 | 0.8889 | 0.8889 | 0.9444 | 0.7222 | 0.0288 / 0.0243秒 |

Denseは改善5、同順位11、悪化2問。descriptorがTop-5へ入り、descriptorと
`re.fullmatch()`がTop-10へ入った一方、`typing.Protocol`はTop-10から外れた。
Hybridは改善6、同順位11、悪化1問。TaskGroupと`ensure_ascii=False`がTop-5へ入り、
Top-5/10から外れた質問はなかった。Holdout Hit@5は両方式とも改善しており、
大幅悪化はない。

## Qwen Answerability 12問

Qwenは`Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`、CUDA、bfloat16、Top-5、
入力8,192 token、最大生成512 token、`answer-or-abstain-v1`を使用した。
質問間で会話履歴を共有していない。

| Metric | Dense baseline | Parent Dense | Hybrid baseline | Parent Hybrid |
| --- | ---: | ---: | ---: | ---: |
| answerable回答 | 4/6 | 5/6 | 6/6 | 5/6 |
| false abstention | 2/6 | 1/6 | 0/6 | 1/6 |
| unanswerable strict abstention | 6/6 | 6/6 | 6/6 | 6/6 |
| false answer | 0/6 | 0/6 | 0/6 | 0/6 |
| contract failure | 0 | 0 | 0 | 0 |
| citation成功 / valid answer | 4/4 | 5/5 | 6/6 | 5/5 |
| 正解source Top-5 | 6/6 | 5/6 | 6/6 | 5/6 |
| 正解source引用 | 4/6 | 4/6 | 6/6 | 4/6 |
| abstain本文/source漏出 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Parent Denseは`Counter.total()`を正しいsource付きで回答し、最低改善条件を満たした。
ただし`typing.Protocol`は引き続きabstainした。また`re.fullmatch()`の正解sourceが
Top-5から外れ、Qwenは`match()`を回答した。datasetのローカルfalse-answer指標は
unanswerable質問だけを数えるため、この意味的誤回答は0件という集計には現れない。

## Qwen RAG quality 10問

OpenAI Judgeは実行せず、ローカル決定的指標と既存の人手rubricだけを用いた。

| Metric | Dense baseline | Parent Dense | Hybrid baseline | Parent Hybrid |
| --- | ---: | ---: | ---: | ---: |
| valid answer | 7/10 | 8/10 | 7/10 | 8/10 |
| valid abstain | 3/10 | 1/10 | 2/10 | 1/10 |
| contract failure | 0/10 | 1/10 | 1/10 | 1/10 |
| citation成功 / valid answer | 7/7 | 8/8 | 7/7 | 8/8 |
| 正解source Top-5 | 8/10 | 9/10 | 7/10 | 9/10 |
| 正解source引用 | 7/10 | 8/10 | 7/10 | 8/10 |
| 無関係sourceのみ | 0 | 0 | 0 | 0 |
| 初回contract成功 | 10/10 | 9/10 | 9/10 | 9/10 |
| retry / 最終失敗 | 0 / 0 | 1 / 1 | 1 / 1 | 1 / 1 |

descriptorとTaskGroupが回答へ改善したが、`ensure_ascii=False`はParent Dense/Hybridの
両方で2回のcontract検証に失敗した。Parent HybridのTaskGroup回答には例外が外側へ
伝播しないという資料と整合しない主張があり、citation形式成功をsemantic成功とは
扱えない。`contextlib.aclosing()`も`await`と早期終了時の条件を含まず、completenessの
改善余地が残る。

## Contextと性能

全Parent runで5 parentすべてが8,192 tokenへ収まり、context budget除外0、
1件も収まらなかった質問0だった。

| dataset / 方式 | 平均prompt token | 中央 | 平均parent文字 | 検索平均 | 生成平均 | 全体平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Answerability Dense baseline | 1,839.4 | 1,808.5 | 2,916 | 0.0120秒 | 0.5624秒 | 0.5917秒 |
| Answerability Parent Dense | 1,955.2 | 2,017.0 | 3,431 | 0.0128秒 | 0.6658秒 | 0.6973秒 |
| Answerability Hybrid baseline | 2,036.8 | 2,022.5 | 3,374 | 0.0214秒 | 0.6191秒 | 0.6591秒 |
| Answerability Parent Hybrid | 2,108.5 | 2,096.5 | 3,674 | 0.0309秒 | 0.6424秒 | 0.6933秒 |
| RAG Dense baseline | 1,871.1 | 1,979.0 | 3,638 | 0.0132秒 | 1.3741秒 | 1.4056秒 |
| RAG Parent Dense | 1,919.2 | 1,971.0 | 3,868 | 0.0143秒 | 1.5367秒 | 1.5705秒 |
| RAG Hybrid baseline | 2,028.1 | 2,057.0 | 3,879 | 0.0230秒 | 1.4877秒 | 1.5306秒 |
| RAG Parent Hybrid | 2,070.3 | 2,207.5 | 4,005 | 0.0318秒 | 1.5446秒 | 1.5968秒 |

ParentStore構築は0.58～1.47秒、child index loadは0.38～0.91秒、Qwen loadは
3.34～6.03秒。process peak CPU RSSは5.72～5.93 GB、CUDA peak allocatedは
9.18～9.25 GB、peak reservedは9.45～9.58 GBだった。すべてRTX 4090
（24,564 MiB）のreference値で、過去のL4/T4と単純比較しない。

## 採否判断

実験実装は保持するが、production CLIの`ask`/`chat`へは統合しない。

採用条件のうち、false answer 0、abstain漏出0、Dense Answerability false abstention
1件改善、Holdout Hit@5改善、token上限維持、child metadata非漏出は満たした。一方、
Parent DenseのRAG contract failureがbaseline 0から1へ増えた。Hybrid Answerabilityは
false abstentionと正解source引用が悪化し、answerable質問の意味的誤回答も確認した。
全44ケースではbaselineのcontract failure 1件に対してParentは2件であり、
「contract failureを増やさない」を満たさない。改善効果はあるが、現時点でCLIの
追加複雑度を正当化する一貫性がない。設定選定に使ったDevelopmentでは一貫して改善せず、
未知Holdoutでは改善するという傾向の反転もあり、改善がRetrieverと質問セットに依存する。
良い方向の結果だけを選んでproduction採用とはしない。

したがってCLI既定とoptionは変更せず、`chunk` modeだけをproduction経路として維持する。
次段階では完全な見出し単位`DocumentSection`を一般HTML ingestionで永続化し、同じ
`ParentStore`境界へ差し替えて再評価する。今回の結果を完全なDocumentSection parentの
評価とも、Parent Document概念全体の否定とも主張しない。直ちにRerankへ進まず、
Python専用版をこのcheckpointで固定して一般HTML版へ移行する。

## Provenance

- 開始branch: `feat/parent-document-retrieval`
- 開始HEAD: `8fb2298b63a3569319868dce0d3e0e7207d565b4`
- experiment revision: `existing-chunk-parent-v1`
- source code parent commit: `8fb2298b63a3569319868dce0d3e0e7207d565b4`
- evaluation definition commit: `f009fc5fc03ea2bdb4df09bbfce2b78c911b7f09`
- 実験実行日時: 2026-07-30 UTC
- child artifact構築: 2026-07-30T21:00:50.687506+00:00 ～
  2026-07-30T21:01:18.159500+00:00
- GPU: NVIDIA GeForce RTX 4090（24,564 MiB）
- NVIDIA driver: `580.159.04`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Sentence Transformers: `5.6.1`
- FAISS: `1.14.3`
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Qwen model: `Qwen/Qwen3-4B-Instruct-2507`
- Qwen revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- dtype: `bfloat16`
- child config: size 400 / overlap 100
- child candidate k: 30
- OpenAI API / OpenAI Judge: 未使用
- API key・秘密値: 保存なし
- parent metadata SHA-256:
  `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869`
- selected child JSONL/metadata SHA-256:
  `7d80dcdd56ab1ea2e2717dcb7d93d9c10703391b1af67c63182cd4fb1d65d425`
- selected child index SHA-256:
  `1a1f94cc685401c448e1d0d9af32c546f3d5aa7d6e68296e77d7b69d8a109ef1`
- Development questions SHA-256:
  `363ed4d55564ba3b925a90100bc002dcf9ffe462a44ec899184323fd5adc2f40`
- Holdout questions SHA-256:
  `f71870a240e7a98ca05716e3e1d8f4d09b2ff856cdd93f608cd6ebeca86ffc36`
- Answerability questions SHA-256:
  `d14734bc35967482e9d54a0300c929182a08371c03b1e8fe8ccc91893f9788b6`
- RAG quality questions SHA-256:
  `585cc4200f697472fe8dd31d4e5b35ce488f2b04a5202e2937cc88ba4bb82de9`

| 詳細JSON | SHA-256 |
| --- | --- |
| `parent_dense_development.json` | `bfde76aac0179b92685fe9c8f93a9fe9634ca744d1dbabfafdf90af7557dea5e` |
| `parent_hybrid_development.json` | `e5b28b252dc9145c4afc84290f03e9f796c2d75d6d90a18f5dafed5a43375e12` |
| `parent_dense_holdout.json` | `b8028bcedcc5055836f135ebcdf23c537069b1f9865375d1acc16249f9020b1c` |
| `parent_hybrid_holdout.json` | `c8835e817d07e3b20a20a7e8c7a068793244752cee1962ff83aeecc1c92c21d8` |
| `answerability_contract_parent_dense.json` | `a6f8ef52fcd320039b61d1611bbdbbec4298607c2eb0cbb959089d98b51b8f97` |
| `answerability_contract_parent_hybrid.json` | `15a3ef27ee409b3f02a436ffef1666e25d9b36b689b98d42c3f68c00bbff08e5` |
| `rag_quality_contract_parent_dense.json` | `7c33a5065bc48644b90d2b49c7d30d62ab6a543a22929db2b86d94d1813319f9` |
| `rag_quality_contract_parent_hybrid.json` | `d910528335c1a568735a696819cf0aca24e20d17e49351f1ca0e164d85a89621` |

詳細JSONとchild artifactsはrepository外のGit管理外data-rootへ保存した。
baseline FAISS、metadata、manifestは上書きしていない。

## 検証

- `uv lock --check`: 成功
- `uv run --frozen pytest -q`: 242 passed
- `uv run --frozen ruff check .`: 成功
- inference・evaluation-openai extra付きpytest: 242 passed
- inference・evaluation-openai extra付きruff: 成功
- `git diff --check`: 成功
- OpenAI API、API key操作: なし
