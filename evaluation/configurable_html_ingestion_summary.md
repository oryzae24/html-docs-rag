# Configurable HTML Ingestion / Portability Smoke

## 目的と範囲

Python公式文書RAGの後半（technical-field retrieval、BGE-M3、mMARCO
reranker、Qwen3-8B、answer-or-abstain-v1、citation finalization）を変更せず、
その入力側を `SiteConfig → DocumentLoader → SourceDocument →
HtmlDocumentParser → DocumentSection → Chunker → SearchChunk →
DatasetArtifactLayout` に分離した。

対象は、robots.txtを尊重し、start URLから導出した境界内だけを取得する静的
HTML文書サイトである。JavaScript rendering、認証、PDF/Word、差分同期、
分散crawl、任意HTMLを自動理解する万能parserは対象外である。

## 実装境界

- `SourceDocument` はdecoded HTML、source/canonical URL、content SHA-256、
  論理path、JSON互換のimmutable metadataを保持する。絶対local pathや秘密値は
  citation metadataへ保存しない。
- Pythonの標準経路は`pinned-local-archive`、現在版は
  `snapshot-http-archive`、展開済みtreeは`local-html-tree`互換経路として、いずれも
  同じ`python-sphinx` parserを独立に選択する。FAQ見出し補完、headerlink除去、
  URL、category、順序をgeneric parserの都合で変更しない。
- `BoundedHttpHtmlLoader` は複数start URLからscheme/origin/path-prefixのunionを
  導出する。決定的BFS、external redirect拒否、Content-Type/body size上限、
  bounded retry、page failure継続、source設定全体のfingerprint、atomic raw cache、
  offline replay、明示refresh、resumeを持つ。
- robots.txtの200応答は規則を適用し、404/410だけを「規則ファイルなし」として
  empty allowにする。401/403、5xx、timeout、decode失敗はfail-closedとする。
- `GenericHtmlParser` はcontent/exclude/title selectorとheading levelをTOMLから
  受け取る。site固有classはgeneric Pythonコードへ埋め込まない。
- `dataset_manifest.json` のpathはdata-root相対だけを許可する。legacy Python
  layoutはcompatibility resolverで維持する。
- algorithm profileのmodel/revision/dimension/prefixと、dataset固有のmetadata、
  symbol、FAISS SHA-256をdataset-local profile artifact manifestへ分離した。

## Python後方互換

このportability smokeで使用した展開済みPython 3.13日本語HTMLを全量処理した結果は、
現在の`configs/sites/python-docs.toml`が参照するfrozen project snapshotからも
共通`prepare`経路でbyte-identicalに再現されている。

| 項目 | 結果 |
| --- | ---: |
| source page | 384 |
| section | 2,766 |
| chunk | 8,677 |
| baselineとのline count | 一致 |
| processed SHA-256 | `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869` |
| baselineとの`cmp` | byte-identical |

この初回portability smokeでは既存BGE-M3 indexとsymbol sidecarを再構築せず、
その既存SHAを参照するdataset-local identity manifestだけをadoptした。
その後のfrozen snapshot clean-room検証ではDense、BGE-M3、symbol sidecarをすべて
再構築して`check --profile recommended-v2`に成功した。新しいsource workflowの実測値は
`evaluation/python_docs_source_snapshot_summary.md`を参照する。

## uv静的HTML portability（取得・解析・artifact）

設定は `configs/sites/uv-docs-smoke.toml` に閉じ込めた。start URLは
`https://docs.astral.sh/uv/getting-started/` と
`https://docs.astral.sh/uv/guides/`、上限40ページ、delay 0.5秒、timeout 15秒である。
2026-08-01 UTCのpreflightでは両URLが200、`text/html; charset=utf-8`、主要本文が
HTTP応答に含まれ、canonicalは境界内だった。robots.txtは404で、上記のempty allow
方針を適用した。

| 項目 | 結果 |
| --- | ---: |
| fetched / parsed / failed page | 31 / 31 / 0 |
| section / chunk | 184 / 270 |
| excluded section / node | 9 / 31 |
| code block / table | 369 / 0 |
| fallback page | 0 |
| duplicate canonical / content | 0 / 0 |
| excluded external/unsupported link | 2,155 |
| distinct source page | 31 |
| duplicate chunk text hash | 0 |
| average / median chunk length | 572.1 / 576.5 characters |
| average / median section length | 807.6 / 565.0 characters |
| duplicate section text hash | 0 |
| selector調整回数 | 0 |

10ページの決定的sampleを確認し、navigation/footerの大量混入がなく、heading、shell例、
command optionのhyphen、code blockが保持され、source URLがconfigured boundaries内で
あることを確認した。smoke質問作成後のparser/retrieval parameter調整は行わない。

### Artifact

data-rootはrepository外のGit管理外領域に置いた。

| Artifact | 件数/次元/サイズ | SHA-256 |
| --- | --- | --- |
| processed chunks | 270 / 230,728 bytes | `83b49de1a8b465f0e8e8c0f20961fdffa9b0c3b92dc76bda76a303a75763dc84` |
| baseline dense FAISS | 270 / 384 / 414,765 bytes | `152220c72d95a0f4235ec54ef4866d0b32cc5c3348e708b81319f00ad249fb1a` |
| BGE-M3 FAISS | 270 / 1,024 / 1,105,965 bytes | `38a5fb7fd117270db613ac7d0fefcc8421a9210cf7500d8689ca1a8fcc28e124` |
| symbol sidecar | 270 / 234,917 bytes | `5a9ac53294f50e70fd2a3036c8970818cea474f5ed70adcbf8be2576c04129dd` |
| dataset profile identity | 270 | `b790f3131246cce827d4c7cdf07facd5a1544fdfe44c559b9ba098a06728853d` |
| fetch manifest | 31 page | `fb72e91d93f8e78a8e41e8db202cb4dc8f81d9caf4202c32b4c45224c9d4ff92` |

baseline dense buildは7.839秒、BGE-M3 buildは18.134秒だった。モデル設定はPythonの
recommended-v2から変更せず、BGE-M3 revisionは
`5617a9f61b028005a4858fdac845db406aefb181` である。

## Portability smoke Q&A

取得済みchunkでsourceとrequired factsを確認した後、8問を固定した。answerableは
exact identifier、conceptual、operationalを各2問、unanswerableは取得境界外の現実的な
設定質問2問である。6つのanswerableは異なるURL anchorを使う。question SHA-256は
`ac59d2e15f42a4f8017077dd02607a1cbd96996a447715462349e10ec86d032b`。
質問作成後にparser、selector、retrieval、profile parameterは変更していない。

| 指標 | 結果 |
| --- | ---: |
| valid answer | 5 |
| valid abstain | 2 |
| false answer | 0/2 |
| false abstention | 0/6 |
| contract failure | 1 |
| correct source Top-5（完成record） | 5/6 |
| correct source cited | 5/6 |
| citation format failure | 0 |
| answer内URL | 0 |
| abstain本文 / source漏出 | 0 / 0 |
| source boundary failure | 0 |

contract failureは`uv sync --no-install-project`のanswerable質問で、JSON契約へ2回失敗
した。生成を再評価せずretrieverだけを独立確認すると、正解Docker sectionはrank 2に
あり、retrievalとしては6/6で正解source Top-5だった。成功5回答の人手相当reviewでは
unsupported claimと重大な意味的誤りは0、required facts完全3・部分2だった。部分回答は
長いhelp commandの説明でshort helpとの差を省略したものと、universal lockfileの理由を
省略したものである。形式成功を完全性成功とは扱わない。

同一serviceを1回構築して8問へ再利用した。完成7件の平均はretrieval 0.114秒、generation
2.944秒、total 3.079秒、平均input 1,330 tokenである。service loadは71.091秒、peak CPU
RSS 5.03 GB、GPU peak allocated 18.22 GB、reserved 18.43 GBだった。実CLI `ask`は
4.233秒、`chat`の1回答は5.051秒で成功し、いずれもconfigured boundary内のmetadata
URLだけを表示した。

詳細JSON SHA-256は
`7518eb9d06a9dc4974ff3899f4b4bbe4dee588f7077de55169ad99d336c465b3`。
これは取得済み範囲を確認して作った機能smokeであり、blind benchmarkではない。

## 実行環境と制約

- GPU: NVIDIA L4（23,034 MiB）
- driver: 580.126.20
- PyTorch: 2.11.0+cu128、CUDA runtime 12.8
- Transformers: 5.14.1
- Sentence Transformers: 5.6.1
- OpenAI API / OpenAI Judge: 未使用
- APIキー・秘密値: artifactへ保存していない

主成果はPython公式文書RAGであり、ここで示す一般化は設定可能な静的HTML文書サイト
への限定的移植性である。site固有selectorは引き続き必要で、JavaScript、認証、
PDF、汎用crawler製品としての要件は満たさない。将来は前半のingestion基盤を独立
componentへ分離する余地がある。

31件のsource content SHA-256はcorpus manifestへ個別保存した。その順序付きdigest列を
newline結合したSHA-256は
`3c86003e61aa6d96dac986a3bd80eb6f9f8c5b6312d2d0df34ba5af0726cba1b`である。
