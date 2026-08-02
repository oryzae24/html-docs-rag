# Source snapshot and explicit refresh policy

## 目的

Python 3.13日本語公式ドキュメントは、保護済み実験を再現する
frozen sourceと、mutable upstreamの現在内容を明示操作で取得するcurrent
sourceを分離する。取得済みsourceを暗黙に差し替えず、source取得と
HTML解析、下流artifact構築の責務を分けることがこのpolicyの中心である。

## Frozen sourceのprovenance

2026-07-20T20:34:21.446060+00:00の取得記録は次を示す。

- requested upstream URL:
  `https://docs.python.org/ja/3.13/archives/python-3.13-docs-html.zip`
- original recorded archive SHA-256:
  `f8ddb3454726cbe34580b4c21723128a1b33b50f1155e9b9184cb790db66d9cb`
- original recorded archive size: 17,367,739 bytes
- upstream URLはpatch固定URLではなくmutable alias

original archive bytesはproject snapshot作成時には現存していなかった。一方、
その取得から保持された展開済みHTML treeは、protected chunk JSONLを
byte-identicalに再現した。リポジトリの
`resources/source_snapshots/python-3.13-ja-2026-07-20.zip`は、この検証済みtreeを
ファイル内容を変更せず決定的にZIP化したproject snapshotである。original
archive bytesの複製ではなく、originalの記録SHAを持つとも表明しない。

project snapshotのidentityは次のとおりである。

- path: `resources/source_snapshots/python-3.13-ja-2026-07-20.zip`
- SHA-256: `1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777`
- size: 17,310,566 bytes
- members: 1,258
- archive root: `python-3.13-docs-html`
- source base URL: `https://docs.python.org/ja/3.13/`
- Python parser version metadata: `3.13`

memberはUTF-8 path byte順、timestamp・mode・compressionを固定して作成される。
詳細なsource tree hash、file/directory数、license/copyright pathは
`resources/source_snapshots/python-3.13-ja-2026-07-20.provenance.json`を正本とする。

## Component boundary

pipelineは次の境界に分かれる。

```text
SiteConfig
  |-- loader settings ----> loader registry ----> DocumentLoader
  |                                              -> SourceDocument
  `-- parser settings ----> parser registry ----> HtmlDocumentParser
                                                 -> DocumentParseResult

SourceDocument -> DocumentSection -> SearchChunk -> index -> profile artifact -> RAG
```

- Loaderはsourceの取得、cache、archive検証、安全な展開、`SourceDocument`作成を
  担当する。DOMとPython versionは知らない。
- Parserは取得済みHTMLをsectionへ変換する。sourceの取得方法は知らない。
- `src/python_doc_rag/ingestion/protocols.py`と`orchestration.py`はsite-neutralな
  protocol、result、parse/chunk orchestrationを持つ。
- `src/python_doc_rag/ingestion/registry.py`はloaderとparserを別々のtype IDで構築し、
  対応する組合せを一つのcapability matrixで検査する。
- `src/python_doc_rag/loaders/zip_archive.py`はsite-neutralなZIP sourceを担当する。
- `src/python_doc_rag/sites/python_docs/parser.py`だけがPython/SphinxのDOM、FAQ、
  permalink、noise selector、section規則を知る。
- `src/python_doc_rag/parsers/generic_html.py`はTOMLのCSS selectorで調整する
  site-neutral parserで、Python parserをimportしない。

loader typeからparser typeを推測しない。dataset name、slug、URL hostによる
parser切り替えも行わない。現在の宣言済み組合せは次の4つである。

| Loader | Parser | 用途 |
| --- | --- | --- |
| `pinned-local-archive` | `python-sphinx` | Python frozen |
| `snapshot-http-archive` | `python-sphinx` | Python current |
| `local-html-tree` | `python-sphinx` | Python展開済みHTML互換 |
| `bounded-http` | `generic-html` | uv portability |

Pythonは公式の一括HTML archiveでページ集合と順序を固定でき、protected
corpusの完全性を検証できるためarchive loaderを使う。uvは小さなportability
datasetとしてcrawl境界をTOMLで明示できるため、従来のbounded page crawlと
generic parserを維持する。

## SiteConfigの3経路

### Frozen

`configs/sites/python-docs.toml`はconfig fileからの相対pathでproject snapshotを参照し、
必須SHA-256を検証する。network transportを持たず、`--source-root`も使用しない。
protected評価、再現検証、デモ、multi-KB APIのPython datasetはこの経路を
正本とする。`--refresh`はsource更新の意味を持たないため拒否し、
下流再構築は`--rebuild`を使う。

### Current

`configs/sites/python-docs-current.toml`はmutable HTTPS aliasを空のdata-rootで一度取得し、
observed archive identityを`data/raw/source.lock.json`に固定する。通常実行は
remoteの現在値を確認せず、lockとcontent-addressed local archiveを正本にする。
`--refresh`を指定した場合だけremoteを再取得し、新しいobserved SHAで
source lockと下流artifactを更新する。currentの件数やSHAをprotected baselineと
して扱わない。

### Local compatibility

`configs/sites/python-docs-local-compat.toml`は利用者が渡す展開済みHTML rootを
読み、従来と同じ`python-sphinx` parserへ渡す。この経路だけ
`--source-root`が必須である。既存Python data-rootは移動やrenameを行わず、
legacy layout readerで後方互換を維持する。

## Source lock

`snapshot-http-archive`の`source.lock.json`はstrictな`source-lock-v1` JSON objectで、
次を保存する。

- `complete=true`とloader type
- requested URLとredirect後のfinal HTTPS URL
- observed SHA-256とbyte size
- archive formatと固定archive root
- citation用source base URLとinclude path prefixes
- fetched timestamp
- `source_config_sha256`と`source_snapshot_sha256`
- data-rootからの安全な相対archive cache path

absolute local pathは保存しない。lockのunknown/missing key、source config不一致、
不正SHA、不正size、HTTPへのredirect、unsafe relative pathは再利用前に拒否する。
cacheの実バイトは再利用時にもSHA-256とsizeを再検証する。

## Fingerprint model

identityを4つに分け、TOMLのコメントや空白変更でsourceを再取得しない。

| Identity | 対象 | 用途 |
| --- | --- | --- |
| `source_config_sha256` | loader type、URL/path、archive root、include prefixes、crawl上限・境界等のsource-affecting設定 | source cache/lockと設定の一致確認 |
| `processing_config_sha256` | dataset metadata、parser、chunking、index、profile | 下流artifact再構築の必要性確認 |
| `source_snapshot_sha256` | archiveではarchive bytes、local tree/crawlでは順序付きportable page identity | 実際に固定されたsource contentのidentity |
| `site_config_file_sha256` | TOML生バイト | 監査用。source cache再利用の唯一基準にはしない |

timeout、retry、request delayのような運用設定は、source contentを変えない限り
archive source identityから分離する。bounded crawlは`max_pages`、query保持、
robots policy、response size上限などをsource identityに含める。

## Cache decision table

| 状態 | 通常`prepare` | `--offline` | `--refresh` | `--rebuild` |
| --- | --- | --- | --- | --- |
| 完成dataset、source/processing fingerprint一致 | 全artifactを検査し再利用、networkなし | 同左 | refresh対応loaderはsource再取得から再構築 | local sourceから下流再構築 |
| dataset未完成、valid source lock/cacheあり | local sourceから構築、networkなし | local sourceから構築 | source再取得から構築 | local sourceから構築 |
| data-rootにsource cacheなし | pinnedはrepository ZIP、localは`--source-root`、current/boundedは初回取得 | pinned/localは指定local sourceを使用、current/boundedはfail-closed | current/boundedは取得、pinned/localは拒否 | pinned/localは指定local sourceを使用、current/boundedはfail-closed |
| processing fingerprintだけ不一致 | `--rebuild`を案内し停止 | 停止 | sourceも更新する意図がある場合のみ | source lock/SHAを変えず再構築 |
| source config不一致 | 暗黙置換せず停止 | 停止 | current/boundedは明示更新 | pinned/localは別data-rootを使用 |

`--offline`はdocument source取得のnetworkを禁止する。Hugging Face modelまで完全offlineに
する場合は、必要model cacheも別途用意する。`--refresh`と`--rebuild`は互いに
排他で、`--offline --refresh`も拒否する。

`--resume`は既存の`.prepare-staging`がsource、processing、targetの同一stateで
検証できる場合だけ再開する。bounded crawlの検証可能なpartial stagingも
再利用できるが、HTTP Rangeによるarchive download途中再開は行わない。

## Publication atomicity

- downloadはdata-root内で作成した一時file descriptorへstreamし、実読込み量に
  `max_archive_bytes`を適用する。retry、truncate、SHA-256計算、公開前検証は同じ
  inodeを保持して行い、完了後のSHA-256とsize確定前にはcacheとして公開しない。
- mutable archiveは`data/raw/archives/<sha256>.zip`としてcontent-addressedに保存する。
- ZIPは一時directoryへ展開し、完全なextraction manifestとarchive rootを検証して
  からrenameで公開する。
- corpus、index、profile、dataset manifestは`.prepare-staging`で構築し、完了後に
  backupとdurable publication journalを伴うrename transactionで公開する。journalが
  残る間はruntime artifact resolutionもfail-closedにする。
- refreshのsource lock/fetch manifestは旧bytesを保持し、下流publication失敗時に戻す。
  bounded crawlも旧cacheをbackupし、失敗時に復元する。
- source publicationとderived publicationの間でprocessが停止しても、次回prepareは
  journalのphase、candidate、backup、現在のsource/processing fingerprintをstrictに
  検証し、同じgeneration全体をforward-finalizeまたはrollbackする。legacy manifestを
  新transactionのcommitted candidateとして受理しない。
- committed sourceに対するderived candidateが後から破損していた場合は、明示した
  `--rebuild`または`--refresh --resume`だけが旧journal/stagingをrecoverable bundleへ
  退避して再構築できる。退避物やcontent-addressed orphanは自動的に完成datasetとは
  見なさず、回復可能性を優先して残す。
- cleanup途中の割込みは、すでにstrict検証できる新generationを旧generationへ戻さず、
  次回prepareでidempotentにcleanupを完了する。

## ZIP security boundary

ZIPの展開は次をfail-closedで拒否する。

- absolute path、`..`、`.` component、Windows drive path、backslash、NUL
- symlink、deviceやその他のspecial entry
- Windows予約名、末尾dot/space、portable component長の超過
- Unicode NFCとcase-foldを考慮したduplicate output path、implicit parent collision
- file/directory衝突、予約済み`extraction_manifest.json`との衝突
- central directoryの実record数、member count、declared/streamed individual member
  size、total extracted size、archive sizeの上限超過
- 展開先の既存symlink/junction、symlink ancestor、展開後cacheへのextra/modified file
- TOMLで固定した`archive_root`の不在

archive pathはSHA検証からcentral-directory preflight、展開完了まで同じnofollow
descriptorを使い、選択HTMLもnofollowで開いた実bytesのdigestを再確認する。archive
rootの自動推測は行わない。include pathで選ばれたHTMLだけを、archive root内の
決定的なlogical path順で読み込む。

## Citation URL boundary

`archive_url`はsource bytesの取得元、`source_base_url`は回答に使う公式出典の
base URLで、目的が異なる。archive loaderは`source_base_url`と安全にquoteした
logical pathから`SourceDocument.source_url`/`canonical_url`を作る。Python parserは
このURLをtrusted page URLとし、section anchorだけを追加する。local archive pathは
chunk、prompt、回答、API responseへ出さない。URLをLLMに生成させない。

## Repository policy and limitations

protected sourceの永続的な再現のため、project snapshot ZIPだけを通常のGitで
管理する。Git LFSは使わない。current archive/cache、展開済みtree、処理済み
corpus、FAISS index、model cacheはGit管理外である。外部storageの実データや
absolute local pathもコミットしない。

現段階で対応するarchive formatはZIPだけで、tar、patch version自動更新、
差分同期、HTTP Range resume、外部pluginの動的読込み、uv専用parserは対象外である。
すべての構築・評価・回答はlocal/open modelで実行し、OpenAI APIは使用しない。

## 2026-08-01 validation record

RunPod NVIDIA L4（23,034 MiB）上で、frozen snapshotからfresh one-command
`prepare`、recommended-v2 `check`/`ask`、完成dataset再利用、local-sourceだけの
`--rebuild`を実行した。384 pages、2,766 sections、8,677 chunksとprotected chunk
SHA-256 `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869`
をbyte-identicalに再現した。

current archiveは2026-08-01に初回取得し、通常再利用、raw lock/cacheだけからの
offline replay、明示`--refresh`を確認した。uvも31 pages、184 sections、270 chunks、
通常再利用、raw-cache-only offline replay、refresh、check/askを確認し、frozen
Python + uvのtwo-KB API smokeもdomain分離とshared-model再利用を保って成功した。
artifact SHA、current observed SHA/size、uv refresh時のraw-only差分、GPU peak、API結果は
`evaluation/python_docs_source_snapshot_summary.md`を正本とする。

このsource workflowのactual Google Colab実行は行っていない。過去のColab T4
互換性試験は別の検証記録であり、このbranchのclean-room smokeとは区別する。
