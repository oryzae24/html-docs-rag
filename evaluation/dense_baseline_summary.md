# Dense FAISS検索ベースライン

## 実行条件

- 実行日: 2026-07-29
- OS: Linux 6.5.0-44-generic x86-64
- Python: 3.12.3
- device: `cuda:0`
- Embeddingモデル: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Sentence Transformers: 5.6.1
- FAISS: 1.14.3 (`IndexFlatIP`、384次元)
- indexチャンク数: 8,677
- 評価質問数: 26
- 取得件数: Top-10
- 完全な結果: データルートの`evaluation/dense_baseline.json`（Git管理外）

Qwenなどの回答生成モデルはロードせず、保存済みDense FAISS indexを変更せずに
測定した。

## 全体結果

| 指標 | 値 |
| --- | ---: |
| Hit@1 | 0.3846 (10/26) |
| Hit@3 | 0.5385 (14/26) |
| Hit@5 | 0.6923 (18/26) |
| Hit@10 | 0.8077 (21/26) |
| MRR@10 | 0.4972 |
| 平均検索時間 | 0.0168秒 |
| 中央検索時間 | 0.0092秒 |

## query_type別結果

| query_type | 質問数 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_identifier | 10 | 0.5000 | 0.7000 | 0.7000 | 0.8000 | 0.5976 |
| conceptual | 8 | 0.2500 | 0.3750 | 0.5000 | 0.7500 | 0.3710 |
| operational | 8 | 0.3750 | 0.5000 | 0.8750 | 0.8750 | 0.4979 |

operationalはHit@5が最も高い。exact_identifierも全体としては強いが、短く希少な
識別子では大きく外れる。conceptualはHit@10まで広げると0.7500まで上がる一方、
MRR@10が最も低く、正解を上位へ集め切れていない。

## 主な成功例

次の質問は正解URLが1位だった。

- `sys.argv[0]には何が入りますか？`
- `subprocess.run()とPopen()の違いは何ですか？`
- `pathlib.Path.exists()は何を返しますか？`
- `unittest.TestCaseを使ってテストケースを定義する方法は？`
- `os.pipe()は何を返しますか？`
- `list.sort()がNoneを返すのはなぜですか？`
- `with文でファイルを開く利点は何ですか？`
- `readlineモジュールで入力履歴をファイルへ保存するには？`

subprocess、pathlib、shlexは各topicのHit@5が1.0000であり、API名と操作語が本文・
節タイトルに現れる質問ではDense検索が機能した。

## 主な失敗例

Top-10に正解がなかった質問は次の5件。

- `isatty()とは何ですか？`
- `EOFErrorはどのような場合に発生しますか？`
- `main関数はどこに配置するべきですか？`
- `標準出力はいつ行バッファリングされ、いつブロックバッファリングされますか？`
- `Pythonでパイプ用のファイルディスクリプタを作るには？`

`os.pipe()は何を返しますか？`は1位だが、自然文に言い換えたパイプ作成質問は
`multiprocessing`や廃止済み`pipes`へ流れた。識別子を含むかどうかで結果が大きく
変わる。標準出力バッファリング質問は`inspect`、`asyncio`、`datetime`へ分散し、
`sys.stdout`の直接記述を取得できなかった。

## main関数質問のTop-10

正解は`library/__main__.html#idiomatic-usage`または
`faq/library.html#how-do-i-test-a-python-program-or-component`。Top-10内に正解なし。

| rank | score | section | URL |
| ---: | ---: | --- | --- |
| 1 | 0.4916 | 優先度キュー実装の注釈 | `library/heapq.html#priority-queue-implementation-notes` |
| 2 | 0.4809 | レイアウト | `library/tkinter.ttk.html#layouts` |
| 3 | 0.4801 | 演算子の優先順位 | `reference/expressions.html#operator-precedence` |
| 4 | 0.4486 | ジェネレータ式とリスト内包表記 | `howto/functional.html#generator-expressions-and-list-comprehensions` |
| 5 | 0.4430 | べき乗演算 | `reference/expressions.html#the-power-operator` |
| 6 | 0.4389 | スケジューラーへのインターフェイス | `library/os.html#interface-to-the-scheduler` |
| 7 | 0.4370 | 組み込み関数 | `library/functions.html#built-in-functions` |
| 8 | 0.4368 | 関数定義 | `reference/compound_stmts.html#function-definitions` |
| 9 | 0.4315 | Geometry management | `library/tkinter.html#geometry-management` |
| 10 | 0.4242 | トップレベルのスクリプト環境 | `library/__main__.html#what-is-the-top-level-code-environment` |

「どこに配置」という日本語の意味が、レイアウト、優先順位、演算子の位置などの
一般的な「配置」に近いチャンクへ強く対応した。`main`はPythonで特別な関数名では
なく、正解本文も慣習として説明しているため、字句手掛かりと意味手掛かりの双方が
弱い。10位の`__main__`ページは関連するが、main関数の配置を直接説明する節ではない
ため正解ラベルを広げなかった。

## isatty質問のTop-10

正解は`library/io.html#i-o-base-classes`または
`library/os.html#file-descriptor-operations`。Top-10内に正解なし。

| rank | score | section | URL |
| ---: | ---: | --- | --- |
| 1 | 0.4685 | 関数およびクラス定義 | `library/ast.html#function-and-class-definitions` |
| 2 | 0.4160 | errnoシステムシンボル | `library/errno.html#module-errno` |
| 3 | 0.4137 | IMAP4オブジェクト | `library/imaplib.html#imap4-objects` |
| 4 | 0.4068 | 関数およびクラス定義 | `library/ast.html#function-and-class-definitions` |
| 5 | 0.4051 | 関数API | `howto/enum.html#functional-api` |
| 6 | 0.4024 | errnoシステムシンボル | `library/errno.html#module-errno` |
| 7 | 0.3996 | IMAP4オブジェクト | `library/imaplib.html#imap4-objects` |
| 8 | 0.3982 | 文 | `library/ast.html#statements` |
| 9 | 0.3900 | 関数API | `howto/enum.html#functional-api` |
| 10 | 0.3890 | プロセスのパラメーター | `library/os.html#process-parameters` |

短い希少識別子`isatty`の字面をDense埋め込みが保持できず、「関数とは何か」に近い
ASTやAPI節へ流れた。同一節の複数チャンクがTop-10を占有するため、多様性も低い。
正解本文には`isatty`が明記されており、Sparseな識別子一致や節タイトルの重み付けが
有効かを次段階で検証できる。

## ラベル品質

- isattyは`IOBase.isatty()`と`os.isatty()`の両方が直接回答するため複数正解。
- main関数は言語仕様上の特別名ではない。`__main__`の通常の使われ方とFAQの
  「global main logic」を慣習上の正解とした。
- `subprocess.run()`と`Popen()`、TestCase、`dict.items()`は複数の節が直接回答する
  ため、実本文を確認した節だけを複数正解にした。
- Ctrl+ZやCtrl+D/Ctrl+Zの差はOS・端末・シェル仕様が混ざるため、評価セットではなく
  `evaluation/unanswerable_candidates.jsonl`へ分離した。

## 次段階の改善候補

まず同じ26問でBM25 Sparse Retrievalを測定し、Denseとの単独比較を行う。その後、
Dense＋BM25のReciprocal Rank Fusion、ページタイトル・節タイトルの重み付け、
同一URL・同一節の重複抑制を優先候補とする。続いてMMR、Query Rewrite、ローカル
reranker、OpenAI API reranker、回答不能判定を比較候補にできる。

このベースラインでは、BM25、Hybrid Retrieval、RRF、MMR、reranker、Query Rewrite、
OpenAI API、回答不能判定は実装していない。

## Baseline provenance

- 評価実行日: 2026-07-29
- Evaluation definition commit:
  `63e305c60eb25a07dc37e0339a12c8c92cabb225`
- Source code parent commit:
  `ca9eec959119674e6867963d74bd092baef59e88`
- questions JSONL SHA-256:
  `363ed4d55564ba3b925a90100bc002dcf9ffe462a44ec899184323fd5adc2f40`
- Dense baseline JSON SHA-256:
  `889585cab61b8db46b89321164f13509f9196af3152228b18a8c2c4a25e89d9c`
- processed JSONL SHA-256:
  `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869`
- FAISS index SHA-256:
  `8b3803b5a458b823d11f86eb3a69b0a82452a5980b82343242fb974dfef338da`
- index manifest SHA-256:
  `0259b1b67126fc9dc27e67f6a2cc8dc44d4ec1090c6614a789e965ae205c3d5d`
- Embeddingモデル: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- chunk設定: `chunk_size=1000`、`chunk_overlap=150`
- Python: 3.12.3
- Sentence Transformers: 5.6.1
- FAISS: 1.14.3
- 実行device: `cuda:0`

基準commitのworking treeには評価基盤の未コミット差分があるため、評価入力の同一性は
commitだけでなく上記questions、processed JSONL、indexのSHA-256で確認する。保存済み
index manifestにはchunk設定が含まれないため、値はプロジェクト設定と既存reportで
確認した。
