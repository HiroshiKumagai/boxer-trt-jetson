# Boxer on Jetson — 最適化戦略ドキュメント

Jetson Orin NX 上で `jetson-boxer` の推論パイプラインを最適化した記録と、最終到達点。

---

## 1. パイプライン構成

Boxer は 3 モデルの連結による 3D 物体検出パイプライン。

```
入力画像 (RGB, 1×3×H×W)
    │
    ├─→ [OWLv2]        : ONNX Runtime / CUDAExecutionProvider
    │     テキストプロンプトで 2D 検出    →  bb2d (1×M×4)
    │
    ├─→ [DinoV3]       : ONNX Runtime / TensorrtExecutionProvider
    │     画像特徴量抽出 (固定 1×3×960×960) →  dino_feat (1×384×60×60)
    │
    └─→ [BoxerNetCore] : ONNX Runtime / CUDAExecutionProvider
          dino_feat + bb2d → query (1×M×dim)
                    │
                    ↓  [head : PyTorch]
                  3D bounding boxes (ObbTW)
```

### 実行プロバイダの選定

| モデル | プロバイダ | 理由 |
|---|---|---|
| OWLv2 | CUDA | TRT が patch embedding の Conv を非対応。FP16 は `logit_scale` オーバーフローで NaN |
| DinoV3 | TRT → CUDA フォールバック | 固定形状で TRT 向き。ORT が TRT 互換サブグラフを自動抽出 |
| BoxerNetCore | CUDA | 入力 `bb2d_norm` の M が動的。TRT は M ごとに再コンパイル (~27 s) のため不可 |
| head | PyTorch | `ObbTW` 構築など動的操作を含むため ONNX 化困難 |

ONNX を中間表現にするのは、TensorRT が PyTorch を直接読めないため。ORT の TRT EP はサブグラフ自動抽出＋非対応ノード CUDA フォールバックで、自力で TRT エンジンを組むより堅牢。

---

## 2. 最終結果

### 到達点 vs 目標

計測: Jetson Orin NX、jetson_clocks ON、`hohen_gen1` (499 フレーム、`--track`)。

RAM 削減を優先する方針のもと、最終的に **D5 + Phase 1 (allocator 絞り込み)** を採用。

| 指標 | ベースライン | 目標 | **D5 + Phase 1 (採用)** | 達成 |
|---|---|---|---|---|
| 処理時間 | 648 s | 400 s | **486 s (-25%)** | ⚠️ -86 s |
| RAM ピーク | 29.4 GB | 17–22 GB (初期目標) / 10 GB (後期目標) | **16.9 GB (-42%)** | ✅ 初期目標 / ❌ 後期目標 |
| GPU util 平均 | 56% | 70% 超 | **68%** | ⚠️ |
| Tracked OBBs | — | — | 81 | ✅ |

RAM 初期目標 (17–22 GB) は達成。後期目標の 10 GB は現行アーキテクチャでは量子化なしには不可と確定。時間と GPU util は D5 より僅かに劣るが、優先度 (RAM) を反映した判断。

### 採用構成と実行方法

```bash
BOXER_DISABLE_PERSISTENT_BUF=1 \
BOXER_CUDA_MEM_LIMIT_GB=4 \
BOXER_TRT_WORKSPACE_MB=256 \
BOXER_CORE_ARENA_STRICT=1 \
  bash benchmark.sh onnx --input sample_data/hohen_gen1 --track --skip_viz
```

| env var | 役割 |
|---|---|
| `BOXER_DISABLE_PERSISTENT_BUF=1` | OWL 永続出力バッファ無効化 (RAM コスト 10 GB 回避) |
| `BOXER_CUDA_MEM_LIMIT_GB=4` | CUDAExecutionProvider の allocator 上限 4 GB/session |
| `BOXER_TRT_WORKSPACE_MB=256` | TRT engine のワークスペース上限 256 MB |
| `BOXER_CORE_ARENA_STRICT=1` | Core CUDA EP の arena extend を `kSameAsRequested` |
| `--skip_viz` | 可視化の完全スキップ (速度・GPU util 優先) |

デフォルト ON: 事前最適化 ONNX キャッシュ、Dino/Core IOBinding、OWL の `synchronize_inputs/outputs`。

**時間優先構成が必要な場合**は上記 `BOXER_CUDA_MEM_LIMIT_GB` 等を外す（D5: 481 s / 18.9 GB / 82 tracked / 70% GPU）。

---

## 3. 採用した施策

### 一覧（効果量順）

| 施策 | 時間 | RAM | GPU util |
|---|---|---|---|
| **可視化スキップ** (`--skip_viz`) | -105 s (-17%) | 横ばい | +12 pp |
| **事前最適化 ONNX キャッシュ** (`*.optimized.onnx`) | -96 s (-15%) | ~0 GB | 小 |
| **PyTorch 重複ロード排除** (Stage 2) | — | -0.8 GB | — |
| **OWL IOBinding + stream 同期** | 同期点削減 | — | — |
| **Dino/Core IOBinding 化** | -19 s (-4%) | — | +1 pp |
| **TRT workspace 1 GB 制限** | — | 数 GB | — |

効果量は Stage 2 (PyTorch 排除済) を出発点として、他施策を有効化/無効化した個別計測に基づく。

### 詳細

**可視化スキップ**。`draw_bb3s` + hstack + imencode + disk write で合計 ~150 ms/frame。最大の単独レバー。viz async 化 (imencode/write のみバックグラウンドスレッド) は -6 s しか効かない（`draw_bb3s` が残るため）。本番運用で可視化不要なら完全スキップが最適。

**事前最適化 ONNX キャッシュ**。`sess_opts.optimized_model_filepath = path + ".optimized.onnx"` で ORT 最適化後グラフを初回実行時にディスクに書き出し、以降は `ORT_DISABLE_ALL` で直接ロード。セッション初期化時のグラフ最適化処理を省略できる。TRT プロバイダ (DinoV3) は compiled node を含み serialize 不可のため OWL と Core のみ。

**OWL IOBinding + stream 同期**。ORT の CUDAExecutionProvider は内部で独自の private stream を持ち、torch の default stream を自動で待たない。明示的に `io_binding.synchronize_inputs()` と `synchronize_outputs()` を挟まないと、torch の前処理が終わる前に ORT が入力を読む → NaN logits → 0 検出、となる決定的な race になる。OWL にこれを入れない限り IOBinding 導入は correctness を壊す（[§5 教訓参照](#5-教訓)）。

**Dino/Core IOBinding 化**。従来は `.cpu().numpy() → session.run → from_numpy().to(cuda)` のラウンドトリップ。IOBinding に置換して同期点を減らし、Dino は 固定 shape で永続出力バッファ、Core は M 動的のためフレーム毎 `torch.empty` + bind_output。OWL と同じく `synchronize_inputs/outputs` を併用。

**PyTorch 重複ロード排除** (Stage 2)。チェックポイントから読み込む `BoxerNet` には Dino・Core の PyTorch 重みが含まれており、ONNX で置換済なのに GPU メモリに残っていた。`head` / `prepare_inputs` / `dino.patch_size` のみ保持し、`model.dino = None` などで参照を切って `torch.cuda.empty_cache()`。

**TRT workspace 制限**。`trt_max_workspace_size = 1 << 30` (1 GB、デフォルト 4 GB)。推論速度に影響せず RAM 抑制。`arena_extend_strategy` や `cudnn_conv_algo_search='HEURISTIC'` は OWL の Conv で悪化したため不採用。

---

## 4. 採用しなかった施策

### 試行して見送り

**OWL 永続出力バッファ**。固定 shape の `pred_logits` / `pred_boxes` を warmup で確保し bind_output。per-frame の `torch.empty` を消して同期点削減を狙う。単独効果 -67 s だが、**peak RAM が +10.3 GB 増える**。ORT 内部のアクティベーションワークスペースが永続バインド時に累積される模様。RAM 目標 (17–22 GB) を満たすため **不採用**。

**OWL ∥ Dino 並列ストリーム**。両者は入力が独立なので別 CUDA stream で重ねれば Dino (~200 ms/frame) を OWL (~395 ms/frame) に隠せる。期待効果 -100 s。`torch.cuda.Stream` (event ベース) と `ThreadPoolExecutor` (スレッドベース) の両方を試行したが、どちらも直後の Core `run_with_iobinding` 呼び出しで `CUDA failure 700: an illegal memory access was encountered` (`cuda_stream_handle.cc:111` at `cudaStreamSynchronize`) が決定的に発生。ORT が CUDAExecutionProvider + TensorrtExecutionProvider を cross-EP / cross-thread で使う際の内部 stream 管理の問題で、IOBinding 持続バッファとの組み合わせが引き金。標準 API の範囲では安全に並列化する術がなく **見送り**。副作用として TRT engine cache が壊れることがあるため、その場合は `trt_cache/` を削除して再ビルドさせる。

**OWL CUDA Graph 有効化**。固定 shape なので `enable_cuda_graph=True` を狙ったが、OWLv2 ONNX に CPU 専用ノードが残り (ORT warning "10 Memcpy nodes are added") CUDA Graph capture の前提を満たせない。ONNX 再エクスポートでの Memcpy ノード除去は可能性としてはあるが、精度影響検証の工数が大きく **見送り**。

**viz async 化** (imencode + disk write のみバックグラウンド化)。-6 s の小さな効果のみ。`draw_bb3s` (CPU numpy 描画) が viz 時間の大半を占めるがスレッド化にはコード構造の refactor が要る。`--skip_viz` で丸ごとスキップする方が遥かに効くため **不採用** (コード自体は `BOXER_VIZ_ASYNC=1` で残置)。

### 検討のみ

**画像の物理共有**（両モデルで 1 つの 960×960 テンソルを使用）。単独効果 ~5–15 ms/frame (~3–8 s / 499)。主眼は並列化の土台。並列化を見送ったため同時に見送り。

**DinoV3 を FP16 運用**。-14% 高速化するが検出数 -7%。精度トレードオフの合意コストが見合わず見送り。

**head の ONNX 化**。`ObbTW` 構築の動的操作を含み ONNX 化困難。

**`torch_tensorrt` 導入**。Jetson サポート不十分で過去に断念済み。

---

## 5. 教訓

### Jetson では sync 点削減が最大のレバー

Jetson Orin は unified memory を持ち、CPU ↔ GPU の「コピー」は物理転送を伴わない。IOBinding で期待される帯域節約の効果は x86 と違いほぼゼロ。一方、**`.cpu()` や `torch.empty()` のたびに挟まる CUDA 同期点は確実にパイプラインを stall させる**。Jetson での最適化はここが主戦場。

### ORT CUDAExecutionProvider + IOBinding は同期必須

CUDA EP は private stream で動き、torch の default stream を自動では待たない。`io_binding.synchronize_inputs()` と `synchronize_outputs()` は **オプションではなく必須**。入れないと decision-deterministic な race (フレーム置きに 0 検出) として silent に壊れる。

### IOBinding 導入後は検出数の分布を必ず確認

wall-clock とトラッカー数だけでは半分スキップバグを検出できない。`owl_2dbbs.csv` / `boxer_3dbbs.csv` のフレームあたり検出数分布を見て、0-det が規則的に出ていないか確認する。本プロジェクトでは「交互 0 検出」症状が bug の発見につながった。

### 永続出力バッファは RAM コスト要注意

「per-frame `torch.empty` を消して同期点削減」は理に適っているが、ORT 内部のアクティベーションワークスペースが永続バインドで累積され、**ピーク RAM が 10 GB 単位で増える**ケースがある。メモリ制約の厳しい Jetson では効果・コストを個別測定してから採否判断すべき。

### 並列 ORT は avoid or ONNX マージ

複数 ORT セッション (とくに異なる EP) の並列呼出は、Jetson の Python/ORT 環境下では現実的に不可能に近い。どうしても必要なら ONNX グラフの物理マージか、ORT 内部 API への介入が必要。

### ベンチマークは full run で

TRT warmup・startup オーバーヘッド (~30 s) は 100 フレーム程度の短い run では 10–15% の誤差になる。最終確認は常に 499 フレーム fullで行う。一方、施策の correctness チェック（エラー有無・検出数）は 30 フレームで十分。

---

## 6. 参考: 関連ファイル

- 実装本体: [python/run_boxer_onnx.py](../python/run_boxer_onnx.py)
- ベンチマークスクリプト: [benchmark.sh](../benchmark.sh)
- env var 制御:
  - `BOXER_DISABLE_OPTIMIZED_CACHE=1` — 事前最適化 ONNX 無効化（Stage 5 の効果検証用）
  - `BOXER_DISABLE_PERSISTENT_BUF=1` — OWL 永続出力バッファ無効化（D2 採用構成）
  - `BOXER_VIZ_ASYNC=1` — imencode/write の非同期化（`--skip_viz` で不要なら未使用）
