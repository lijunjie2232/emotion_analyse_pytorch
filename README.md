# emotion analyse pytorch

emotion analyse by pytorch・Pytorchフレームワークを用いた感情認識

このプロジェクトは、ブログの記事「[pytorch 実践](https://blog.lijunjie.dpdns.org/2022/08/12/dl_pytorch_prct)」のサンプルコードです。

---
# 目次
- [emotion analyse pytorch](#emotion-analyse-pytorch)
- [目次](#目次)
  - [🧰 環境依存](#-環境依存)
  - [📁 データ準備](#-データ準備)
  - [🧠 モデル構造](#-モデル構造)
  - [⚙️ トレーニング設定](#️-トレーニング設定)
  - [🚀 使用方法](#-使用方法)
  - [📊 トレーニング結果](#-トレーニング結果)
  - [📄 参考文献](#-参考文献)


## 🧰 環境依存

```bash
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
kagglehub
tqdm
ipython
```

パッケージをインストール:

```bash
pip install -r requirements.tx
````

## 📁 データ準備
[Kaggle API認証情報を設定](https://www.kaggle.com/docs/api#authentication)

ダウンロードした`kaggle.json`をカレントディレクトリに配置し、以下のコマンドでパーミッションを設定する。

```bash
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 🧠 モデル構造
```python
EmotionNet(
  (backbone): ModuleList(
    [0] Conv(3, 16, k=3, s=2) 
    [1] Conv(16, 32, k=3, s=2)
    [2] C3k2(32, 64, n=1) 
    [3] Conv(64, 128, k=3, s=2)
    [4] C3k2(128, 128, n=1)
    [5] Conv(128, 128, k=3, s=2)
    [6] A2C2f(128, 128, n=2) 
    [7] Conv(128, 256, k=3, s=2)
    [8] A2C2f(256, 256, n=2)
  )
  (classify): Classify(256, 7)
)
```

## ⚙️ トレーニング設定
- **最適化アルゴリズム**：AdamW (lr=1e-3, betas=(0.9,0.999))
- **学習率スケジューラー**：StepLR (step_size=10, gamma=0.1)
- **損失関数**：交差エントロピー損失
- **混合精度**：GradScalerによる高速化
- **アーリーストッピング**：patience=20 epochs=120

## 🚀 使用方法
1. 依存パッケージをインストール：

```bash
pip install -r requirements.txt
```

2. トレーニングを開始：

- method 1: jupyter notebookを起動、main.ipynbを使う
- method 2: python main.pyを実行

3. トレーニングを再開：

```python
checkpoint = Path("last_model.pt")
if checkpoint.is_file():
    start_epoch = load(model=model, optimizer=optimizer, path=checkpoint)+1
```

## 📊 トレーニング結果
- 最高検証精度：`best_acc`変数に記録
- トレーニングログはNotebookのプログレスバーでリアルタイム表示

## 📄 参考文献
- ultralytics: https://github.com/ultralytics/ultralytics
- Area Attention: https://arxiv.org/abs/2108.09084
- CSPNet: https://arxiv.org/abs/1911.11929
