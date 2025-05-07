# emotion_analyse_pytorch
emotion analyse by pytorch・Pytorchフレームワークを用いた感情認識

## 🧰 環境依存

```bash
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
kagglehub
tqdm
ipython
```

```bash
pip install -r requirements.txt
````

## 📁 データ準備
1. Kaggle CLIをインストール：
```bash
pip install kaggle
```
2. API認証情報を設定：
ダウンロードした`kaggle.json`をカレントディレクトリに配置し、以下のコマンドでパーミッションを設定する。
```bash
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
