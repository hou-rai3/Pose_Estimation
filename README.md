# Pose_Estimation

このリポジトリは、Webカメラ映像からリアルタイムで人物の向きや腕の動きを推定するPythonプログラムです。

## 特徴
- MoveNet（Google製）による高速・高精度な単一人物ポーズ推定
- GPU自動利用（環境に応じて自動切替）
- 人の向き（正面/左右/後ろ）と腕の動き（曲げ/伸ばし/上げ/下げ等）をリアルタイムで判定

## 必要環境
- Python 3.8～3.11
- Windows 10/11（Webカメラ必須）
- pip

## セットアップ手順

1. 仮想環境の作成（推奨）

```powershell
python -m venv pose_env
pose_env\Scripts\activate
```

2. 依存パッケージのインストール

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. プログラムの実行

```powershell
python Pose_Estimation.py
```

## 共同開発者向けメモ
- `pose_env/` フォルダはリポジトリに含めていません。各自で仮想環境を作成してください。
- 必要なパッケージは `requirements.txt` で管理しています。
- コードやREADMEの改善プルリク歓迎！

## トラブルシューティング
- TensorFlowのインストールでエラーが出る場合は、`pip install tensorflow-cpu` も試してください。
- Webカメラが認識されない場合は、他のカメラデバイス番号（例: `cv2.VideoCapture(1)`）を試してください。

---

ご不明点はGitHubのIssueまたは開発者までご連絡ください。
