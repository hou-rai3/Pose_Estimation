"""
リアルタイムポーズ推定システム
=================================

【概要】
WebカメラからのリアルタイムVideo映像を使用して、人物の向きと腕の動きを分析するシステム

【主要機能】
1. GPU/CPU自動切り替えによる高速ポーズ推定
2. 人の向き検出（正面/左向き/右向き/後ろ向き）
3. 腕の動き分析（曲げ/伸ばし/上げ/下げ等）
4. リアルタイムデバッグ表示

【使用技術】
- MoveNet（Google製軽量ポーズ推定モデル）
- TensorFlow + TensorFlow Hub
- OpenCV（画像処理・表示）
"""

import cv2
import numpy as np
import time

import tensorflow as tf
import tensorflow_hub as hub

# ===================================================================
# GPU/CPU自動検出・設定
# ===================================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
USE_GPU = False
DEVICE_NAME = '/CPU:0'

if gpus:
  try:
    # GPU使用可能な場合の最適化設定（メモリ動的割り当て）
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    USE_GPU = True
    DEVICE_NAME = '/GPU:0'
    print(f"GPU使用設定: {len(gpus)}個のGPUを検出 - 常にGPUで実行します")
  except RuntimeError as e:
    print(f"GPU設定エラー: {e} - CPUで実行します")
    USE_GPU = False
    DEVICE_NAME = '/CPU:0'
else:
  print("GPU未検出: CPUで実行します")

# ===================================================================
# ポーズ推定パラメータ設定
# ===================================================================
KEYPOINT_THRESHOLD = 0.15  # キーポイント検出の信頼度閾値（低いほど敏感）
FRAME_SKIP = 1  # フレームスキップ間隔（1=2フレームに1回推論）
INPUT_SIZE = 192  # MoveNetモデル入力解像度（192x192固定）

# ===================================================================
# デバッグ・表示設定
# ===================================================================
SHOW_DEBUG_INFO = True    # コンソールにデバッグ情報を表示
CONSOLE_LOG_INTERVAL = 30  # 統計出力間隔（フレーム数）
SHOW_WINDOW = True        # 映像ウィンドウの表示（本番時はFalseに）
SHOW_DEBUG_OVERLAY = True  # キーポイント表示のON/OFF

def main():
  """
  メイン処理関数

  【処理フロー】
  1. MoveNetモデル読み込み
  2. Webカメラ初期化・設定最適化
  3. リアルタイム画像処理ループ
  4. 終了時統計表示
  """

  # ===================================================================
  # 1. AIモデル初期化（MoveNet軽量版）
  # ===================================================================
  print("モデル読み込み中...")
  model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
  movenet = model.signatures['serving_default']
  print("モデル読み込み完了！")

  # ===================================================================
  # 2. Webカメラ初期化・高精度設定
  # ===================================================================
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファ削減（遅延軽減）
  # カメラの解像度を上げて精度向上
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  # フレームレートを安定化
  cap.set(cv2.CAP_PROP_FPS, 30)

  # ===================================================================
  # 3. 処理状態管理変数
  # ===================================================================
  frame_count = 0      # 総フレーム数
  inference_count = 0  # 推論実行回数
  latest_result = None  # 最新の推論結果を保存（フレームスキップ時に再利用）

  # パフォーマンス測定用
  fps_start_time = time.time()
  frame_times = []     # フレーム時間を記録
  inference_times = []  # 推論時間を記録

  # ===================================================================
  # 4. メイン処理ループ（リアルタイム分析）
  # ===================================================================
  while (True):
    frame_start_time = time.time()

    # フレーム取得
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1

    # -----------------------------------
    # ポーズ推定実行（フレームスキップ適用）
    # -----------------------------------
    # フレームスキップによる処理軽量化：FRAME_SKIPフレームに1回だけ推論実行
    if frame_count % (FRAME_SKIP + 1) == 0:
      # 推論時間を測定
      inference_start = time.time()
      latest_result = run_inference(movenet, frame)
      inference_end = time.time()

      inference_count += 1
      inference_times.append(inference_end - inference_start)

    # -----------------------------------
    # 人の動作解析・結果出力
    # -----------------------------------
    # 人の向きと腕の動きを計算・出力
    current_direction = None
    arm_status = None
    if latest_result is not None:
      current_direction = calculate_person_direction(latest_result)  # 人の向き判定
      arm_status = analyze_arm_movement(latest_result)               # 腕の動き分析

      # 1行で向きと腕の状態を表示
      direction_text = current_direction if current_direction else "不明"
      arm_text = arm_status if arm_status else "検出なし"
      print(f"フレーム{frame_count}: 向き={direction_text} | 腕={arm_text}")

    # -----------------------------------
    # デバッグ用映像表示
    # -----------------------------------
    if SHOW_WINDOW:
      debug_frame = frame.copy()

      # 最低限のキーポイント表示（デバッグ用）
      if latest_result is not None and SHOW_DEBUG_OVERLAY:
        keypoints_list, scores_list, bbox_list = latest_result
        if keypoints_list and scores_list:
          keypoints = keypoints_list[0]
          scores = scores_list[0]

          # 主要なキーポイントのみ表示（顔部分+腕）
          important_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 鼻、目、耳、肩、肘、手首
          colors = [(0, 255, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255),
                    (0, 0, 255), (255, 255, 0), (255, 255, 0), (0, 255, 255),
                    (0, 255, 255), (255, 0, 255), (255, 0, 255)]

          for i, idx in enumerate(important_points):
            if idx < len(scores) and scores[idx] > KEYPOINT_THRESHOLD:
              x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
              cv2.circle(debug_frame, (x, y), 5, colors[i], -1)

      # デバッグ情報をフレームに表示
      if current_direction:
        cv2.putText(debug_frame, f"Direction: {current_direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
      if arm_status:
        cv2.putText(debug_frame, f"Arms: {arm_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
      cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

      cv2.imshow('Debug View', debug_frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q') or key == 27:  # QまたはESCで終了
        break
    else:
      # ウィンドウなしの場合のキー入力チェック
      key = cv2.waitKey(1) & 0xFF
      if key == 27:  # ESCキー
        break

    # -----------------------------------
    # パフォーマンス測定
    # -----------------------------------
    # フレーム時間を記録
    frame_end_time = time.time()
    frame_times.append(frame_end_time - frame_start_time)

    # 30フレームごとに詳細統計をコンソールに出力（コメントアウト）
    # if frame_count % CONSOLE_LOG_INTERVAL == 0:
    #   avg_fps = CONSOLE_LOG_INTERVAL / (time.time() - fps_start_time)
    #   avg_frame_time = np.mean(frame_times[-CONSOLE_LOG_INTERVAL:]) * 1000
    #   avg_inference_time_ms = np.mean(
    #       inference_times[-10:]) * 1000 if len(inference_times) > 0 else 0
    #   inference_fps = inference_count / (time.time() - fps_start_time)

    #   print(f"=== フレーム {frame_count} 統計 ===")
    #   print(f"平均FPS: {avg_fps:.1f}")
    #   print(f"平均フレーム時間: {avg_frame_time:.1f}ms")
    #   print(f"推論FPS: {inference_fps:.1f}")
    #   print(f"平均推論時間: {avg_inference_time_ms:.1f}ms")
    #   print(
    #       f"推論スキップ率: {((frame_count - inference_count) / frame_count * 100):.1f}%")
    #   print("=" * 30)

    #   fps_start_time = time.time()

  # ===================================================================
  # 5. 終了処理・リソース解放
  # ===================================================================
  cap.release()
  if SHOW_WINDOW:
    cv2.destroyAllWindows()

  # 終了時の総合統計
  if len(frame_times) > 0 and len(inference_times) > 0:
    total_time = sum(frame_times)
    avg_fps = len(frame_times) / total_time
    avg_frame_time = np.mean(frame_times) * 1000
    avg_inference_time = np.mean(inference_times) * 1000

    print("\n" + "=" * 50)
    print("総合統計")
    print("=" * 50)
    print(f"総フレーム数: {frame_count}")
    print(f"総推論回数: {inference_count}")
    print(f"平均FPS: {avg_fps:.2f}")
    print(f"平均フレーム時間: {avg_frame_time:.2f}ms")
    print(f"平均推論時間: {avg_inference_time:.2f}ms")
    print(
        f"推論スキップ率: {((frame_count - inference_count) / frame_count * 100):.1f}%")
    print(f"推論が全体時間に占める割合: {(avg_inference_time / avg_frame_time * 100):.1f}%")
    print("=" * 50)

def run_inference(model, image):
  """
  MoveNetを使用したポーズ推定実行

  【目的】
  入力画像から人体の17個のキーポイント（関節位置）を検出

  【処理内容】
  1. 画像前処理（リサイズ・色変換・データ型変換）
  2. MoveNet推論実行（GPU/CPU自動切り替え）
  3. 結果の後処理（座標変換・バウンディングボックス生成）
  """

  # ===================================================================
  # 画像前処理 - MoveNetの入力仕様に合わせて調整
  # ===================================================================
  input_image = cv2.resize(image, dsize=(
      INPUT_SIZE, INPUT_SIZE))  # 192x192にリサイズ
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)       # BGR→RGB変換

  # MoveNetモデルは192x192、int32を期待するため調整
  input_image = cv2.resize(input_image, dsize=(192, 192))  # モデル仕様確実適用
  input_image = np.expand_dims(input_image, 0)             # バッチ次元追加
  input_image = tf.cast(input_image, dtype=tf.int32)       # int32型に変換

  # ===================================================================
  # GPU/CPU推論実行（起動時に決定したデバイスで常時実行）
  # ===================================================================
  with tf.device(DEVICE_NAME):
    outputs = model(input_image)
    keypoints = outputs['output_0'].numpy()[0, 0]  # 単一人物用の出力形式

  # ===================================================================
  # 結果の後処理 - 座標変換とデータ整形
  # ===================================================================
  image_height, image_width = image.shape[:2]

  # 単一人物のキーポイント処理（ループを削減）
  kp_coords = keypoints[:51].reshape(17, 3)  # 17個のキーポイント x (y, x, score)

  # 座標変換を一括処理（正規化座標→実際の画像座標）
  kp_x = (image_width * kp_coords[:, 1]).astype(int)   # X座標
  kp_y = (image_height * kp_coords[:, 0]).astype(int)  # Y座標
  scores = kp_coords[:, 2]                             # 信頼度

  keypoints_list = [np.column_stack([kp_x, kp_y]).tolist()]
  scores_list = [scores.tolist()]

  # 簡易バウンディングボックス（有効なキーポイントから計算）
  valid_points = kp_coords[scores > KEYPOINT_THRESHOLD]
  if len(valid_points) > 0:
    valid_x = (image_width * valid_points[:, 1]).astype(int)
    valid_y = (image_height * valid_points[:, 0]).astype(int)
    bbox_xmin = max(0, int(np.min(valid_x)) - 20)
    bbox_ymin = max(0, int(np.min(valid_y)) - 20)
    bbox_xmax = min(image_width, int(np.max(valid_x)) + 20)
    bbox_ymax = min(image_height, int(np.max(valid_y)) + 20)
    bbox_score = float(np.mean(scores[scores > KEYPOINT_THRESHOLD]))
  else:
    bbox_xmin = bbox_ymin = bbox_xmax = bbox_ymax = 0
    bbox_score = 0.0

  bbox_list = [[bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score]]

  return keypoints_list, scores_list, bbox_list

def calculate_person_direction(result):
  """
  人の向き判定システム

  【目的】
  ポーズ推定結果から人がどの方向を向いているかを判定

  【判定方法】
  1. 顔の特徴分析（鼻・目の位置関係） - 最優先
  2. 耳の可視性分析 - 副次的判定
  3. 肩の位置関係分析 - 補完的判定

  戻り値: "正面", "左向き", "右向き", "後ろ向き", None
  """
  keypoints_list, scores_list, bbox_list = result

  if not keypoints_list or not scores_list:
    return None

  keypoints = keypoints_list[0]  # 最初の人物
  scores = scores_list[0]

  # ===================================================================
  # MoveNetキーポイントインデックス定義
  # ===================================================================
  nose = 0           # 鼻
  left_eye = 1       # 左目
  right_eye = 2      # 右目
  left_ear = 3       # 左耳
  right_ear = 4      # 右耳
  left_shoulder = 5  # 左肩
  right_shoulder = 6  # 右肩

  # 信頼度チェック
  if len(scores) < 7:
    return None

  # 有効なキーポイントの座標と信頼度を取得
  def get_point_info(idx):
    if idx < len(scores) and scores[idx] > KEYPOINT_THRESHOLD:
      return keypoints[idx], scores[idx], True
    return None, 0, False

  nose_pos, nose_score, nose_valid = get_point_info(nose)
  left_eye_pos, left_eye_score, left_eye_valid = get_point_info(left_eye)
  right_eye_pos, right_eye_score, right_eye_valid = get_point_info(right_eye)
  left_ear_pos, left_ear_score, left_ear_valid = get_point_info(left_ear)
  right_ear_pos, right_ear_score, right_ear_valid = get_point_info(right_ear)
  left_shoulder_pos, left_shoulder_score, left_shoulder_valid = get_point_info(
      left_shoulder)
  right_shoulder_pos, right_shoulder_score, right_shoulder_valid = get_point_info(
      right_shoulder)

  # ===================================================================
  # 判定ロジック：座標位置を重視した分析
  # ===================================================================

  # 1. 顔の特徴による判定（最重要）- 座標幾何学による精密判定
  if nose_valid:
    nose_x = nose_pos[0]

    # 両目が見える場合の判定
    if left_eye_valid and right_eye_valid:
      left_eye_x = left_eye_pos[0]
      right_eye_x = right_eye_pos[0]

      # 目の間隔と鼻の位置関係で判定
      eye_distance = abs(left_eye_x - right_eye_x)
      nose_to_left_eye = abs(nose_x - left_eye_x)
      nose_to_right_eye = abs(nose_x - right_eye_x)

      if eye_distance > 20:  # 目が十分離れている = 正面に近い
        # 鼻が目の中央付近にある
        if abs(nose_to_left_eye - nose_to_right_eye) < eye_distance * 0.3:
          return "正面"
        # 鼻が右目に近い = 左向き
        elif nose_to_right_eye < nose_to_left_eye:
          return "左向き"
        # 鼻が左目に近い = 右向き
        else:
          return "右向き"
      else:
        return "正面"  # 目が近い = 正面

    # 片目のみ見える場合
    elif left_eye_valid and not right_eye_valid:
      # 左目のみ見える場合の詳細判定
      if left_ear_valid:
        # 左耳も見える = 右を向いている
        return "右向き"
      else:
        return "右向き"  # 左目のみ = 通常右向き

    elif right_eye_valid and not left_eye_valid:
      # 右目のみ見える場合の詳細判定
      if right_ear_valid:
        # 右耳も見える = 左を向いている
        return "左向き"
      else:
        return "左向き"  # 右目のみ = 通常左向き

  # 2. 耳の位置関係による判定
  if left_ear_valid and right_ear_valid:
    # 両耳見える = 正面または後ろ向き
    if nose_valid or left_eye_valid or right_eye_valid:
      return "正面"
    else:
      return "後ろ向き"
  elif left_ear_valid and not right_ear_valid:
    return "左向き"
  elif right_ear_valid and not left_ear_valid:
    return "右向き"

  # 3. 肩の位置関係による判定
  if left_shoulder_valid and right_shoulder_valid:
    shoulder_distance = abs(left_shoulder_pos[0] - right_shoulder_pos[0])

    if shoulder_distance > 50:  # 肩が十分離れている = 正面
      if not nose_valid and not left_eye_valid and not right_eye_valid:
        return "後ろ向き"
      else:
        return "正面"
    else:
      # 肩が近い = 横向き、より高い肩の方向を向いている
      if left_shoulder_pos[1] < right_shoulder_pos[1]:  # 左肩が上
        return "右向き"
      else:
        return "左向き"

  return None  # 判定不能

def analyze_arm_movement(result):
  """
  腕の動き分析システム

  【目的】
  ポーズ推定結果から腕の状態（曲げ/伸ばし/上げ/下げ等）を分析

  【分析内容】
  1. 肘の角度計算：ベクトル内積による角度測定
  2. 手の位置判定：肩との相対位置による上げ/下げ判定
  3. 両手の関係性：接近度・同期度の分析

  戻り値: 腕の状態を表す文字列
  """
  keypoints_list, scores_list, bbox_list = result

  if not keypoints_list or not scores_list:
    return None

  keypoints = keypoints_list[0]
  scores = scores_list[0]

  # ===================================================================
  # 腕関連のキーポイントインデックス
  # ===================================================================
  left_shoulder = 5   # 左肩
  right_shoulder = 6  # 右肩
  left_elbow = 7      # 左肘
  right_elbow = 8     # 右肘
  left_wrist = 9      # 左手首
  right_wrist = 10    # 右手首

  def get_point_info(idx):
    if idx < len(scores) and scores[idx] > KEYPOINT_THRESHOLD:
      return keypoints[idx], scores[idx], True
    return None, 0, False

  # 各関節の情報取得
  l_shoulder_pos, _, l_shoulder_valid = get_point_info(left_shoulder)
  r_shoulder_pos, _, r_shoulder_valid = get_point_info(right_shoulder)
  l_elbow_pos, _, l_elbow_valid = get_point_info(left_elbow)
  r_elbow_pos, _, r_elbow_valid = get_point_info(right_elbow)
  l_wrist_pos, _, l_wrist_valid = get_point_info(left_wrist)
  r_wrist_pos, _, r_wrist_valid = get_point_info(right_wrist)

  arm_states = []

  # ===================================================================
  # 左腕の分析 - 角度計算と位置判定
  # ===================================================================
  if l_shoulder_valid and l_elbow_valid:
    if l_wrist_valid:
      # 完全な左腕のキーポイントがある場合
      shoulder_to_elbow = np.array(l_elbow_pos) - np.array(l_shoulder_pos)
      elbow_to_wrist = np.array(l_wrist_pos) - np.array(l_elbow_pos)

      # 角度計算（肘の曲がり具合）- ベクトル内積による計算
      dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
      norms = np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist)

      if norms > 0:
        cos_angle = dot_product / norms
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

        if angle < 60:
          arm_states.append("左腕：曲げ")
        elif angle > 140:
          arm_states.append("左腕：伸ばし")
        else:
          arm_states.append("左腕：中間")

      # 手の位置による動作判定
      if l_wrist_pos[1] < l_shoulder_pos[1] - 30:  # 手が肩より上
        arm_states.append("左手上げ")
      elif l_wrist_pos[1] > l_shoulder_pos[1] + 30:  # 手が肩より下
        arm_states.append("左手下げ")

  # ===================================================================
  # 右腕の分析 - 左腕と同様の処理
  # ===================================================================
  if r_shoulder_valid and r_elbow_valid:
    if r_wrist_valid:
      # 完全な右腕のキーポイントがある場合
      shoulder_to_elbow = np.array(r_elbow_pos) - np.array(r_shoulder_pos)
      elbow_to_wrist = np.array(r_wrist_pos) - np.array(r_elbow_pos)

      # 角度計算（肘の曲がり具合）
      dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
      norms = np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist)

      if norms > 0:
        cos_angle = dot_product / norms
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

        if angle < 60:
          arm_states.append("右腕：曲げ")
        elif angle > 140:
          arm_states.append("右腕：伸ばし")
        else:
          arm_states.append("右腕：中間")

      # 手の位置による動作判定
      if r_wrist_pos[1] < r_shoulder_pos[1] - 30:  # 手が肩より上
        arm_states.append("右手上げ")
      elif r_wrist_pos[1] > r_shoulder_pos[1] + 30:  # 手が肩より下
        arm_states.append("右手下げ")

  # ===================================================================
  # 両手の相対位置分析 - 接近度・同期度の判定
  # ===================================================================
  if l_wrist_valid and r_wrist_valid:
    wrist_distance = np.linalg.norm(
        np.array(l_wrist_pos) - np.array(r_wrist_pos))
    if wrist_distance < 50:
      arm_states.append("両手接近")

    # 両手の高さ比較
    if abs(l_wrist_pos[1] - r_wrist_pos[1]) < 20:
      if l_shoulder_valid and r_shoulder_valid:
        avg_shoulder_y = (l_shoulder_pos[1] + r_shoulder_pos[1]) / 2
        if l_wrist_pos[1] < avg_shoulder_y - 30:
          arm_states.append("両手上げ")

  return ", ".join(arm_states) if arm_states else None

def render(image, keypoints_list, scores_list, bbox_list):
  """
  骨格描画関数（旧版互換性維持用）

  【注意】現在はデバッグ表示で代替しているため、使用されていません
  """
  render = image.copy()

  # 主要な骨格のみ描画（処理軽減）
  kp_links = [
      (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # 腕
      (11, 12), (5, 11), (6, 12),                  # 胴体
      (11, 13), (13, 15), (12, 14), (14, 16)      # 脚
  ]

  for i, (keypoints, scores, bbox) in enumerate(zip(keypoints_list, scores_list, bbox_list)):
    if bbox[4] < 0.1:  # 閾値を下げて検出しやすく
      continue

    # バウンディングボックス描画
    cv2.rectangle(render, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 255, 0), 2)

    # NumPy配列に変換して高速化
    keypoints_np = np.array(keypoints)
    scores_np = np.array(scores)
    valid_mask = scores_np > KEYPOINT_THRESHOLD

    # 骨格線の描画 - 軽量化
    for kp_idx_1, kp_idx_2 in kp_links:
      if kp_idx_1 < len(valid_mask) and kp_idx_2 < len(valid_mask):
        if valid_mask[kp_idx_1] and valid_mask[kp_idx_2]:
          pt1 = tuple(keypoints_np[kp_idx_1].astype(int))
          pt2 = tuple(keypoints_np[kp_idx_2].astype(int))
          cv2.line(render, pt1, pt2, (0, 0, 255), 2)

    # 重要なキーポイントのみ描画
    important_points = [0, 5, 6, 11, 12]  # 鼻、左右肩、左右腰
    for idx in important_points:
      if idx < len(valid_mask) and valid_mask[idx]:
        cv2.circle(render, tuple(
            keypoints_np[idx].astype(int)), 4, (255, 0, 0), -1)

  return render

# ===================================================================
# プログラム実行部
# ===================================================================
if __name__ == '__main__':
  main()
