import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm
import multiprocessing as mproc
from functools import partial
import os

print("="*60)
print("STEP 1: EXTRACT FEATURES (ULTRA-FAST VERSION)")
print("="*60)

# --- OPTIMIZED CONFIG ---
TARGET_FRAMES = 100
FRAME_SKIP = 3           # ⚡ Tăng từ 2 lên 3 (tốc độ +50%)
VISIBILITY_THR = 0.5
MODEL_COMPLEXITY = 0     # ⚡ Giảm từ 1 xuống 0 (tốc độ +30%)
RESIZE_WIDTH = 480       # ⚡ Resize video về 480p (tốc độ +40%)
NUM_WORKERS = 4          # ⚡ Số CPU cores (adjust theo máy bạn)
USE_GPU = False          # ⚡ Dùng GPU nếu có

# --- HELPER FUNCTIONS ---
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

def compute_distance(a, b):
    return np.linalg.norm(a - b)

def interpolate_features(features, target_len=100):
    if len(features) < 2:
        return features
    x_old = np.linspace(0, 1, len(features))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, features, axis=0, kind='linear')
    return f(x_new)

# --- OPTIMIZED EXTRACTION ---
def extract_pose_landmarks_fast(video_path):
    """⚡ Version tối ưu: GPU + Resize + Frame skip"""
    mp_pose = mp.solutions.pose
    
    # GPU config (nếu có)
    if USE_GPU:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False  # Tắt segmentation để tăng tốc
        )
    else:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video info
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or np.isnan(original_fps):
        original_fps = 30.0
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate resize ratio
    if original_width > RESIZE_WIDTH:
        resize_ratio = RESIZE_WIDTH / original_width
        new_width = RESIZE_WIDTH
        new_height = int(original_height * resize_ratio)
    else:
        new_width = original_width
        new_height = original_height
    
    frames_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ⚡ Skip frames
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # ⚡ Resize frame
        if new_width != original_width:
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Quick visibility check
            avg_vis = np.mean([p.visibility for p in lm])
            if avg_vis < VISIBILITY_THR:
                continue
            
            # Extract coordinates (no visibility to save memory)
            coords = []
            for landmark in lm:
                coords.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            frames_data.append(coords)
    
    cap.release()
    pose.close()
    
    effective_fps = original_fps / FRAME_SKIP
    return np.array(frames_data) if len(frames_data) > 10 else None, effective_fps

def extract_216_features(pose_data, fps):
    """Extract features (giữ nguyên logic)"""
    if pose_data is None or len(pose_data) == 0:
        return None
    
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    
    features_per_frame = []
    
    for frame in pose_data:
        landmarks = frame.reshape(33, 4)[:, :3]
        
        l_shoulder = landmarks[LEFT_SHOULDER]
        r_shoulder = landmarks[RIGHT_SHOULDER]
        l_hip = landmarks[LEFT_HIP]
        r_hip = landmarks[RIGHT_HIP]
        l_elbow = landmarks[LEFT_ELBOW]
        r_elbow = landmarks[RIGHT_ELBOW]
        l_wrist = landmarks[LEFT_WRIST]
        r_wrist = landmarks[RIGHT_WRIST]
        l_knee = landmarks[LEFT_KNEE]
        r_knee = landmarks[RIGHT_KNEE]
        l_ankle = landmarks[LEFT_ANKLE]
        r_ankle = landmarks[RIGHT_ANKLE]
        
        torso_size_l = compute_distance(l_shoulder, l_hip)
        torso_size_r = compute_distance(r_shoulder, r_hip)
        body_scale = (torso_size_l + torso_size_r) / 2.0
        if body_scale == 0: body_scale = 1.0
        
        angles = [
            compute_angle(l_shoulder, l_elbow, l_wrist),
            compute_angle(r_shoulder, r_elbow, r_wrist),
            compute_angle(l_elbow, l_shoulder, l_hip),
            compute_angle(r_elbow, r_shoulder, r_hip),
            compute_angle(l_shoulder, l_hip, l_knee),
            compute_angle(r_shoulder, r_hip, r_knee)
        ]
        
        distances = [
            compute_distance(l_shoulder, r_shoulder),
            compute_distance(l_hip, r_hip),
            compute_distance(l_shoulder, l_wrist),
            compute_distance(r_shoulder, r_wrist),
            compute_distance(l_hip, l_ankle),
            compute_distance(r_hip, r_ankle)
        ]
        
        normalized_distances = [d / body_scale for d in distances]
        features_per_frame.append(angles + normalized_distances)
    
    features_array = np.array(features_per_frame)
    
    # Velocities
    velocities = []
    dt = 1.0 / fps
    
    for i in range(len(features_array)):
        if i == 0:
            vel = [0] * 6
        else:
            curr = pose_data[i].reshape(33, 4)[:, :3]
            prev = pose_data[i-1].reshape(33, 4)[:, :3]
            
            prev_scale = (compute_distance(prev[LEFT_SHOULDER], prev[LEFT_HIP]) + 
                          compute_distance(prev[RIGHT_SHOULDER], prev[RIGHT_HIP])) / 2.0
            if prev_scale == 0: prev_scale = 1.0
            
            joints = [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, 
                     RIGHT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER]
            
            vel = []
            for j_idx in joints:
                dist = compute_distance(curr[j_idx], prev[j_idx])
                v = (dist / dt) / prev_scale
                vel.append(v)
        velocities.append(vel)
    
    velocities_array = np.array(velocities)
    all_raw_features = np.hstack([features_array, velocities_array])
    interpolated_features = interpolate_features(all_raw_features, target_len=TARGET_FRAMES)
    
    # Statistics
    final_vector = []
    for i in range(interpolated_features.shape[1]):
        series = interpolated_features[:, i]
        stats_vec = [
            np.mean(series), np.std(series), np.min(series), np.max(series),
            np.median(series), np.ptp(series), np.percentile(series, 25),
            np.percentile(series, 75), stats.skew(series), stats.kurtosis(series),
            series[0], series[-1]
        ]
        final_vector.extend(stats_vec)
    
    return np.array(final_vector)

# --- MULTIPROCESSING WORKER ---
def process_single_video(row):
    """Worker function cho multiprocessing"""
    video_path = row['full_path']
    filename = row['filename']
    video_id = row['video_id']
    view = row['view']
    
    try:
        pose_data, fps = extract_pose_landmarks_fast(video_path)
        
        if pose_data is None:
            return None
        
        features = extract_216_features(pose_data, fps)
        
        if features is None:
            return None
        
        return {
            'filename': filename,
            'full_path': video_path,
            'video_id': video_id,
            'features': features.tolist(),
            'view': view,
            'level': 'pro'
        }
    except Exception as e:
        print(f"\n❌ Error {filename}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("\n📂 Loading metadata...")
    try:
        df = pd.read_csv('dataset_metadata_pro.csv')
        print(f"✅ Found {len(df)} videos")
        print(f"   - Backview: {len(df[df['view']=='back'])}")
        print(f"   - Sideview: {len(df[df['view']=='side'])}")
        
        # Detect CPU cores
        available_cores = mproc.cpu_count()
        print(f"\n⚙️  System: {available_cores} CPU cores detected")
        print(f"   Using: {NUM_WORKERS} workers")
        print(f"   FRAME_SKIP: {FRAME_SKIP}")
        print(f"   MODEL_COMPLEXITY: {MODEL_COMPLEXITY}")
        print(f"   RESIZE_WIDTH: {RESIZE_WIDTH}px")
        print()
    except FileNotFoundError:
        print("❌ dataset_metadata_pro.csv not found!")
        exit()
    
    # Tách theo view
    df_back = df[df['view'] == 'back'].reset_index(drop=True)
    df_side = df[df['view'] == 'side'].reset_index(drop=True)
    
    # Process từng view với multiprocessing
    for view_name, df_view in [('back', df_back), ('side', df_side)]:
        print("="*60)
        print(f"PROCESSING {view_name.upper()} VIEW ({len(df_view)} videos)")
        print("="*60)
        
        # Convert DataFrame to list of dicts
        rows = df_view.to_dict('records')
        
        # ⚡ MULTIPROCESSING
        print(f"\n🚀 Starting {NUM_WORKERS} parallel workers...")
        with mproc.Pool(processes=NUM_WORKERS) as pool:
            results_raw = list(tqdm(
                pool.imap(process_single_video, rows),
                total=len(rows),
                desc=f"{view_name} view"
            ))
        
        # Filter out None results
        results = [r for r in results_raw if r is not None]
        failed = len(rows) - len(results)
        
        # Save results
        print(f"\n📊 {view_name.upper()} VIEW SUMMARY:")
        print(f"   ✅ Success: {len(results)}/{len(df_view)}")
        print(f"   ❌ Failed: {failed}")
        
        if len(results) > 0:
            # Convert to DataFrame
            features_list = [r['features'] for r in results]
            features_array = np.array(features_list)
            
            feature_df = pd.DataFrame(
                features_array,
                columns=[f'feat_{i}' for i in range(216)]
            )
            
            feature_df['filename'] = [r['filename'] for r in results]
            feature_df['full_path'] = [r['full_path'] for r in results]
            feature_df['video_id'] = [r['video_id'] for r in results]
            feature_df['view'] = view_name
            feature_df['level'] = 'pro'
            
            # Save
            csv_filename = f'features_{view_name}_view.csv'
            feature_df.to_csv(csv_filename, index=False)
            print(f"   💾 Saved: {csv_filename}")
            
            json_filename = f'features_{view_name}_view.json'
            with open(json_filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   💾 Saved: {json_filename}")
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
