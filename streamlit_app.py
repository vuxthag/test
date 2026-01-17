# 🏌️ Phân Tích Golf Swing Pro - Phiên Bản Nâng Cấp
# Hệ Thống Phân Tích Sinh Cơ Học Golf Bằng AI
# Data Storm Competition 2025 - VTK Team

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import joblib
from scipy import stats
from scipy.interpolate import interp1d
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime
# ===================================================== 
# CẤU HÌNH TRANG
# =====================================================
st.set_page_config(
    page_title="Phân Tích Golf Swing Pro",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - BACKGROUND + TOOLTIP [FIXED]
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #e0e7ff 0%, #fce7f3 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #fce7f3 100%);
    }
    
    /* ========== MÀU CHỮ XÁM ĐẬM ========== */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #1e293b !important;
    }
    
    .css-1d391kg, .css-1d391kg p, [data-testid="stSidebar"] .stMarkdown {
        color: #1e293b !important;
    }
    
    /* ========== FIX: EXPANDER TEXT COLOR ========== */
    .streamlit-expanderContent, .streamlit-expanderContent p, .streamlit-expanderContent li {
        color: #1e293b !important;
    }
    
    .streamlit-expanderHeader {
        color: #1e293b !important;
    }
    
    div[data-testid="stExpander"] p, 
    div[data-testid="stExpander"] li,
    div[data-testid="stExpander"] span {
        color: #1e293b !important;
    }
    
    .element-container, .stText {
        color: #1e293b !important;
    }
    
    .stCaption, small, .css-16huue1 {
        color: #475569 !important;
    }
    
    .stRadio label {
        color: #1e293b !important;
    }
    
    .uploadedFileName {
        color: #1e293b !important;
    }
    
    /* ========== TOOLTIP STYLING ========== */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 2px dotted #667eea;
        cursor: help;
        color: #1e293b;
        font-weight: 600;
    }
    
    .tooltip .tooltiptext {
    visibility: hidden;
    width: 300px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Gradient tím */
    color: #ffffff;                /* Chữ trắng */
    text-align: left;
    border-radius: 8px;
    padding: 12px 16px;
    position: absolute;
    z-index: 9999;
    bottom: 125%;
    left: 50%;
    margin-left: -150px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 13px;
    line-height: 1.6;
    font-weight: 500;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
}

.tooltip .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #667eea transparent transparent transparent;  /* Mũi tên tím */
}

    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* ========== BUTTONS ========== */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    
    .score-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .badge-excellent {
        background: #10b981;
        color: white;
    }
    
    .badge-good {
        background: #3b82f6;
        color: white;
    }
    
    .badge-average {
        background: #f59e0b;
        color: white;
    }
    
    .badge-poor {
        background: #ef4444;
        color: white;
    }
    
    h1 {
        color: #1e293b !important;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #334155 !important;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #1e293b !important;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
    }
    
    .impact-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .impact-value {
        font-size: 32px;
        font-weight: 700;
        color: #667eea;
    }
    
    .impact-label {
        font-size: 14px;
        color: #64748b;
        margin-top: 5px;
    }
    
    .download-section {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .stProgress > div > div {
        color: #1e293b !important;
    }
    
    /* Comparison Video Cards */
    .comparison-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# ĐỊNH NGHĨA CHỈ SỐ (TOOLTIP CONTENT)
# =====================================================
METRIC_DEFINITIONS = {
    "Điểm AI": "Điểm tổng hợp được tính bằng thuật toán AI dựa trên khoảng cách của bạn so với golfer chuyên nghiệp. Điểm càng cao càng giống phong cách PRO.",
    
    "Góc Quay": "Góc chụp video - Back View (nhìn từ phía sau) hoặc Side View (nhìn từ bên hông). AI tự động phát hiện góc tối ưu.",
    
    "Percentile": "Vị trí xếp hạng của bạn so với các golfer chuyên nghiệp. P50 = giữa bảng xếp hạng, P10 = top 10%.",
    
    "Độ Tin Cậy": "Mức độ chắc chắn của AI model khi đánh giá swing của bạn. Cao hơn = đánh giá chính xác hơn.",
    
    "Tính Nhất Quán": "Đo độ ổn định của tư thế trong suốt cú swing. 100% = rất ổn định, thấp = cần cải thiện stability.",
    
    "Vận Tốc Tối Đa": "Tốc độ di chuyển nhanh nhất của cổ tay trong swing. Phản ánh sức mạnh và timing của cú đánh.",
    
    "Góc Gấp Đầu Gối": "Góc gấp đầu gối tại thời điểm impact. Góc tối ưu giúp chuyển lực hiệu quả từ chân lên thân.",
    
    "Góc Cánh Tay": "Góc giữa vai-khuỷu-cổ tay tại đỉnh backswing. Góc đúng giúp tạo lực xoay và tăng khoảng cách.",
    
    "Distance": "Khoảng cách Euclidean giữa đặc trưng của bạn và tâm của nhóm PRO. Càng nhỏ = càng giống PRO.",
    
    "FPS": "Frames Per Second - số khung hình xử lý mỗi giây. Cao hơn = phân tích chi tiết hơn.",
}

def create_tooltip(text, definition):
    """Tạo text có tooltip"""
    return f'<span class="tooltip">{text}<span class="tooltiptext">{definition}</span></span>'

# =====================================================
# CONFIG
# =====================================================
TARGET_FRAMES = 100
FRAME_SKIP = 3
VISIBILITY_THR = 0.5
MODEL_COMPLEXITY = 0
RESIZE_WIDTH = 480

# =====================================================
# HELPER FUNCTIONS
# =====================================================
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

def get_score_color(score):
    if score >= 85:
        return "#10b981"
    elif score >= 70:
        return "#3b82f6"
    elif score >= 55:
        return "#f59e0b"
    else:
        return "#ef4444"

def get_score_label(score):
    if score >= 85:
        return "Cấp Độ PRO 🏆"
    elif score >= 70:
        return "Nâng Cao ⭐"
    elif score >= 55:
        return "Trung Cấp 📊"
    else:
        return "Mới Bắt Đầu 💪"

def get_badge_class(score):
    if score >= 85:
        return "badge-excellent"
    elif score >= 70:
        return "badge-good"
    elif score >= 55:
        return "badge-average"
    else:
        return "badge-poor"

def detect_view_advanced(pose_data):
    if len(pose_data) < 20:
        return 'side', []
    
    sample_frames = [10, 20, 30, 40, 50]
    sample_frames = [f for f in sample_frames if f < len(pose_data)]
    
    votes = {'back': 0, 'side': 0}
    debug_info = []
    
    for frame_idx in sample_frames:
        lm = pose_data[frame_idx].reshape(33, 4)[:, :3]
        
        l_shoulder = lm[11]
        r_shoulder = lm[12]
        l_hip = lm[23]
        r_hip = lm[24]
        
        shoulder_width_x = abs(l_shoulder[0] - r_shoulder[0])
        hip_width_x = abs(l_hip[0] - r_hip[0])
        shoulder_depth = abs(l_shoulder[2] - r_shoulder[2])
        
        frame_score = {'back': 0, 'side': 0}
        
        if shoulder_width_x > 0.28:
            frame_score['back'] += 3
        elif shoulder_width_x > 0.20:
            frame_score['back'] += 2
        else:
            frame_score['side'] += 2
        
        if hip_width_x > 0.22:
            frame_score['back'] += 2
        elif hip_width_x < 0.12:
            frame_score['side'] += 2
        
        if shoulder_depth > 0.15:
            frame_score['side'] += 3
        
        if frame_score['back'] > frame_score['side']:
            votes['back'] += 1
        else:
            votes['side'] += 1
        
        debug_info.append({
            'frame': frame_idx,
            'shoulder_x': shoulder_width_x,
            'score_back': frame_score['back'],
            'score_side': frame_score['side']
        })
    
    final_view = 'back' if votes['back'] > votes['side'] else 'side'
    return final_view, debug_info

def extract_pose_landmarks(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or np.isnan(original_fps):
        original_fps = 30.0
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if original_width > RESIZE_WIDTH:
        new_width = RESIZE_WIDTH
        new_height = int(original_height * RESIZE_WIDTH / original_width)
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
        if frame_count % FRAME_SKIP != 0:
            continue
        
        if new_width != original_width:
            frame = cv2.resize(frame, (new_width, new_height))
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            avg_vis = np.mean([p.visibility for p in lm])
            if avg_vis < VISIBILITY_THR:
                continue
            
            coords = []
            for landmark in lm:
                coords.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            frames_data.append(coords)
    
    cap.release()
    pose.close()
    
    effective_fps = original_fps / FRAME_SKIP
    pose_data = np.array(frames_data) if len(frames_data) > 10 else None
    
    if pose_data is not None:
        detected_view, debug_info = detect_view_advanced(pose_data)
    else:
        detected_view = 'side'
        debug_info = []
    
    return pose_data, effective_fps, detected_view, debug_info

def extract_216_features(pose_data, fps):
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
        if body_scale == 0: 
            body_scale = 1.0
        
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
            if prev_scale == 0: 
                prev_scale = 1.0
            
            joints = [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER]
            
            vel = []
            for j_idx in joints:
                dist = compute_distance(curr[j_idx], prev[j_idx])
                v = (dist / dt) / prev_scale
                vel.append(v)
        velocities.append(vel)
    
    velocities_array = np.array(velocities)
    all_raw_features = np.hstack([features_array, velocities_array])
    interpolated_features = interpolate_features(all_raw_features, target_len=TARGET_FRAMES)
    
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

def calculate_top_metrics(pose_data, fps):
    """Tính toán các chỉ số sinh cơ học quan trọng"""
    if pose_data is None or len(pose_data) == 0:
        return None
    
    metrics = {}
    
    mid_frame = len(pose_data) // 2
    impact_frame = int(len(pose_data) * 0.7)
    
    for frame_idx, frame_name in [(0, 'setup'), (mid_frame, 'top'), (impact_frame, 'impact')]:
        if frame_idx < len(pose_data):
            landmarks = pose_data[frame_idx].reshape(33, 4)[:, :3]
            
            l_elbow_angle = compute_angle(landmarks[11], landmarks[13], landmarks[15])
            r_elbow_angle = compute_angle(landmarks[12], landmarks[14], landmarks[16])
            l_knee_angle = compute_angle(landmarks[23], landmarks[25], landmarks[27])
            r_knee_angle = compute_angle(landmarks[24], landmarks[26], landmarks[28])
            
            shoulder_line = landmarks[12] - landmarks[11]
            hip_line = landmarks[24] - landmarks[23]
            
            metrics[frame_name] = {
                'left_arm_angle': l_elbow_angle,
                'right_arm_angle': r_elbow_angle,
                'left_knee_flex': l_knee_angle,
                'right_knee_flex': r_knee_angle,
                'posture_height': landmarks[0][1]
            }
    
    if len(pose_data) > 1:
        wrist_movements = []
        for i in range(1, len(pose_data)):
            curr_wrist = pose_data[i].reshape(33, 4)[15, :3]
            prev_wrist = pose_data[i-1].reshape(33, 4)[15, :3]
            movement = compute_distance(curr_wrist, prev_wrist)
            wrist_movements.append(movement)
        
        max_velocity = max(wrist_movements) * fps if wrist_movements else 0
        avg_velocity = np.mean(wrist_movements) * fps if wrist_movements else 0
    else:
        max_velocity = 0
        avg_velocity = 0
    
    metrics['velocity'] = {
        'max': max_velocity,
        'avg': avg_velocity
    }
    
    if len(pose_data) > 5:
        heights = [pose_data[i].reshape(33, 4)[0, 1] for i in range(len(pose_data))]
        consistency = 100 - (np.std(heights) * 1000)
        consistency = np.clip(consistency, 0, 100)
    else:
        consistency = 50
    
    metrics['consistency'] = consistency
    
    return metrics

# =====================================================
# VISUALIZATION FUNCTIONS - THUẦN VIỆT
# =====================================================
def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#1a1a1a', 'family': 'Poppins'}},
        number={'font': {'size': 60, 'color': get_score_color(score), 'family': 'Poppins'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': get_score_color(score), 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 55], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [55, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [70, 85], 'color': 'rgba(59, 130, 246, 0.1)'},
                {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': get_score_color(score), 'width': 6},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Poppins"}
    )
    return fig

def create_percentile_chart(percentile, pro_distances, user_distance):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=pro_distances,
        name='Phân Bố PRO',
        marker=dict(color='rgba(102, 126, 234, 0.6)', line=dict(color='#667eea', width=2)),
        nbinsx=30
    ))
    
    fig.add_vline(
        x=user_distance,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Vị Trí Của Bạn (P{percentile:.0f})",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="So Sánh Với Phân Bố PRO",
        title_font=dict(size=20, family='Poppins', color='#1a1a1a'),
        xaxis_title="Khoảng Cách Từ Tâm PRO",
        yaxis_title="Số Lượng Golfer Chuyên Nghiệp",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.9)",
        font=dict(family='Poppins')
    )
    
    return fig

def create_top_metrics_chart(top_metrics):
    """Tạo biểu đồ các chỉ số quan trọng"""
    if not top_metrics:
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Giai Đoạn Setup", "Giai Đoạn Top", "Giai Đoạn Impact"),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    phases = ['setup', 'top', 'impact']
    colors = ['#3b82f6', '#f59e0b', '#10b981']
    
    for idx, phase in enumerate(phases):
        if phase in top_metrics:
            avg_angle = (top_metrics[phase]['left_arm_angle'] + top_metrics[phase]['right_arm_angle']) / 2
            normalized_score = min(100, max(0, (180 - abs(avg_angle - 90)) / 180 * 100))
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=normalized_score,
                title={'text': f"Vị Trí Tay", 'font': {'size': 14}},
                delta={'reference': 85},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': colors[idx]},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [50, 85], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                },
                number={'font': {'size': 20}}
            ), row=1, col=idx+1)
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Poppins'}
    )
    
    return fig

# =====================================================
# COMPARISON FUNCTIONS
# =====================================================
def create_comparison_chart(data1, data2, labels):
    """Tạo biểu đồ so sánh 2 video"""
    fig = go.Figure()
    
    metrics_names = ['Điểm Tổng Hợp', 'Tính Nhất Quán', 'Vận Tốc Tối Đa']
    
    values1 = [
        data1['score'],
        data1['top_metrics'].get('consistency', 50),
        data1['top_metrics']['velocity']['max']
    ]
    
    values2 = [
        data2['score'],
        data2['top_metrics'].get('consistency', 50),
        data2['top_metrics']['velocity']['max']
    ]
    
    fig.add_trace(go.Bar(
        name=labels[0],
        x=metrics_names,
        y=values1,
        marker=dict(color='#667eea')
    ))
    
    fig.add_trace(go.Bar(
        name=labels[1],
        x=metrics_names,
        y=values2,
        marker=dict(color='#f59e0b')
    ))
    
    fig.update_layout(
        title="So Sánh Các Chỉ Số Chính",
        barmode='group',
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.9)",
        font=dict(family='Poppins')
    )
    
    return fig

# =====================================================
# EXPORT FUNCTIONS - FIXED JSON SERIALIZATION
# =====================================================
def create_json_export(analysis_data):
    """Xuất kết quả dạng JSON - FIXED numpy serialization"""
    # Deep copy and convert numpy types to Python native types
    export_ready = {}
    
    for key, value in analysis_data.items():
        if isinstance(value, np.ndarray):
            export_ready[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            export_ready[key] = int(value)
        elif isinstance(value, (np.float64, np.float32, np.float16)):
            export_ready[key] = float(value)
        elif isinstance(value, dict):
            # Recursively handle nested dicts
            export_ready[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    export_ready[key][k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                    export_ready[key][k] = float(v)
                elif isinstance(v, dict):
                    export_ready[key][k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, (np.int64, np.float64, np.ndarray)):
                            export_ready[key][k][k2] = float(v2) if not isinstance(v2, np.ndarray) else v2.tolist()
                        else:
                            export_ready[key][k][k2] = v2
                else:
                    export_ready[key][k] = v
        else:
            export_ready[key] = value
    
    export_data = {
        'thoi_gian': datetime.now().isoformat(),
        'ket_qua_phan_tich': export_ready,
        'phien_ban_model': '8.0',
        'loai_phan_tich': 'percentile_based'
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def create_report_text(analysis_data):
    """Tạo báo cáo văn bản"""
    report = f"""
{'='*60}
BÁO CÁO PHÂN TÍCH GOLF SWING
{'='*60}

Ngày phân tích: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Video: {analysis_data.get('video_name', 'N/A')}

{'='*60}
KẾT QUẢ TỔNG QUAN
{'='*60}

Điểm AI:            {analysis_data.get('score', 0):.1f}/100
Cấp Độ:             {analysis_data.get('level', 'N/A')}
Percentile:         P{analysis_data.get('percentile', 0):.0f}
Góc Quay:           {analysis_data.get('view', 'N/A')}
Độ Tin Cậy:         {analysis_data.get('confidence', 0)}%

{'='*60}
PHÂN TÍCH CHI TIẾT
{'='*60}

Khoảng Cách:        {analysis_data.get('distance', 0):.4f}
PRO P50 (Trung Vị): {analysis_data.get('pro_p50', 0):.4f}
PRO P75:            {analysis_data.get('pro_p75', 0):.4f}
PRO P90:            {analysis_data.get('pro_p90', 0):.4f}

So sánh: Bạn tốt hơn {100-analysis_data.get('percentile', 0):.0f}% golfer chuyên nghiệp

{'='*60}
KHUYẾN NGHỊ
{'='*60}

{analysis_data.get('recommendations', 'N/A')}

{'='*60}
Báo cáo được tạo bởi Golf Swing Pro AI
VTK Team - Data Storm 2025
{'='*60}
"""
    return report

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models_and_reference():
    try:
        models = {
            'scaler_back': joblib.load('scaler_back_v2.pkl'),
            'scaler_side': joblib.load('scaler_side_v2.pkl'),
            'model_back': joblib.load('model_back_v2.pkl'),
            'model_side': joblib.load('model_side_v2.pkl')
        }
        
        reference = {}
        for view in ['back', 'side']:
            df = pd.read_csv(f'features_{view}_view.csv')
            feature_cols = [c for c in df.columns if c.startswith('feat_')]
            X = df[feature_cols].values
            
            scaler = models[f'scaler_{view}']
            X_scaled = scaler.transform(X)
            
            centroid = models[f'model_{view}']['centroid']
            pro_distances = np.linalg.norm(X_scaled - centroid, axis=1)
            
            reference[view] = {
                'distances': pro_distances,
                'min': float(np.min(pro_distances)),
                'p25': float(np.percentile(pro_distances, 25)),
                'p50': float(np.percentile(pro_distances, 50)),
                'p75': float(np.percentile(pro_distances, 75)),
                'p90': float(np.percentile(pro_distances, 90)),
                'max': float(np.max(pro_distances))
            }
        
        return models, reference
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file: {e}")
        return None, None

# =====================================================
# ANALYZE VIDEO FUNCTION
# =====================================================
def analyze_video(video_file, models, reference, manual_view_choice="🤖 Tự Động Nhận Diện"):
    """Hàm phân tích video tổng quát"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name
    
    try:
        pose_data, fps, detected_view, debug_info = extract_pose_landmarks(video_path)
        
        if pose_data is None:
            os.remove(video_path)
            return None
        
        if manual_view_choice == "🔙 Back View":
            detected_view = 'back'
            view_source = "Thủ Công"
        elif manual_view_choice == "👉 Side View":
            detected_view = 'side'
            view_source = "Thủ Công"
        else:
            view_source = "AI"
        
        features = extract_216_features(pose_data, fps)
        top_metrics = calculate_top_metrics(pose_data, fps)
        
        if features is None:
            os.remove(video_path)
            return None
        
        # PERCENTILE-BASED SCORING
        scaler = models[f'scaler_{detected_view}']
        model_data = models[f'model_{detected_view}']
        ref_data = reference[detected_view]
        
        feat_scaled = scaler.transform([features])[0]
        centroid = model_data['centroid']
        distance = np.linalg.norm(feat_scaled - centroid)
        
        pro_distances = ref_data['distances']
        percentile = (np.sum(pro_distances < distance) / len(pro_distances)) * 100
        
        # SCORING LOGIC
        if percentile <= 50:
            ml_score = 100 - (percentile * 0.3)
            level = "PRO 🏆"
            category = "PRO"
            confidence = 95
            recommendations = """
✅ Tập luyện 4-5 lần/tuần để duy trì phong độ
✅ Tinh chỉnh các chi tiết nhỏ
✅ Cân nhắc thi đấu chuyên nghiệp
✅ Làm việc với chuyên gia tâm lý thể thao
            """
        elif percentile <= 75:
            ml_score = 85 - ((percentile - 50) * 0.6)
            level = "Nâng Cao ⭐"
            category = "Nâng Cao"
            confidence = 85
            recommendations = """
🔄 Tăng tính nhất quán (tập 5-7 ngày/tuần)
🔄 Làm việc với huấn luyện viên chuyên nghiệp
🔄 Phân tích video thường xuyên
🔄 Tập trung vào điểm yếu
            """
        elif percentile <= 90:
            ml_score = 70 - ((percentile - 75) * 1.0)
            level = "Trung Cấp 📊"
            category = "Trung Cấp"
            confidence = 75
            recommendations = """
📚 Tập luyện 3-5 lần/tuần có mục tiêu
📚 Bài tập cơ bản: setup, alignment, tempo
📚 Xem lại video mỗi buổi tập
📚 Rèn luyện sức mạnh cơ core
            """
        elif percentile <= 100:
            ml_score = 55 - ((percentile - 90) * 1.5)
            level = "Mới Bắt Đầu 📈"
            category = "Mới Bắt Đầu"
            confidence = 70
            recommendations = """
💪 Tập trung nền tảng: cách cầm gậy, tư thế, canh hàng
💪 Tập tối thiểu 3 lần/tuần
💪 Luyện tập trước gương hàng ngày
💪 Tham khảo huấn luyện viên
            """
        else:
            excess = percentile - 100
            ml_score = max(40 - (excess * 0.5), 10)
            level = "Nghiệp Dư 💪"
            category = "Nghiệp Dư"
            confidence = 80
            recommendations = """
🎯 Học nền tảng từ đầu
🎯 Tập với huấn luyện viên chuyên nghiệp
🎯 Phân tích video thường xuyên
🎯 Rèn luyện thể lực: core, flexibility
            """
        
        ml_score = np.clip(ml_score, 10, 100)
        
        analysis_data = {
            'video_name': video_file.name,
            'score': float(ml_score),
            'level': level,
            'category': category,
            'percentile': float(percentile),
            'distance': float(distance),
            'view': detected_view,
            'view_source': view_source,
            'confidence': confidence,
            'pro_p50': ref_data['p50'],
            'pro_p75': ref_data['p75'],
            'pro_p90': ref_data['p90'],
            'pro_max': ref_data['max'],
            'fps': float(fps),
            'frames': len(pose_data),
            'features_count': len(features),
            'top_metrics': top_metrics,
            'recommendations': recommendations,
            'pro_distances': pro_distances,
            'features': features
        }
        
        os.remove(video_path)
        return analysis_data
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise e

# =====================================================
# MAIN APP
# =====================================================
st.markdown("""
<div style='text-align: center; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;'>
    <h1 style='font-size: 48px; margin-bottom: 10px;'>⛳ Phân Tích Golf Swing Pro</h1>
    <p style='font-size: 18px; color: #64748b;'>Hệ Thống Phân Tích Sinh Cơ Học Golf Bằng AI | Data Storm 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #1e293b;'>📋 Hướng Dẫn Sử Dụng</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: #334155;'>
    <p><strong>Bước 1:</strong> Tải lên video golf swing của bạn</p>
    
    <p><strong>Bước 2:</strong> Chọn góc quay (Tự động/Sau/Bên)</p>
    
    <p><strong>Bước 3:</strong> Nhấn "Phân Tích" và chờ kết quả</p>
    
    <p><strong>Bước 4:</strong> Xem chi tiết & tải báo cáo</p>
    
    <p><strong>Bước 5:</strong> Dùng tab "So Sánh Video" để so sánh với video mẫu</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2 style='color: #1e293b;'>🏆 Thang Điểm</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='score-badge badge-excellent'>85-100: Cấp Độ PRO 🏆</div><br/>
    <div class='score-badge badge-good'>70-85: Nâng Cao ⭐</div><br/>
    <div class='score-badge badge-average'>50-70: Trung Cấp 📊</div><br/>
    <div class='score-badge badge-poor'>0-50: Mới Bắt Đầu 💪</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2 style='color: #1e293b;'>🔬 Thông Tin Hệ Thống</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #475569; font-size: 14px;'>Loại: Chấm Điểm Dựa Trên Percentile</p>
    <p style='color: #475569; font-size: 14px;'>Phiên Bản: 8.0 Nâng Cấp</p>
    <p style='color: #475569; font-size: 14px;'>Model AI Thích Ứng Đa Miền</p>
    <p style='color: #475569; font-size: 14px;'>Có Tính Năng Xuất File & So Sánh</p>
    """, unsafe_allow_html=True)

# Load models
models, reference = load_models_and_reference()
if models is None or reference is None:
    st.stop()

# MAIN TABS
main_tab1, main_tab2 = st.tabs(["📊 Phân Tích Đơn", "🔄 So Sánh Video"])

# =====================================================
# TAB 1: SINGLE VIDEO ANALYSIS
# =====================================================
with main_tab1:
    uploaded_file = st.file_uploader(
        "📹 Tải Lên Video Golf Swing",
        type=['mp4', 'mov', 'avi'],
        help="Chọn video golf swing (tối đa 100MB)",
        key="single_upload"
    )
    
    if uploaded_file:
        col_video, col_info = st.columns([2, 1])
        
        with col_video:
            st.video(uploaded_file)
        
        with col_info:
            st.markdown("<h3 style='color: #1e293b;'>📊 Thông Tin Video</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #334155;'><strong>Tên:</strong> {uploaded_file.name}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #334155;'><strong>Kích Thước:</strong> {uploaded_file.size / 1024 / 1024:.2f} MB</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<h3 style='color: #1e293b;'>📐 Góc Quay</h3>", unsafe_allow_html=True)
            
            manual_view = st.radio(
                "Chọn góc:",
                options=["🤖 Auto Detect", "🔙 Back View", "👉 Side View"],
                index=0,
                key="view_selector"
            )
            
            st.markdown("---")
            analyze_btn = st.button("🚀 Phân Tích Video", type="primary", use_container_width=True, key="analyze_single")
        
        if analyze_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Đang trích xuất các điểm đánh dấu tư thế...")
                progress_bar.progress(20)
                
                # Reset file pointer
                uploaded_file.seek(0)
                analysis_data = analyze_video(uploaded_file, models, reference, manual_view)
                
                if analysis_data is None:
                    st.error("❌ Không thể phát hiện pose! Vui lòng thử video khác.")
                    st.stop()
                
                progress_bar.progress(100)
                status_text.text("✅ Hoàn thành phân tích!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # DISPLAY RESULTS với TOOLTIP
                st.markdown("---")
                st.markdown("<h2 style='color: #1e293b;'>🎯 Kết Quả Phân Tích</h2>", unsafe_allow_html=True)
                
                # Main Metrics Row với Tooltip
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='impact-card'>
                        <div class='impact-value'>{analysis_data['score']:.1f}</div>
                        <div class='impact-label'>{create_tooltip("Điểm AI", METRIC_DEFINITIONS["Điểm AI"])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    view_emoji = "🔙" if analysis_data['view'] == 'back' else "👉"
                    view_text = "SAU" if analysis_data['view'] == 'back' else "BÊN"
                    st.markdown(f"""
                    <div class='impact-card'>
                        <div class='impact-value'>{view_emoji}</div>
                        <div class='impact-label'>{create_tooltip(f"Góc {view_text}", METRIC_DEFINITIONS["Góc Quay"])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='impact-card'>
                        <div class='impact-value'>P{analysis_data['percentile']:.0f}</div>
                        <div class='impact-label'>{create_tooltip("Percentile", METRIC_DEFINITIONS["Percentile"])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='impact-card'>
                        <div class='impact-value'>{analysis_data['confidence']}%</div>
                        <div class='impact-label'>{create_tooltip("Độ Tin Cậy", METRIC_DEFINITIONS["Độ Tin Cậy"])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # TOP IMPACT METRICS với Tooltip
                top_metrics = analysis_data['top_metrics']
                if top_metrics:
                    st.markdown("---")
                    st.markdown("<h2 style='color: #1e293b;'>📊 Các Chỉ Số Quan Trọng</h2>", unsafe_allow_html=True)
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        consistency = top_metrics.get('consistency', 50)
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #667eea; margin: 0;'>{consistency:.1f}%</h3>
                            <p style='margin: 5px 0 0 0; color: #64748b;'>{create_tooltip("Tính Nhất Quán", METRIC_DEFINITIONS["Tính Nhất Quán"])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m2:
                        max_vel = top_metrics['velocity']['max']
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #f59e0b; margin: 0;'>{max_vel:.2f}</h3>
                            <p style='margin: 5px 0 0 0; color: #64748b;'>{create_tooltip("Vận Tốc Tối Đa", METRIC_DEFINITIONS["Vận Tốc Tối Đa"])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m3:
                        if 'impact' in top_metrics:
                            knee_flex = (top_metrics['impact']['left_knee_flex'] + top_metrics['impact']['right_knee_flex']) / 2
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3 style='color: #10b981; margin: 0;'>{knee_flex:.1f}°</h3>
                                <p style='margin: 5px 0 0 0; color: #64748b;'>{create_tooltip("Góc Gấp Đầu Gối", METRIC_DEFINITIONS["Góc Gấp Đầu Gối"])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_m4:
                        if 'top' in top_metrics:
                            arm_angle = (top_metrics['top']['left_arm_angle'] + top_metrics['top']['right_arm_angle']) / 2
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3 style='color: #3b82f6; margin: 0;'>{arm_angle:.1f}°</h3>
                                <p style='margin: 5px 0 0 0; color: #64748b;'>{create_tooltip("Góc Cánh Tay", METRIC_DEFINITIONS["Góc Cánh Tay"])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Phase Analysis Chart
                    st.plotly_chart(create_top_metrics_chart(top_metrics), use_container_width=True)
                
                # Gauge Chart
                st.markdown("---")
                st.plotly_chart(create_gauge_chart(analysis_data['score'], "Điểm Tổng Hợp"), use_container_width=True)
                
                # DOWNLOAD SECTION
                st.markdown("---")
                st.markdown("<h2 style='color: #1e293b;'>💾 Tải Báo Cáo</h2>", unsafe_allow_html=True)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    json_data = create_json_export(analysis_data)
                    st.download_button(
                        label="📥 Tải Dữ Liệu JSON",
                        data=json_data,
                        file_name=f"golf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col_dl2:
                    report_text = create_report_text(analysis_data)
                    st.download_button(
                        label="📄 Tải Báo Cáo Văn Bản",
                        data=report_text,
                        file_name=f"golf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Detail Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng Quan", "📈 So Sánh Chi Tiết", "💡 Khuyến Nghị", "📉 Biểu Đồ"])
                
                with tab1:
                    st.markdown(f"<h3 style='color: #1e293b;'>💡 Đánh Giá: <strong>{analysis_data['category'].upper()}</strong></h3>", unsafe_allow_html=True)
                    
                    if analysis_data['category'] == "PRO":
                        st.success(f"""
                        🏆 **CHÚC MỪNG! BẠN ĐẠT CẤP ĐỘ CHUYÊN NGHIỆP!**
                        
                        **Điểm:** {analysis_data['score']:.1f}/100
                        
                        📊 **Phân Tích Percentile:**
                        - Bạn đang ở **P{analysis_data['percentile']:.0f}** - nghĩa là tốt hơn **{100-analysis_data['percentile']:.0f}%** golfer chuyên nghiệp
                        - Khoảng cách: {analysis_data['distance']:.2f} (rất gần tâm PRO)
                        - Độ tin cậy: {analysis_data['confidence']}%
                        
                        ✅ **Điểm Mạnh:**
                        - Sinh cơ học chuẩn PRO
                        - Tính nhất quán xuất sắc
                        - Kỹ thuật đỉnh cao
                        - Timing & rhythm hoàn hảo
                        """)
                    
                    elif analysis_data['category'] == "Nâng Cao":
                        st.info(f"""
                        ⭐ **XUẤT SẮC! BẠN Ở CẤP ĐỘ NÂNG CAO**
                        
                        **Điểm:** {analysis_data['score']:.1f}/100
                        
                        📊 **Phân Tích Percentile:**
                        - Percentile: **P{analysis_data['percentile']:.0f}**
                        - Tốt hơn **{100-analysis_data['percentile']:.0f}%** golfer PRO
                        - Chỉ còn **{analysis_data['percentile']-50:.0f}%** nữa là đạt cấp PRO
                        
                        ✅ **Điểm Mạnh:**
                        - Nền tảng vững chắc
                        - Kỹ thuật tốt, ổn định
                        - Gần đạt tiêu chuẩn PRO
                        """)
                    
                    else:
                        st.warning(f"""
                        📊 **BẠN Ở CẤP ĐỘ {analysis_data['category'].upper()}**
                        
                        **Điểm:** {analysis_data['score']:.1f}/100
                        
                        ⚠️  **Phân Tích:**
                        - Percentile: **P{analysis_data['percentile']:.0f}**
                        - Khoảng cách: {analysis_data['distance']:.2f}
                        - Cấp độ: {analysis_data['category']}
                        
                        🔧 **Cần Cải Thiện:**
                        - Sinh cơ học cơ bản
                        - Tính nhất quán
                        - Timing & tư thế
                        """)
                
                with tab2:
                    st.markdown("<h3 style='color: #1e293b;'>📈 So Sánh Chi Tiết Với PRO</h3>", unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("<p style='color: #334155;'><strong>📏 Các Chỉ Số Khoảng Cách:</strong></p>", unsafe_allow_html=True)
                        comparison_df = pd.DataFrame({
                            'Chỉ Số': ['Khoảng Cách Của Bạn', 'PRO P50 (Trung Vị)', 'PRO P75', 'PRO P90', 'PRO Tối Đa'],
                            'Giá Trị': [
                                f"{analysis_data['distance']:.4f}",
                                f"{analysis_data['pro_p50']:.4f}",
                                f"{analysis_data['pro_p75']:.4f}",
                                f"{analysis_data['pro_p90']:.4f}",
                                f"{analysis_data['pro_max']:.4f}"
                            ]
                        })
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    with col_b:
                        st.markdown("<p style='color: #334155;'><strong>🎯 Phân Tích Điểm:</strong></p>", unsafe_allow_html=True)
                        scoring_df = pd.DataFrame({
                            'Khía Cạnh': ['Xếp Hạng Percentile', 'Điểm', 'Cấp Độ', 'Độ Tin Cậy'],
                            'Giá Trị': [
                                f"P{analysis_data['percentile']:.1f}",
                                f"{analysis_data['score']:.1f}/100",
                                analysis_data['level'],
                                f"{analysis_data['confidence']}%"
                            ]
                        })
                        st.dataframe(scoring_df, use_container_width=True, hide_index=True)
                    
                    # Progress bars
                    st.markdown("---")
                    st.markdown("<h3 style='color: #1e293b;'>📊 Tiến Độ Trực Quan</h3>", unsafe_allow_html=True)
                    
                    col_prog1, col_prog2 = st.columns(2)
                    
                    with col_prog1:
                        st.markdown("<p style='color: #334155;'><strong>Tiến Độ Điểm:</strong></p>", unsafe_allow_html=True)
                        st.progress(analysis_data['score'] / 100)
                        st.markdown(f"<p style='color: #475569; font-size: 14px;'>{analysis_data['score']:.1f}/100 - {get_score_label(analysis_data['score'])}</p>", unsafe_allow_html=True)
                    
                    with col_prog2:
                        st.markdown("<p style='color: #334155;'><strong>Vị Trí Percentile:</strong></p>", unsafe_allow_html=True)
                        st.progress(min(analysis_data['percentile'] / 100, 1.0))
                        st.markdown(f"<p style='color: #475569; font-size: 14px;'>P{analysis_data['percentile']:.0f} - Top {max(0, 100-analysis_data['percentile']):.0f}% PRO</p>", unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("<h3 style='color: #1e293b;'>💡 Khuyến Nghị Cải Thiện</h3>", unsafe_allow_html=True)
                    st.info(analysis_data['recommendations'])
                    
                    st.markdown("---")
                    st.markdown("<h3 style='color: #1e293b;'>📚 Bài Tập Chi Tiết</h3>", unsafe_allow_html=True)
                    
                    col_ex1, col_ex2 = st.columns(2)
                    
                    with col_ex1:
                        st.markdown("""
                        <div style='color: #334155;'>
                        <p><strong>🏋️ Rèn Luyện Thể Lực:</strong></p>
                        <ul>
                        <li>Plank: 45 giây x 3 hiệp</li>
                        <li>Russian twist: 20 lần x 3 hiệp</li>
                        <li>Nâng tạ một chân: 10 lần/chân</li>
                        <li>Bài tập xoay hông</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_ex2:
                        st.markdown("""
                        <div style='color: #334155;'>
                        <p><strong>⛳ Bài Tập Golf Chuyên Biệt:</strong></p>
                        <ul>
                        <li>Luyện trước gương (5 phút/ngày)</li>
                        <li>Bài tập canh hàng với gậy</li>
                        <li>Bài tập nhịp độ (3:1 rhythm)</li>
                        <li>Luyện chuyển trọng tâm</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab4:
                    st.markdown("<h3 style='color: #1e293b;'>📉 Biểu Đồ Phân Tích</h3>", unsafe_allow_html=True)
                    
                    # Distribution Chart
                    st.plotly_chart(
                        create_percentile_chart(
                            analysis_data['percentile'], 
                            analysis_data['pro_distances'], 
                            analysis_data['distance']
                        ),
                        use_container_width=True
                    )
                    
                    # Comparison Bar Chart
                    fig_bar = go.Figure()
                    
                    metrics_names = ['Điểm Của Bạn', 'PRO Trung Vị', 'PRO P75', 'PRO P90']
                    your_score_vis = analysis_data['score']
                    pro_median_vis = 100 - (50 * 0.3)
                    pro_p75_vis = 85 - (25 * 0.6)
                    pro_p90_vis = 70
                    
                    metrics_values = [your_score_vis, pro_median_vis, pro_p75_vis, pro_p90_vis]
                    colors_bar = [get_score_color(analysis_data['score']), '#fbbf24', '#60a5fa', '#34d399']
                    
                    fig_bar.add_trace(go.Bar(
                        x=metrics_names,
                        y=metrics_values,
                        marker=dict(color=colors_bar),
                        text=[f"{v:.1f}" for v in metrics_values],
                        textposition='outside'
                    ))
                    
                    fig_bar.update_layout(
                        title="So Sánh Điểm Với Ngưỡng PRO",
                        yaxis_title="Điểm",
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.9)",
                        font=dict(family='Poppins')
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Technical Details
                with st.expander("🔬 Chi Tiết Kỹ Thuật"):
                    col_tech1, col_tech2, col_tech3 = st.columns(3)
                    
                    with col_tech1:
                        st.markdown("<p style='color: #334155;'><strong>📏 Chỉ Số Model:</strong></p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• {create_tooltip('Khoảng Cách', METRIC_DEFINITIONS['Distance'])}: {analysis_data['distance']:.4f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• Percentile: P{analysis_data['percentile']:.1f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• Điểm: {analysis_data['score']:.1f}/100</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• Độ tin cậy: {analysis_data['confidence']}%</p>", unsafe_allow_html=True)
                    
                    with col_tech2:
                        st.markdown("<p style='color: #334155;'><strong>📹 Thông Tin Video:</strong></p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• Góc: {analysis_data['view']} ({analysis_data['view_source']})</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• Đặc trưng: {analysis_data['features_count']}D</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• Khung hình: {analysis_data['frames']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #475569;'>• {create_tooltip('FPS', METRIC_DEFINITIONS['FPS'])}: {analysis_data['fps']:.1f}</p>", unsafe_allow_html=True)
                    
                    with col_tech3:
                        st.markdown("<p style='color: #334155;'><strong>🎯 Hiệu Suất:</strong></p>", unsafe_allow_html=True)
                        if top_metrics:
                            st.markdown(f"<p style='color: #475569;'>• Tính nhất quán: {top_metrics.get('consistency', 0):.1f}%</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #475569;'>• Vận tốc tối đa: {top_metrics['velocity']['max']:.2f}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #475569;'>• Vận tốc trung bình: {top_metrics['velocity']['avg']:.2f}</p>", unsafe_allow_html=True)
                
                # Save to session state for comparison
                st.session_state['last_analysis'] = analysis_data
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# =====================================================
# TAB 2: VIDEO COMPARISON
# =====================================================
with main_tab2:
    st.markdown("<h2 style='color: #1e293b;'>🔄 So Sánh Hai Video</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #334155;'>So sánh video swing của bạn với video mẫu hoặc video trước đó của bạn</p>", unsafe_allow_html=True)
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("### 📹 Video Thứ Nhất")
        video1 = st.file_uploader(
            "Tải video thứ nhất",
            type=['mp4', 'mov', 'avi'],
            help="Video người dùng hoặc video mẫu",
            key="video1_upload"
        )
        
        if video1:
            st.video(video1)
            view1 = st.radio(
                "Góc quay video 1:",
                options=["🤖 Auto Detect", "🔙 Back View", "👉 Side View"],
                index=0,
                key="view1"
            )
    
    with col_comp2:
        st.markdown("### 📹 Video Thứ Hai")
        video2 = st.file_uploader(
            "Tải video thứ hai",
            type=['mp4', 'mov', 'avi'],
            help="Video để so sánh",
            key="video2_upload"
        )
        
        if video2:
            st.video(video2)
            view2 = st.radio(
                "Góc quay video 2:",
                options=["🤖 Auto Detect", "🔙 Back View", "👉 Side View"],
                index=0,
                key="view2"
            )
    
    if video1 and video2:
        compare_btn = st.button("🔬 So Sánh Hai Video", type="primary", use_container_width=True, key="compare_btn")
        
        if compare_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Analyze video 1
                status_text.text("🔍 Đang phân tích video thứ nhất...")
                progress_bar.progress(25)
                video1.seek(0)
                analysis1 = analyze_video(video1, models, reference, view1)
                
                if analysis1 is None:
                    st.error("❌ Không thể phân tích video thứ nhất!")
                    st.stop()
                
                # Analyze video 2
                status_text.text("🔍 Đang phân tích video thứ hai...")
                progress_bar.progress(75)
                video2.seek(0)
                analysis2 = analyze_video(video2, models, reference, view2)
                
                if analysis2 is None:
                    st.error("❌ Không thể phân tích video thứ hai!")
                    st.stop()
                
                progress_bar.progress(100)
                status_text.text("✅ Hoàn thành so sánh!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # DISPLAY COMPARISON
                st.markdown("---")
                st.markdown("<h2 style='color: #1e293b;'>📊 Kết Quả So Sánh</h2>", unsafe_allow_html=True)
                
                # Comparison Summary
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.markdown(f"""
                    <div class='comparison-card'>
                        <h3 style='color: #667eea;'>📹 Video 1: {video1.name}</h3>
                        <p style='color: #334155;'><strong>Điểm:</strong> {analysis1['score']:.1f}/100</p>
                        <p style='color: #334155;'><strong>Cấp độ:</strong> {analysis1['level']}</p>
                        <p style='color: #334155;'><strong>Percentile:</strong> P{analysis1['percentile']:.0f}</p>
                        <p style='color: #334155;'><strong>Tính nhất quán:</strong> {analysis1['top_metrics'].get('consistency', 0):.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_sum2:
                    st.markdown(f"""
                    <div class='comparison-card'>
                        <h3 style='color: #f59e0b;'>📹 Video 2: {video2.name}</h3>
                        <p style='color: #334155;'><strong>Điểm:</strong> {analysis2['score']:.1f}/100</p>
                        <p style='color: #334155;'><strong>Cấp độ:</strong> {analysis2['level']}</p>
                        <p style='color: #334155;'><strong>Percentile:</strong> P{analysis2['percentile']:.0f}</p>
                        <p style='color: #334155;'><strong>Tính nhất quán:</strong> {analysis2['top_metrics'].get('consistency', 0):.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparison Chart
                st.plotly_chart(
                    create_comparison_chart(
                        analysis1, 
                        analysis2, 
                        [f"Video 1: {video1.name[:20]}", f"Video 2: {video2.name[:20]}"]
                    ),
                    use_container_width=True
                )
                
                # Detailed Comparison Table
                st.markdown("### 📋 Bảng So Sánh Chi Tiết")
                
                comparison_table = pd.DataFrame({
                    'Chỉ Số': [
                        'Điểm Tổng Hợp',
                        'Percentile',
                        'Tính Nhất Quán (%)',
                        'Vận Tốc Tối Đa',
                        'Vận Tốc TB',
                        'Độ Tin Cậy (%)',
                        'Góc Quay'
                    ],
                    f'{video1.name[:30]}': [
                        f"{analysis1['score']:.1f}",
                        f"P{analysis1['percentile']:.0f}",
                        f"{analysis1['top_metrics'].get('consistency', 0):.1f}",
                        f"{analysis1['top_metrics']['velocity']['max']:.2f}",
                        f"{analysis1['top_metrics']['velocity']['avg']:.2f}",
                        f"{analysis1['confidence']}",
                        analysis1['view']
                    ],
                    f'{video2.name[:30]}': [
                        f"{analysis2['score']:.1f}",
                        f"P{analysis2['percentile']:.0f}",
                        f"{analysis2['top_metrics'].get('consistency', 0):.1f}",
                        f"{analysis2['top_metrics']['velocity']['max']:.2f}",
                        f"{analysis2['top_metrics']['velocity']['avg']:.2f}",
                        f"{analysis2['confidence']}",
                        analysis2['view']
                    ],
                    'Chênh Lệch': [
                        f"{analysis1['score'] - analysis2['score']:+.1f}",
                        f"{analysis1['percentile'] - analysis2['percentile']:+.0f}",
                        f"{analysis1['top_metrics'].get('consistency', 0) - analysis2['top_metrics'].get('consistency', 0):+.1f}",
                        f"{analysis1['top_metrics']['velocity']['max'] - analysis2['top_metrics']['velocity']['max']:+.2f}",
                        f"{analysis1['top_metrics']['velocity']['avg'] - analysis2['top_metrics']['velocity']['avg']:+.2f}",
                        f"{analysis1['confidence'] - analysis2['confidence']:+d}",
                        "-"
                    ]
                })
                
                st.dataframe(comparison_table, use_container_width=True, hide_index=True)
                
                # Insights
                st.markdown("### 💡 Phân Tích & Khuyến Nghị")
                
                if analysis1['score'] > analysis2['score']:
                    better_video = "Video 1"
                    diff = analysis1['score'] - analysis2['score']
                    st.success(f"✅ **{better_video}** tốt hơn với chênh lệch **{diff:.1f} điểm**")
                elif analysis2['score'] > analysis1['score']:
                    better_video = "Video 2"
                    diff = analysis2['score'] - analysis1['score']
                    st.success(f"✅ **{better_video}** tốt hơn với chênh lệch **{diff:.1f} điểm**")
                else:
                    st.info("📊 Hai video có điểm tương đương nhau")
                
                # Key differences
                st.markdown("#### 🔍 Điểm Khác Biệt Chính:")
                
                consistency_diff = analysis1['top_metrics'].get('consistency', 0) - analysis2['top_metrics'].get('consistency', 0)
                velocity_diff = analysis1['top_metrics']['velocity']['max'] - analysis2['top_metrics']['velocity']['max']
                
                if abs(consistency_diff) > 5:
                    if consistency_diff > 0:
                        st.info(f"📊 Video 1 có tính nhất quán cao hơn {abs(consistency_diff):.1f}%")
                    else:
                        st.info(f"📊 Video 2 có tính nhất quán cao hơn {abs(consistency_diff):.1f}%")
                
                if abs(velocity_diff) > 0.1:
                    if velocity_diff > 0:
                        st.info(f"⚡ Video 1 có vận tốc cao hơn {abs(velocity_diff):.2f}")
                    else:
                        st.info(f"⚡ Video 2 có vận tốc cao hơn {abs(velocity_diff):.2f}")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi so sánh: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
    <h3 style='color: #1e293b; margin-bottom: 10px;'>⛳ Phát Triển Bởi VTK Team</h3>
    <p style='color: #334155; font-size: 16px; margin: 5px 0;'><strong>Lâm Tuấn Vũ • Nguyễn Vũ Thắng • Đỗ Gia Khiêm</strong></p>
    <p style='color: #64748b; margin: 5px 0;'>Data Storm Competition 2025 | Hệ Thống Phân Tích Sinh Cơ Học Golf Bằng AI</p>
</div>
""", unsafe_allow_html=True)


