"""
AI-powered Golf Swing Recommendations using OpenAI GPT
Tự động phân tích và tạo khuyến nghị dựa trên metrics thực tế
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Load environment variables
load_dotenv()

def get_openai_client():
    """Initialize OpenAI client for AgentRouter (Claude)"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("⚠️ OPENAI_API_KEY not found in .env file")
    if not base_url:
        raise ValueError("⚠️ OPENAI_BASE_URL not found in .env file")

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )
def generate_ai_recommendations(analysis_data):
    """
    Tạo khuyến nghị chi tiết bằng AI dựa trên dữ liệu phân tích
    
    Args:
        analysis_data (dict): Chứa tất cả metrics từ phân tích video:
            - score: float (0-100)
            - level: str (PRO, Nâng Cao, Trung Cấp, Mới Bắt Đầu)
            - percentile: float
            - distance: float
            - top_metrics: dict với consistency, velocity, phases
            - view: str (back/side)
            
    Returns:
        str: Khuyến nghị chi tiết format Markdown
    """
    
    try:
        client = get_openai_client()
        
        # Build comprehensive prompt
        prompt = build_detailed_prompt(analysis_data)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="claude-sonnet-4-5-20250929",  # Hoặc "gpt-4o-mini" để rẻ hơn
            messages=[
                {
                    "role": "system",
                    "content": """Bạn là chuyên gia phân tích golf swing với 20 năm kinh nghiệm, 
                    chuyên về biomechanics và training. Nhiệm vụ của bạn là phân tích dữ liệu 
                    sinh cơ học chi tiết và đưa ra khuyến nghị luyện tập CỤ THỂ, CÁ NHÂN HÓA.
                    
                    Phong cách viết:
                    - Tiếng Việt chuyên nghiệp nhưng dễ hiểu
                    - Format Markdown sạch đẹp
                    - Số liệu cụ thể (số lần, số hiệp, thời gian)
                    - Giải thích TẠI SAO mỗi bài tập quan trọng
                    - Động viên nhưng thực tế"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.75,  # Vừa creative vừa consistent
            max_tokens=1800,
            top_p=0.9
        )
        
        recommendations = response.choices[0].message.content
        return recommendations
        
    except Exception as e:
        st.warning(f"⚠️ AI recommendations tạm thời không khả dụng: {str(e)}")
        # Fallback to basic recommendations
        return get_fallback_recommendations(analysis_data)


def build_detailed_prompt(data):
    """Xây dựng prompt chi tiết với tất cả metrics"""
    
    # Extract metrics
    score = data.get('score', 0)
    level = data.get('level', 'N/A')
    category = data.get('category', 'N/A')
    percentile = data.get('percentile', 0)
    distance = data.get('distance', 0)
    view = data.get('view', 'back')
    
    top_metrics = data.get('top_metrics', {})
    consistency = top_metrics.get('consistency', 0)
    velocity = top_metrics.get('velocity', {})
    vel_max = velocity.get('max', 0)
    vel_avg = velocity.get('avg', 0)
    
    # Build phase analysis
    phase_analysis = ""
    for phase_name in ['setup', 'top', 'impact']:
        if phase_name in top_metrics:
            phase = top_metrics[phase_name]
            left_arm = phase.get('left_arm_angle', 0)
            right_arm = phase.get('right_arm_angle', 0)
            left_knee = phase.get('left_knee_flex', 0)
            right_knee = phase.get('right_knee_flex', 0)
            
            phase_analysis += f"""
**Giai đoạn {phase_name.upper()}:**
- Góc tay trái: {left_arm:.1f}° | Góc tay phải: {right_arm:.1f}°
- Góc gối trái: {left_knee:.1f}° | Góc gối phải: {right_knee:.1f}°
"""
    
    # Identify specific weaknesses
    weaknesses = identify_technical_weaknesses(data)
    weaknesses_text = "\n".join(f"- {w}" for w in weaknesses)
    
    # Build comprehensive prompt
    prompt = f"""
Hãy phân tích dữ liệu biomechanics golf swing sau và tạo khuyến nghị chi tiết:

## 📊 THÔNG SỐ NGƯỜI CHƠI

**Đánh giá tổng quan:**
- Điểm tổng hợp: **{score:.1f}/100**
- Cấp độ: **{category}**
- Xếp hạng Percentile: **P{percentile:.0f}** (tốt hơn {100-percentile:.0f}% golfer PRO)
- Khoảng cách từ trung tâm PRO: **{distance:.4f}**
- Góc quay video: **{view.upper()}**

## 🔍 CHỈ SỐ SINH CƠ HỌC CHI TIẾT

**Metrics tổng quan:**
- Tính nhất quán (Consistency): **{consistency:.1f}/100**
- Vận tốc tối đa: **{vel_max:.2f} m/s**
- Vận tốc trung bình: **{vel_avg:.2f} m/s**

**Phân tích theo từng giai đoạn swing:**
{phase_analysis}

## ⚠️ CÁC ĐIỂM YẾU ĐÃ PHÁT HIỆN

{weaknesses_text}

---

## 📝 YÊU CẦU TẠO KHUYẾN NGHỊ

Hãy tạo khuyến nghị theo **ĐÚNG ĐỊNH DẠNG** sau (Markdown):

### 🎯 Đánh Giá Tổng Quan
(2-3 câu ngắn gọn: tình trạng hiện tại, điểm mạnh/yếu chính, tiềm năng cải thiện)

### 🚨 3 Lỗi Chính Cần Khắc Phục
1. **[Tên lỗi cụ thể]** - [Giải thích tại sao đây là vấn đề và ảnh hưởng gì đến swing]
2. **[Tên lỗi cụ thể]** - [Giải thích tại sao]
3. **[Tên lỗi cụ thể]** - [Giải thích tại sao]

### 💪 Bài Tập Khắc Phục Chi Tiết

**Tuần 1-2: Xây Dựng Nền Tảng**
- **[Tên bài tập 1]:** [Mô tả CHI TIẾT - số lần, số hiệp, thời gian, cách thực hiện, lưu ý]
- **[Tên bài tập 2]:** [Mô tả chi tiết]
- **[Tên bài tập 3]:** [Mô tả chi tiết]

**Tuần 3-4: Nâng Cao & Tối Ưu**
- **[Tên bài tập 4]:** [Mô tả chi tiết - liên kết với lỗi cụ thể nào]
- **[Tên bài tập 5]:** [Mô tả chi tiết]
- **[Tên bài tập 6]:** [Mô tả chi tiết]

### 📅 Lộ Trình Luyện Tập
- **Tần suất:** [X lần/tuần] - gợi ý ngày cụ thể (Thứ 2,4,6...)
- **Thời lượng mỗi buổi:** [X-Y phút] - phân bổ thời gian cho từng phần
- **Chu kỳ review:** [Mỗi X tuần quay video để kiểm tra tiến bộ]
- **Thời gian dự kiến cải thiện:** [X tuần/tháng để đạt điểm mục tiêu]

### 🎯 Mục Tiêu Cụ Thể (1 tháng)
- Tăng điểm từ {score:.1f} lên [X điểm] (+[delta])
- Tăng consistency từ {consistency:.1f} lên [X]
- Tăng velocity từ {vel_max:.2f} lên [X] m/s
- [Mục tiêu khác nếu cần]

### 💡 Lời Khuyên Bổ Sung
- **Tâm lý:** [Tips về mindset, focus, patience]
- **Kỹ thuật:** [Chi tiết về form, timing, rhythm]
- **Thể lực:** [Dinh dưỡng, nghỉ ngơi, recovery]

---

**LƯU Ý QUAN TRỌNG:**
- Khuyến nghị phải DỰA VÀO METRICS THỰC TẾ ở trên, không chung chung
- Bài tập phải CỤ THỂ với số lượng/thời gian/cách thực hiện rõ ràng
- Lộ trình phải THỰC TẾ với người chơi cấp {category}
- Giải thích rõ mỗi bài tập KHẮC PHỤC LỖI NÀO trong 3 lỗi chính
- Mục tiêu phải ACHIEVABLE trong 1 tháng, không quá tham vọng
- Viết bằng tiếng Việt, dùng icon emoji phù hợp, format Markdown đẹp
"""
    
    return prompt


def identify_technical_weaknesses(data):
    """Phát hiện điểm yếu cụ thể dựa trên metrics"""
    weaknesses = []
    
    score = data.get('score', 0)
    percentile = data.get('percentile', 0)
    top_metrics = data.get('top_metrics', {})
    
    consistency = top_metrics.get('consistency', 0)
    velocity = top_metrics.get('velocity', {})
    vel_max = velocity.get('max', 0)
    vel_avg = velocity.get('avg', 0)
    
    # Check consistency issues
    if consistency < 40:
        weaknesses.append(f"Tính nhất quán RẤT THẤP ({consistency:.1f}/100) - Swing path thay đổi liên tục, muscle memory chưa được xây dựng")
    elif consistency < 60:
        weaknesses.append(f"Tính nhất quán TRUNG BÌNH ({consistency:.1f}/100) - Cần cải thiện stability và repeatability")
    elif consistency < 75:
        weaknesses.append(f"Tính nhất quán KHÁ TỐT ({consistency:.1f}/100) - Chỉ cần fine-tune thêm chút nữa")
    
    # Check velocity issues
    if vel_max < 2.0:
        weaknesses.append(f"Vận tốc RẤT THẤP ({vel_max:.2f} m/s vs 3.5+ PRO) - Chưa tạo đủ lực xoay, timing chưa chuẩn, thiếu power transfer")
    elif vel_max < 2.8:
        weaknesses.append(f"Vận tốc CHƯA TỐI ƯU ({vel_max:.2f} m/s vs 3.5+ PRO) - Còn tiềm năng tăng thêm đáng kể")
    elif vel_max < 3.2:
        weaknesses.append(f"Vận tốc TỐT ({vel_max:.2f} m/s) - Gần đạt chuẩn PRO, cần optimize thêm")
    
    # Check velocity consistency
    if vel_max > 0 and vel_avg > 0:
        vel_ratio = vel_avg / vel_max
        if vel_ratio < 0.65:
            weaknesses.append(f"Chênh lệch vận tốc LỚN (avg/max = {vel_ratio:.2f}) - Tốc độ không đều, cần cải thiện rhythm")
    
    # Check impact phase
    if 'impact' in top_metrics:
        impact = top_metrics['impact']
        left_knee = impact.get('left_knee_flex', 0)
        right_knee = impact.get('right_knee_flex', 0)
        avg_knee = (left_knee + right_knee) / 2
        
        if avg_knee < 130:
            weaknesses.append(f"Góc gối tại IMPACT QUÁ THẤP ({avg_knee:.1f}° vs chuẩn 140-150°) - Mất stability, giảm power transfer")
        elif avg_knee > 160:
            weaknesses.append(f"Góc gối tại IMPACT QUÁ CAO ({avg_knee:.1f}° vs chuẩn 140-150°) - Không flexion đủ, mất lực từ chân")
        elif avg_knee < 138:
            weaknesses.append(f"Góc gối tại IMPACT HƠI THẤP ({avg_knee:.1f}°) - Cần tăng nhẹ để tối ưu power transfer")
    
    # Check top phase (backswing)
    if 'top' in top_metrics:
        top = top_metrics['top']
        left_arm = top.get('left_arm_angle', 0)
        right_arm = top.get('right_arm_angle', 0)
        avg_arm = (left_arm + right_arm) / 2
        
        # Ideal arm angle at top: 100-120°
        if avg_arm < 85 or avg_arm > 135:
            weaknesses.append(f"Góc cánh tay tại BACKSWING CHƯA CHUẨN (trái: {left_arm:.1f}°, phải: {right_arm:.1f}° vs chuẩn 100-120°) - Ảnh hưởng độ dài backswing và power")
    
    # Check setup phase
    if 'setup' in top_metrics:
        setup = top_metrics['setup']
        setup_knee_l = setup.get('left_knee_flex', 0)
        setup_knee_r = setup.get('right_knee_flex', 0)
        
        if setup_knee_l < 130 or setup_knee_r < 130:
            weaknesses.append(f"Setup: Góc gối quá thấp ({setup_knee_l:.1f}°, {setup_knee_r:.1f}°) - Tư thế ban đầu chưa tối ưu")
    
    # Overall level-based assessment
    if score < 50:
        weaknesses.append("Nền tảng kỹ thuật YẾU - Cần focus vào basics: grip, stance, alignment, posture")
    elif score < 65:
        weaknesses.append("Kỹ thuật CƠ BẢN - Cần xây dựng consistency và muscle memory vững chắc hơn")
    elif score < 80:
        weaknesses.append("Kỹ thuật KHÁ TỐT - Cần refine các chi tiết nhỏ để lên level cao hơn")
    
    # Percentile-based
    if percentile > 85:
        weaknesses.append(f"Khoảng cách từ PRO LỚN (P{percentile:.0f}) - Cần lộ trình dài hạn và kiên trì để cải thiện")
    
    # Ensure at least 3 weaknesses for better recommendations
    if len(weaknesses) < 3:
        weaknesses.append("Cần phân tích video với góc quay tốt hơn để phát hiện thêm chi tiết")
    
    return weaknesses[:6]  # Max 6 weaknesses to keep prompt manageable


def get_fallback_recommendations(data):
    """Fallback recommendations nếu API fail"""
    level = data.get('category', 'N/A')
    score = data.get('score', 0)
    consistency = data.get('top_metrics', {}).get('consistency', 0)
    vel_max = data.get('top_metrics', {}).get('velocity', {}).get('max', 0)
    
    return f"""
### 🎯 Khuyến Nghị Cơ Bản (Cấp {level})

**⚠️ Lưu ý:** AI recommendations tạm thời không khả dụng. Đây là gợi ý cơ bản dựa trên điểm số.

#### Đánh Giá Nhanh
- Điểm hiện tại: **{score:.1f}/100**
- Tính nhất quán: **{consistency:.1f}/100**
- Vận tốc tối đa: **{vel_max:.2f} m/s**

#### 💪 Bài Tập Cơ Bản

**Rèn luyện thể lực:**
- **Plank Core Stability:** 45 giây x 3 hiệp, nghỉ 30s giữa các hiệp
- **Russian Twist:** 20 lần x 3 hiệp (tăng lực xoay)
- **Squat:** 15 lần x 3 hiệp (tăng sức mạnh chân)
- **Hip Rotation Drill:** 10 lần mỗi bên x 3 hiệp

**Golf chuyên biệt:**
- **Swing chậm trước gương:** 10 phút/ngày - focus vào swing path nhất quán
- **Half swing drills:** 3 set x 15 reps - xây dựng muscle memory
- **Tempo training:** Dùng metronome 3:1 ratio (backswing:downswing)
- **Impact bag drill:** 20 lần/ngày - cải thiện impact position

#### 📅 Lộ Trình
- **Tần suất:** 3-4 lần/tuần (Thứ 2, 4, 6, CN)
- **Thời lượng:** 30-45 phút/buổi
- **Review:** Quay video mỗi 2 tuần để đánh giá tiến bộ
- **Mục tiêu:** Tăng điểm lên {min(score + 8, 100):.0f}+ trong 1 tháng

#### 💡 Tips
- Tập chậm chính xác hơn tập nhanh sai
- Focus vào 1-2 điểm mỗi buổi tập
- Record video thường xuyên để tự kiểm tra
- Làm việc với coach nếu có thể

💡 **Để có khuyến nghị chi tiết từ AI, vui lòng thử lại sau hoặc kiểm tra API key.**
"""
