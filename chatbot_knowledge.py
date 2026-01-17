"""
Knowledge Base cho VTK Golf Swing Analysis Chatbot
Chứa tất cả thông tin về chức năng, hướng dẫn, FAQ
"""

# Thông tin cơ bản về hệ thống
SYSTEM_INFO = """
🏌️ HỆ THỐNG PHÂN TÍCH GOLF SWING - VTK TEAM

**Tên đầy đủ:** Phân Tích Golf Swing Pro
**Phát triển bởi:** VTK Team Data Storm
**Mục đích:** Phân tích video golf swing bằng AI và MediaPipe, đưa ra đánh giá chuyên nghiệp và khuyến nghị cải thiện.

**Công nghệ sử dụng:**
- MediaPipe Pose Detection (Google)
- Machine Learning Models (Scikit-learn)
- OpenAI GPT-4o-mini (AI Recommendations)
- Streamlit (Web Framework)
- Python 3.11+
"""

# Các chức năng chính
FEATURES = {
    "upload_video": {
        "name": "📤 Tải Lên Video Golf Swing",
        "description": "Upload video swing của bạn (MP4, AVI, MOV)",
        "requirements": [
            "Video độ dài 5-60 giây",
            "Chất lượng tốt, ánh sáng đủ",
            "Góc quay: Back view (sau lưng) hoặc Side view (bên hông)",
            "Người chơi rõ ràng, toàn thân trong khung hình"
        ],
        "how_to": "Bước 1 ở sidebar bên trái → Chọn file video → Click 'Phân tích'"
    },
    
    "choose_club": {
        "name": "⛳ Chọn Gậy Golf",
        "description": "Chọn loại gậy để AI so sánh chính xác",
        "options": ["Tự động (AI detect)", "Sắu (Iron)", "Bên (Wood)"],
        "how_to": "Bước 2 ở sidebar → Chọn dropdown menu → Chọn gậy phù hợp"
    },
    
    "view_analysis": {
        "name": "🔬 Phân Tích Chi Tiết",
        "description": "Xem kết quả phân tích toàn diện",
        "metrics": [
            "Điểm AI tổng hợp (0-100)",
            "Cấp độ: PRO / Nâng Cao / Trung Cấp / Mới Bắt Đầu",
            "Percentile: So sánh với golfer PRO",
            "Độ tin cậy: Chất lượng phân tích",
            "Tính nhất quán (Consistency): Độ ổn định swing",
            "Vận tốc: Tốc độ swing tối đa và trung bình",
            "Góc khớp: Tay, gối, hông tại các giai đoạn"
        ],
        "tabs": [
            "📊 Tổng Quan: Điểm số và đánh giá tổng quan",
            "📈 So Sánh Chi Tiết: So sánh với PRO",
            "💡 Khuyến Nghị: Gợi ý từ AI",
            "📊 Biểu Đồ: Visualization dữ liệu"
        ]
    },
    
    "ai_recommendations": {
        "name": "🤖 Khuyến Nghị Cải Thiện AI",
        "description": "AI phân tích và đưa ra bài tập cải thiện CỤ THỂ",
        "includes": [
            "Đánh giá tổng quan swing hiện tại",
            "3 lỗi chính cần khắc phục",
            "Bài tập chi tiết (số lần, số hiệp, thời gian)",
            "Lộ trình luyện tập (tuần 1-4)",
            "Mục tiêu cụ thể (tăng điểm, consistency, velocity)"
        ],
        "how_to": "Tab 'Khuyến Nghị' sau khi phân tích video"
    },
    
    "compare_videos": {
        "name": "🎯 So Sánh Video PRO",
        "description": "So sánh swing của bạn với video mẫu PRO",
        "how_to": "Tab 'So Sánh Chi Tiết' → Chọn video PRO → Xem comparison side-by-side"
    },
    
    "export_results": {
        "name": "💾 Xuất Kết Quả",
        "description": "Tải về kết quả phân tích dạng PDF/CSV",
        "formats": ["PDF Report", "CSV Data", "JSON Metrics"],
        "how_to": "Button 'Xuất File' ở phần kết quả"
    },
    
    "thang_diem": {
        "name": "🏆 Thang Điểm",
        "description": "Hiểu ý nghĩa các cấp độ",
        "levels": {
            "85-100": "⛳ Cấp Độ PRO - Chuyên nghiệp, kỹ thuật hoàn hảo",
            "70-85": "⭐ Nâng Cao - Tốt, cần fine-tune nhỏ",
            "50-70": "📚 Trung Cấp - Cơ bản ổn, cần cải thiện consistency",
            "0-50": "💪 Mới Bắt Đầu - Cần xây dựng nền tảng từ đầu"
        }
    }
}

# Hướng dẫn sử dụng từng bước
STEP_BY_STEP_GUIDE = """
## 📖 HƯỚNG DẪN SỬ DỤNG ĐẦY ĐỦ

### Bước 1️⃣: Tải Video Lên
1. Click vào sidebar bên trái
2. Tìm phần "📤 Bước 1: Tải lên video golf swing"
3. Click nút "Browse files" hoặc kéo thả file vào
4. Chọn file video (.mp4, .avi, .mov)
5. Đợi video tải lên (hiển thị preview)

### Bước 2️⃣: Chọn Góc Quay
1. Phần "📐 Bước 2: Chọn góc quay"
2. Chọn "Tự động/Sau/Bên" tùy theo cách quay video
3. **Back view (Sau)**: Quay từ phía sau lưng golfer
4. **Side view (Bên)**: Quay từ bên hông

### Bước 3️⃣: Nhấn "Phân Tích" và Chờ
1. Click nút lớn "🔍 Phân Tích Video"
2. Đợi 10-30 giây (tùy độ dài video)
3. Hệ thống sẽ:
   - Phát hiện người trong video
   - Trích xuất các điểm khớp cơ thể
   - Tính toán góc độ, vận tốc
   - AI đánh giá và cho điểm

### Bước 4️⃣: Xem Kết Quả
1. **Tab "Tổng Quan"**: Xem điểm tổng hợp và cấp độ
2. **Tab "So Sánh Chi Tiết"**: So với PRO
3. **Tab "Khuyến Nghị"**: Đọc gợi ý từ AI về bài tập cải thiện
4. **Tab "Biểu Đồ"**: Xem visualization

### Bước 5️⃣: Tải Kết Quả (Tùy chọn)
1. Click "💾 Xuất File" 
2. Chọn định dạng (PDF/CSV)
3. Lưu vào máy
"""

# Câu hỏi thường gặp
FAQ = {
    "video_requirements": {
        "q": "Video cần đáp ứng yêu cầu gì?",
        "a": """
✅ **Yêu cầu video:**
- Độ dài: 5-60 giây
- Định dạng: MP4, AVI, MOV, MKV
- Chất lượng: HD (720p+) tốt nhất
- Ánh sáng: Đủ sáng, tránh ngược sáng
- Khung hình: Toàn thân golfer, từ đầu đến chân
- Góc quay: Cố định, không rung, không zoom in/out
- Background: Đơn giản, tránh quá nhiều người di chuyển
"""
    },
    
    "accuracy": {
        "q": "Độ chính xác của AI như thế nào?",
        "a": """
📊 **Độ chính xác:**
- Phát hiện pose: 95%+ (MediaPipe)
- Đánh giá cấp độ: 85-90% (so với chuyên gia)
- Consistency analysis: 90%+
- **Lưu ý:** Độ tin cậy hiển thị trong kết quả (70%+ là tốt)
"""
    },
    
    "low_score": {
        "q": "Tôi bị điểm thấp, có phải video sai không?",
        "a": """
🤔 **Nếu điểm thấp (<50):**
1. Kiểm tra "Độ tin cậy" - Nếu <60% → Video chất lượng kém
2. Nếu Độ tin cậy >70% → Điểm phản ánh đúng kỹ thuật hiện tại
3. Đọc phần "Khuyến Nghị AI" để biết lỗi cụ thể
4. Follow bài tập AI gợi ý để cải thiện

💡 **Mẹo:** Quay lại video sau 1-2 tuần luyện tập để thấy tiến bộ!
"""
    },
    
    "difference_back_side": {
        "q": "Khác biệt giữa Back view và Side view?",
        "a": """
📐 **So sánh góc quay:**

**Back View (Sau lưng):**
- Quay từ phía sau golfer, nhìn theo hướng bóng bay
- Phân tích tốt: Swing path, hip rotation, shoulder turn
- Khuyến nghị: Tốt nhất cho người mới

**Side View (Bên hông):**
- Quay từ bên phải/trái golfer
- Phân tích tốt: Spine angle, knee flex, arm angles
- Khuyến nghị: Cho golfer có kinh nghiệm muốn phân tích sâu
"""
    },
    
    "how_improve": {
        "q": "Làm sao để cải thiện điểm số?",
        "a": """
📈 **Cách cải thiện:**
1. **Đọc kỹ AI Recommendations** - Tập trung vào 3 lỗi chính
2. **Làm theo bài tập** - Đúng số lần, số hiệp được gợi ý
3. **Quay video thường xuyên** - Mỗi 1-2 tuần để track tiến bộ
4. **Focus vào Consistency** - Tăng độ nhất quán trước khi tăng power
5. **Kiên trì** - Cải thiện swing cần 4-8 tuần luyện tập đều đặn

🎯 **Mục tiêu thực tế:**
- Tuần 1-2: +3-5 điểm
- Tháng 1: +8-12 điểm
- Tháng 2-3: +15-20 điểm
"""
    }
}

# Quick reply suggestions
QUICK_REPLIES = [
    "🎥 Làm sao upload video?",
    "📊 Giải thích các chỉ số?",
    "🏆 Thang điểm nghĩa là gì?",
    "💪 Làm sao cải thiện điểm?",
    "🤖 AI Recommendations hoạt động thế nào?",
    "❓ Video cần yêu cầu gì?",
    "🔄 So sánh Back view vs Side view?",
    "💾 Tải kết quả về máy như thế nào?"
]

# Greeting messages
GREETINGS = [
    "Xin chào! Tôi là trợ lý ảo của VTK Team. Tôi có thể giúp gì cho bạn? 😊",
    "Chào mừng đến với Golf Swing Analysis! Bạn cần hỗ trợ gì? ⛳",
    "Hi! Tôi ở đây để hướng dẫn bạn sử dụng hệ thống. Hỏi tôi bất cứ điều gì! 🤖"
]

# Error messages
ERROR_MESSAGES = {
    "out_of_scope": "Xin lỗi, tôi chỉ hỗ trợ về hệ thống Golf Swing Analysis. Bạn có câu hỏi nào về phân tích swing không? 🏌️",
    "unclear": "Hmm, tôi chưa hiểu rõ câu hỏi. Bạn có thể hỏi rõ hơn hoặc chọn câu hỏi mẫu bên dưới nhé! 😊"
}