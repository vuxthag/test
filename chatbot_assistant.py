import streamlit as st
import os
from openai import OpenAI

# =====================================================
# OPENAI CLIENT
# =====================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# CONTEXT (READ ONLY)
# =====================================================
def get_analysis_context():
    """
    An toàn tuyệt đối:
    - Chưa phân tích video → trả về context rỗng
    - last_analysis = None → không crash
    """

    ana = st.session_state.get("last_analysis")

    # ✅ CHƯA CÓ HOẶC LÀ NONE
    if not isinstance(ana, dict):
        return "Chưa có dữ liệu swing. Người chơi chưa phân tích video."

    return f"""
Thông tin swing (chỉ để tham khảo):
- Điểm: {ana.get('score', 0):.1f}/100
- Cấp độ: {ana.get('level', '')}
- Percentile: P{ana.get('percentile', 0):.0f}
- Độ ổn định: {ana.get('top_metrics', {}).get('consistency', 0):.1f}%
- Vận tốc cổ tay max: {ana.get('top_metrics', {}).get('velocity', {}).get('max', 0):.2f} m/s
""".strip()



# =====================================================
# CHATGPT – HLV GOLF
# =====================================================
def chatgpt_ai_chat(user_message: str, context: str) -> str:

    system_prompt = """
Bạn là HLV Golf ngoài đời thật, đang nói chuyện trực tiếp với học viên.

Yêu cầu:
- Nói chuyện tự nhiên, thân thiện
- Không viết báo cáo
- Không chia mục
- Dữ liệu chỉ dùng ngầm
- Có thể hỏi ngược lại người chơi
"""

    user_prompt = f"""
{context}

Học viên hỏi:
{user_message}

Trả lời như HLV đang đứng cạnh người chơi.
"""

    res = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_tokens=500,
    )

    return res.choices[0].message.content.strip()


# =====================================================
# CHATBOT – GIỮ NGUYÊN UI CỦA BẠN
# =====================================================
def render_chatbot():

    # INIT CHAT STATE
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Chào mừng đến với Golf Swing Analysis! "
                    "Tôi là trợ lý VTK, tôi có thể giúp gì cho bạn? ⛳"
                )
            }
        ]

    # ===== CSS + HEADER (GIỮ NGUYÊN) =====
    st.markdown("""
    <style>
        .chat-container {
            background-color: #f0f2f6;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 10px;
            max-height: 450px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 85%;
            font-size: 14px;
            line-height: 1.4;
        }
        .assistant-bubble {
            background-color: #ffffff;
            color: #333;
            border-bottom-left-radius: 2px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }
        .user-bubble {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 2px;
        }
        .chat-header {
            background: linear-gradient(135deg, #ee0033, #aa0022);
            color: white;
            padding: 10px;
            border-radius: 10px 10px 0 0;
            text-align: center;
            font-weight: bold;
            margin-bottom: 6px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-header">🤖 Trợ Lý Ảo VTK Golf</div>', unsafe_allow_html=True)

    # ===== CLEAR CHAT BUTTON =====
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_messages = st.session_state.chat_messages[:1]

    # ===== CHAT PLACEHOLDER (FIX TRỄ CHAT) =====
    chat_placeholder = st.empty()

    def render_chat():
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_messages:
            bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            chat_html += f'<div class="chat-bubble {bubble_class}">{msg["content"]}</div>'
        chat_html += '</div>'
        chat_placeholder.markdown(chat_html, unsafe_allow_html=True)

    # Render lần đầu
    render_chat()

    # ===== INPUT =====
    if prompt := st.chat_input("Nhập nội dung cần hỗ trợ..."):

        # 1️⃣ Append USER
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )
        render_chat()  # 🔥 HIỆN NGAY – FIX CHAT TRỄ

        # 2️⃣ AI trả lời
        with st.spinner("🤖 AI đang trao đổi cùng bạn..."):
            context = get_analysis_context()
            reply = chatgpt_ai_chat(prompt, context)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": reply}
        )
        render_chat()  # 🔥 HIỆN NGAY – KHÔNG ĐỢI LẦN SAU





