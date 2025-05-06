import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Báo cáo cuối kỳ",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #d0e6f7, #a0d2eb);
            font-family: 'Segoe UI', sans-serif;
        }

        .main-title {
            color: #ffffff;
            font-size: 2.8em;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            letter-spacing: 1px;
            padding: 15px;
            background-color: #0077b6;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }

        .content-card {
            background-color: #05445E; /* Xanh đậm */
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .content-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
        }

        .content-card * {
            color: #148783 !important;
        }

        .toc-header {
            color: #148783
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 20px;
        }

        .toc-list {
            color: #148783
            list-style: none;
            padding-left: 0;
        }

        .toc-list li {
            color: #148783
            margin: 10px 0;
            display: flex;
            align-items: center;
            font-size: 1.2em;
            transition: all 0.3s ease;
        }

        .toc-list li:hover {
            color: #00bcd4;
            font-weight: 500;
        }

        .toc-list li::before {
            content: "📌";
            margin-right: 10px;
            color: #00bcd4;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Chào mừng thầy đến với bài báo cáo cuối kỳ! 👋</div>', unsafe_allow_html=True)

# Student Info Section
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("""
    <div class="content-text">
        <h3>📚 Sinh viên thực hiện</h3>
        <h4>🎓 Trịnh Trung Hào - 22110316</h4>
        <h4>🎓 Lê Xuân Thịnh - 22110427</h4>
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Table of Contents Section
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown('<div class="toc-header">📋 Mục lục bài báo cáo:</div>', unsafe_allow_html=True)
st.markdown(
    """
    <ul class="toc-list">
        <li>Nhận diện trái cây</li>
        <li>Nhận diện khuôn mặt</li>
        <li>Phát hiện đối tượng</li>
        <li>Nhận diện biển báo trên đường bộ</li>
        <li>Nhận diện màu sắc</li>
        <li>Nhận diện cờ tướng</li>   
        <li>Đếm ngón tay</li>
        <li>Xử lý ảnh (Chương 3, 4, 9)</li>
    </ul>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)