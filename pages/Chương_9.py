import streamlit as st
import numpy as np
import cv2
from PIL import Image

L = 256

def Erosion(imgin):
    w = cv2. getStructuringElement(cv2.MORPH_RECT, (45,45))
    imgout = cv2.erode(imgin,w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgout = cv2.erode(imgin,w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    return cv2.subtract(imgin, temp)

def Contour(imgin):
    img_color = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, (0, 128, 0), 2)  # Màu xanh
    return img_color

st.set_page_config(
    page_title="Xử lý ảnh - Chương 9",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #d0e6f7, #a0d2eb);
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #ffffff;
            text-align: center;
            background-color: #0077b6;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }
        .stFileUploader, .stSelectbox, .stButton {
            background-color: #0077b6 !important;
            border-radius: 10px !important;
            padding: 5px 10px !important;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .css-1aumxhk, .css-1v0mbdj, .css-1x8cf1d {
            color: #ffffff !important;
        }
        .stSelectbox > div > div {
            color: #ffffff;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Ứng dụng Xử lý ảnh - Chương 9")

uploaded_file = st.file_uploader("📁 Chọn hình ảnh...", type=["jpg", "jpeg", "png", "tif"])

technique = st.selectbox(
    "🛠️ Chọn kỹ thuật xử lý ảnh",
    ("Erosion", "Dilation", "Boundary", "Contour")
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    frame = np.array(image)
    
    with col1:
        st.image(frame, caption="📷 Hình ảnh gốc", use_column_width=True, channels="GRAY")

    # Process the image
    if technique == "Erosion":
        processed_img = Erosion(frame)
        channels = "GRAY"
    elif technique == "Dilation":
        processed_img = Dilation(frame)
        channels = "GRAY"
    elif technique == "Boundary":
        processed_img = Boundary(frame)
        channels = "GRAY"
    elif technique == "Contour":
        processed_img = Contour(frame)
        channels = "BGR"

    with col2:
        st.image(processed_img, caption="🛠️ Hình ảnh đã xử lý", use_column_width=True, channels=channels)