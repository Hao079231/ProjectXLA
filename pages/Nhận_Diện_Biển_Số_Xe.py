import streamlit as st
from PIL import Image
from Nhan_Dien_Bien_So_Xe.Image_test import predict
import cv2
import numpy as np

def main():
    st.title('Nhận Diện Biển Số Xe')
    img_in = None
    
    # Tải lên hình ảnh
    uploaded_img1 = st.file_uploader("Tải lên hình ảnh 1", type=["jpg", "png", "jpeg"], key="image1")
    
    # Layout cho 2 hình ảnh và văn bản
    col1, col2= st.columns([2, 2])
    
    with col1:
        if uploaded_img1 is not None:
            img = Image.open(uploaded_img1)
            frame = np.array(img)
            img_in = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            st.image(img, caption='Hình ảnh 1', use_column_width=True)
    if st.button('Predict'):
        img_out, text = predict(img_in)
        for i in range(len(text)):
            with col2:  
                st.image(img_out[i], caption='Biển số', use_column_width=True)
                st.write("Kết quả", text[i])

if __name__ == "__main__":
    main()
