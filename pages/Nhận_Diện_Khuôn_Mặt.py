import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os

st.set_page_config(
    page_title="Nhận diện khuôn mặt",
    page_icon="📸",
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

        .css-1aumxhk, .css-1v0mbdj, .css-1x8cf1d {  /* Container chung */
            color: #ffffff !important;
        }

        .stSelectbox > div > div {
            color: #ffffff;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title('Nhận diện khuôn mặt')

FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

if 'stop' not in st.session_state:
    st.session_state.stop = False

col1, col2 = st.columns([1,1])
with col1:
    click = st.button("Nhận diện")
with col2:
    press = st.button('Dừng lại')

if press:
    st.session_state.stop = not st.session_state.stop
    if st.session_state.stop:
        cap.release()
    else:
        cap = cv.VideoCapture(0)

if 'frame_stop' not in st.session_state:
    stop_image_path = 'Nhan_Dien_Khuon_Mat/stop.jpg'
    if os.path.exists(stop_image_path):
        frame_stop = cv.imread(stop_image_path)
        st.session_state.frame_stop = frame_stop if frame_stop is not None else None
    else:
        st.session_state.frame_stop = None

if st.session_state.stop and st.session_state.frame_stop is not None:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')

# Load models
try:
    svc, encoder = joblib.load('Nhan_Dien_Khuon_Mat/svc.pkl') 
    detector = cv.FaceDetectorYN.create(
        'Nhan_Dien_Khuon_Mat/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    recognizer = cv.FaceRecognizerSF.create(
        'Nhan_Dien_Khuon_Mat/face_recognition_sface_2021dec.onnx', "")
except Exception as e:
    st.error(f"Không thể tải mô hình: {e}")
    st.stop()

def visualize(input, faces, fps, thickness=2):
    if input is None:
        return input
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            try:
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                for i in range(5):  # 5 landmarks
                    cv.circle(input, (coords[4+i*2], coords[5+i*2]), 2, (0, 255, 255), thickness)
                face_align = recognizer.alignCrop(input, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature.reshape(1, -1))
                result = encoder.inverse_transform(test_predict)[0]
                cv.putText(input, result, (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Lỗi khi xử lý khuôn mặt {idx}: {e}")
    cv.putText(input, f'FPS: {fps:.2f}', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return input

if __name__ == '__main__':
    if click and not st.session_state.stop:
        tm = cv.TickMeter()
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

        while True:
            if st.session_state.stop:
                break
            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            frame = cv.flip(frame, 1)
            tm.start()
            faces = detector.detect(frame)
            tm.stop()
            frame = visualize(frame, faces, tm.getFPS())
            FRAME_WINDOW.image(frame, channels='BGR')
        cap.release()
        cv.destroyAllWindows()
