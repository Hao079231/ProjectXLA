import cv2
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Nhận dạng màu sắc",
    page_icon="🎨",
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

        .stFileUploader, .stSelectbox, .stButton, .stCheckbox {
            background-color: #0077b6 !important;
            border-radius: 10px !important;
            padding: 5px 10px !important;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .css-1aumxhk, .css-1v0mbdj, .css-1x8cf1d {  /* Container chung */
            color: #ffffff !important;
        }

        .stSelectbox > div > div, .stCheckbox > label {
            color: #ffffff;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title('Nhận dạng màu sắc')

def loadmausac():
    # Bắt đầu camera
    camera_on = st.checkbox("📷 Bật/Tắt Camera")

    if camera_on:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể kết nối với camera. Vui lòng kiểm tra lại.")
            return
        
        cap.set(3, 720)
        cap.set(4, 1280)
        
        video_placeholder = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc khung hình từ camera. Vui lòng kiểm tra lại.")
                break
            
            frame = cv2.flip(frame, 1)
            
            x, y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)
            b, g, r = frame[y, x]
            b, g, r = int(b), int(g), int(r)
            
            color_name = getColorName(b, g, r)
            drawSquare(frame, x, y)
            putText(frame, x, y, color_name, b, g, r)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Dừng camera khi kết thúc
        cap.release()
        cv2.destroyAllWindows()

def drawSquare(img, x, y):
    YELLOW = (0, 255, 255)
    BLUE = (255, 225, 0)
    
    cv2.line(img, (x - 150, y - 150), (x - 100, y - 150), YELLOW, 2)
    cv2.line(img, (x - 150, y - 150), (x - 150, y - 100), BLUE, 2)
    
    cv2.line(img, (x + 150, y - 150), (x + 100, y - 150), YELLOW, 2)
    cv2.line(img, (x + 150, y - 150), (x + 150, y - 100), BLUE, 2)
    
    cv2.line(img, (x + 150, y + 150), (x + 100, y + 150), YELLOW, 2)
    cv2.line(img, (x + 150, y + 150), (x + 150, y + 100), BLUE, 2)
    
    cv2.line(img, (x - 150, y + 150), (x - 100, y + 150), YELLOW, 2)
    cv2.line(img, (x - 150, y + 150), (x - 150, y + 100), BLUE, 2)
    
    cv2.circle(img, (x, y), 5, (255, 255, 153), -1)

index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv("Nhan_Dien_Mau_Sac/colors.csv", names=index, header=None)

def getColorName(b, g, r):
    minimum = 1000 
    cname = None
    for i in range(len(csv)):
        d = abs(b - int(csv.loc[i, "B"])) + abs(g - int(csv.loc[i, "G"])) + abs(r - int(csv.loc[i, "R"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

def putText(img, x, y, color_name, b, g, r):
    cv2.rectangle(img, (x - 150, y - 220), (x + 300, y - 170), (b, g, r), -1)
    text = f"{color_name} | R={r} G={g} B={b}"
    cv2.putText(img, text, (x - 140, y - 190), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

loadmausac()