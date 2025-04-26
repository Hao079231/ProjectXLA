import cv2
import time
import os
import finger.hand as htm
import streamlit as st

st.set_page_config(
    page_title="Đếm ngón tay",
    page_icon="🖐️",
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

st.title('Đếm ngón tay')

camera_on = st.checkbox("📷 Bật/Tắt Camera")

FolderPath="finger/Fingers"
lst=os.listdir(FolderPath)
print(lst)
lst_2=[]  # khai báo list chứa các mảng giá trị của các hình ảnh/
for i in lst:
    image=cv2.imread(f"{FolderPath}/{i}")  # Fingers/1.jpg , Fingers/2.jpg ...
    lst_2.append(image)

if camera_on:
    cap=cv2.VideoCapture(0) # nếu có nhiều cam thì thêm id webcam  1,2,3..
    if not cap.isOpened():
        st.error("Không thể kết nối với camera. Vui lòng kiểm tra lại.")
        camera_on = False

    pTime=0

    detector = htm.handDetector(detectionCon=0.55)

    fingerid= [4,8,12,16,20]
    video_placeholder = st.image([])

    while camera_on:
        ret,frame =cap.read()
        if not ret:
            st.error("Không thể đọc khung hình từ camera. Vui lòng kiểm tra lại.")
            break

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False) # phát hiện vị trí

        if len(lmList) !=0:
            fingers= []
            # viết cho ngón cái (ý tường là điểm 4 ở bên trái hay bên phải điểm 2 )
            if lmList[fingerid[0]][1] < lmList[fingerid[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # viết cho 4 ngón dài
            for id in range(1,5):
                if lmList[fingerid[id]][2] < lmList[fingerid[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            songontay=fingers.count(1)

            h, w, c = lst_2[songontay-1].shape
            frame[0:h,0:w] = lst_2[songontay-1]  # nếu số ngón tay =0 thì lst_2[-1] đẩy về phần tử cuối cùng của list là ảnh 6

            # vẽ thêm hình chữ nhật hiện số ngón tay
            cv2.rectangle(frame,(0,200),(150,400),(0,255,0),-1)
            cv2.putText(frame,str(songontay),(30,390),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),5)

        cTime=time.time()  # trả về số giây, tính từ 0:0:00 ngày 1/1/1970 theo giờ  utc , gọi là(thời điểm bắt đầu thời gian)
        fps=1/(cTime-pTime) # tính fps Frames per second - đây là  chỉ số khung hình trên mỗi giây
        pTime=cTime
        # show fps lên màn hình, fps hiện đang là kiểu float , ktra print(type(fps))
        cv2.putText(frame, f"FPS: {int(fps)}",(150,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame)
        if cv2.waitKey(1)== ord("q"): # độ trễ 1/1000s , nếu bấm q sẽ thoát
            break
    
    cap.release() # giải phóng camera
    cv2.destroyAllWindows() # thoát tất cả các cửa sổ