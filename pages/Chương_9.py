import cv2
import numpy as np
import streamlit as st

L = 256

def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    imgout = cv2.erode(imgin, w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgout = cv2.dilate(imgin, w)
    return imgout
    
def OpeningClosing(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_OPEN, w)
    imgout = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.erode(imgin,w)
    imgout = imgin - temp
    return imgout

def HoleFill(imgin):
    imgout = imgin
    M, N = imgout.shape
    mask = np.zeros((M+2,N+2),np.uint8)
    cv2.floodFill(imgout,mask,(105,297),L-1)
    return imgout

def MyConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    M, N = temp.shape
    dem = 0
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            if temp[x,y] == L-1:
                mask = np.zeros((M+2,N+2),np.uint8)
                cv2.floodFill(temp, mask, (y,x), (color,color,color))
                dem = dem + 1
                color = color + 1
    print('Co %d thanh phan lien thong' % dem)
    a = np.zeros(L, int)
    for x in range(0, M):
        for y in range(0, N):
            r = temp[x,y]
            if r > 0:
                a[r] = a[r] + 1
    dem = 1
    for r in range(0, L):
        if a[r] > 0:
            print('%4d   %5d' % (dem, a[r]))
            dem = dem + 1
    return temp

def ConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d thanh phan lien thong' % (dem-1) 
    print(text)

    a = np.zeros(dem, dtype=int)  # Sử dụng `dtype=int` thay vì `np.int`
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(1, dem):
        print('%4d %10d' % (r, a[r]))
    label = label.astype(np.uint8)
    cv2.putText(label, text, (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return label

def CountRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d hat gao' % (dem-1) 
    print(text)
    a = np.zeros(dem, dtype=int)
    M, N = label.shape
    color = 150

    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x, y] = label[x, y] + color

    for r in range(0, dem):
        print('%4d %10d' % (r, a[r]))

    max_count = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max_count:
            max_count = a[r]
            rmax = r

    xoa = np.array([], dtype=int)
    for r in range(1, dem):
        if a[r] < 0.5 * max_count:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                r = r - color
                if r in xoa:
                    label[x, y] = 0

    label = label.astype(np.uint8)
    cv2.putText(label, text, (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return label

# Streamlit App
st.title("Xử lý ảnh")

uploaded_file = st.file_uploader("Chọn hình ảnh...", type=["jpg", "jpeg", "png", "tif"])
technique = st.selectbox(
        "Chọn kỹ thuật",
        ("Erosion",
         "Dilation",
         "OpeningClosing",
         "Boundary",
         "HoleFill",
         "MyConnectedComponent",
         "ConnectedComponent",
         "CountRice")
    )
col1, col2 = st.columns([1,1])

if uploaded_file is not None:
    imgin = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    with col1:  st.image(imgin, caption="Hình ảnh tải lên", use_column_width=True)

    if technique == "Erosion":
        processed_img = Erosion(imgin)

    elif technique == "Dilation":
        processed_img = Dilation(imgin)

    elif technique == "OpeningClosing":
        processed_img = OpeningClosing(imgin)

    elif technique == "Boundary":
        processed_img = Boundary(imgin)

    elif technique == "HoleFill":
        processed_img = HoleFill(imgin)

    elif technique == "MyConnectedComponent":
        processed_img = MyConnectedComponent(imgin)

    elif technique == "ConnectedComponent":
        processed_img = ConnectedComponent(imgin)

    elif technique == "CountRice":
        processed_img = CountRice(imgin)    

    with col2:  st.image(processed_img, caption="Hình ảnh đã xử lý", use_column_width=True)
