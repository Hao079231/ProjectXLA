import streamlit as st
import numpy as np
import cv2
from PIL import Image

L = 256

def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            s = L-1-r
            imgout[x,y] = s
    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r == 0:
                r = 1
            s = c*np.log(1+r)
            imgout[x,y] = np.uint8(s)
    return imgout

def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    gamma = 5.0
    c = np.power(L-1,1-gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            s = c*np.power(r,gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    rmin, rmax, vi_tri_rmin, vi_tri_rmax = cv2.minMaxLoc(imgin)
    L = 256
    
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r < r1:
                if r1 != 0:  
                    s = s1 / r1 * r
                else:
                    s = 0
            elif r < r2:
                if (r2 - r1) != 0:  
                    s = (s2 - s1) / (r2 - r1) * (r - r1) + s1
                else:
                    s = 0
            else:
                if (L - 1 - r2) != 0:  
                    s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
                else:
                    s = 0
            imgout[x, y] = np.uint8(s)
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,L), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h/(M*N)
    scale = 2000
    for r in range(0, L):
        cv2.line(imgout,(r,M-1),(r,M-1-int(scale*p[r])), (0,0,0))
    return imgout

def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h/(M*N)

    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k+1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

def HistEqualColor(imgin):
    B = imgin[:,:,0]
    G = imgin[:,:,1]
    R = imgin[:,:,2]
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    imgout = np.array([B, G, R])
    imgout = np.transpose(imgout, axes = [1,2,0]) 
    return imgout

def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            w = cv2.equalizeHist(w)
            imgout[x,y] = w[a,b]
    return imgout

def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    mG, sigmaG = cv2.meanStdDev(imgin)
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            msxy, sigmasxy = cv2.meanStdDev(w)
            r = imgin[x,y]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*r)
            else:
                imgout[x,y] = r
    return imgout

def MyBoxFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 11
    n = 11
    w = np.ones((m,n))
    w = w/(m*n)

    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, M-b):
            r = 0.0
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    r = r + w[s+a,t+b]*imgin[x+s,y+t]
            imgout[x,y] = np.uint8(r)
    return imgout

def BoxFilter(imgin):
    m = 21
    n = 21
    w = np.ones((m,n))
    w = w/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
    return imgout

def Threshold(imgin):
    temp = cv2.blur(imgin, (15,15))
    retval, imgout = cv2.threshold(temp,64,255,cv2.THRESH_BINARY)
    return imgout

def MedianFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 5
    n = 5
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(0, M):
        for y in range(0, N):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[(x+s)%M,(y+t)%N]
            w_1D = np.reshape(w, (m*n,))
            w_1D = np.sort(w_1D)
            imgout[x,y] = w_1D[m*n//2]
    return imgout

def Sharpen(imgin):
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout
 
def Gradient(imgin):
    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    gx = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 1, dy = 0)
    gy = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 0, dy = 1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

st.set_page_config(
    page_title="Xử lý ảnh cơ bản",
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

        .css-1aumxhk, .css-1v0mbdj, .css-1x8cf1d {  /* Container chung */
            color: #ffffff !important;
        }

        .stSelectbox > div > div {
            color: #ffffff;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Ứng dụng Xử lý ảnh cơ bản")

uploaded_file = st.file_uploader("📁 Chọn hình ảnh...", type=["jpg", "jpeg", "png", "tif"])

technique = st.selectbox(
    "🛠️ Chọn kỹ thuật xử lý ảnh",
    (
        "Negative",
        "Logarit",
        "Power",
        "PiecewiseLinear",
        "Histogram",
        "HistEqual",
        "HistEqualColor",
        "LocalHist",
        "HistStat",
        "MyBoxFilter",
        "BoxFilter",
        "Threshold",
        "MedianFilter",
        "Sharpen",
        "Gradient"
    )
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    imgin = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    with col1:
        st_image = st.image(imgin, caption="📷 Hình ảnh tải lên", use_column_width=True)

    if technique == "Negative":
        processed_img = Negative(imgin)
    elif technique == "Logarit":
        processed_img = Logarit(imgin)
    elif technique == "Power":
        processed_img = Power(imgin)
    elif technique == "PiecewiseLinear":
        processed_img = PiecewiseLinear(imgin)
    elif technique == "Histogram":
        processed_img = Histogram(imgin)
    elif technique == "HistEqual":
        processed_img = HistEqual(imgin)
    elif technique == "HistEqualColor":
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_img = HistEqualColor(frame)
        with col1:
            st_image.empty()
            st.image(frame, caption="📷 Hình ảnh gốc (RGB)", use_column_width=True)
    elif technique == "LocalHist":
        processed_img = LocalHist(imgin)
    elif technique == "HistStat":
        processed_img = HistStat(imgin)
    elif technique == "MyBoxFilter":
        processed_img = MyBoxFilter(imgin)
    elif technique == "BoxFilter":
        processed_img = BoxFilter(imgin)
    elif technique == "Threshold":
        processed_img = Threshold(imgin)
    elif technique == "MedianFilter":
        processed_img = MedianFilter(imgin)
    elif technique == "Sharpen":
        processed_img = Sharpen(imgin)
    elif technique == "Gradient":
        processed_img = Gradient(imgin)

    with col2:
        st.image(processed_img, caption="🛠️ Hình ảnh đã xử lý", use_column_width=True)
