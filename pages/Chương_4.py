import streamlit as st
import numpy as np
import cv2

L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin
    fp = fp/(L-1)
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S, 0, L-1)
    return S.astype(np.uint8)

def FrequencyFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)
    H = np.zeros((P,Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u-P//2)**2 + (v-Q//2)**2)
            if Duv > 0:
                H[u,v] = 1.0/(1.0 + np.power(D0/Duv,2*n))
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    gp = g[:,:,0]
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    imgout = gp[0:M,0:N]
    return np.clip(imgout,0,L-1).astype(np.uint8)

def CreateNotchRejectFilter():
    P = 250
    Q = 180
    D0 = 10
    n = 2
    H = np.ones((P,Q), np.float32)
    coords = [(44, 58), (40, 119), (86, 59), (82, 119)]
    for u in range(P):
        for v in range(Q):
            h = 1.0
            for (ui, vi) in coords:
                for du, dv in [(ui, vi), (P-ui, Q-vi)]:
                    Duv = np.sqrt((u-du)**2 + (v-dv)**2)
                    h *= 1.0/(1.0 + np.power(D0/Duv,2*n)) if Duv > 0 else 0.0
            H[u,v] = h
    return H

def DrawNotchRejectFilter():
    H = CreateNotchRejectFilter()
    return (H*(L-1)).astype(np.uint8)

def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)
    H = CreateNotchRejectFilter()
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    gp = g[:,:,0]
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    imgout = gp[0:M,0:N]
    return np.clip(imgout,0,L-1).astype(np.uint8)


st.set_page_config(page_title="Xử lý ảnh tần số", page_icon="📈", layout="wide")

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

        .stFileUploader, .stSelectbox {
            background-color: #0077b6!important;
            border-radius: 10px !important;
            padding: 10px 12px !important;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .stSelectbox > div > div {
            color: #ffffff;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Ứng dụng Xử lý ảnh trong miền tần số")

uploaded_file = st.file_uploader("📁 Chọn hình ảnh...", type=["jpg", "jpeg", "png", "tif"])
technique = st.selectbox(
    "🛠️ Chọn kỹ thuật xử lý:",
    (
        "Spectrum",
        "FrequencyFilter",
        "CreateNotchRejectFilter",
        "DrawNotchRejectFilter",
        "RemoveMoire"
    )
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    imgin = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    with col1:
        st.image(imgin, caption="📷 Hình ảnh gốc", use_column_width=True)

    if technique == "Spectrum":
        processed_img = Spectrum(imgin)
    elif technique == "FrequencyFilter":
        processed_img = FrequencyFilter(imgin)
    elif technique == "CreateNotchRejectFilter":
        processed_img = CreateNotchRejectFilter()
    elif technique == "DrawNotchRejectFilter":
        processed_img = DrawNotchRejectFilter()
    elif technique == "RemoveMoire":
        processed_img = RemoveMoire(imgin)

    with col2:
        st.image(processed_img, caption="🛠️ Hình ảnh sau xử lý", use_column_width=True)
