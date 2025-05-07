import streamlit as st
import numpy as np
import cv2

L = 256

def Spectrum(imgin):
    f = imgin.astype(np.float32)/(L-1)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    FR = F.real.copy()
    FI = F.imag.copy()
    S = np.sqrt(FR**2 + FI**2)
    S = np.clip(S, 0, L-1)
    return S.astype(np.uint8)

def FrequencyFilter(imgin, H):
    f = imgin.astype(np.float32)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    G = F * H
    G = np.fft.ifftshift(G)
    g = np.fft.ifft2(G)
    gR = g.real.copy()
    gR = np.clip(gR, 0, L-1)
    return gR.astype(np.uint8)

def CreateNotchFilter(P,Q):
    H = np.ones((P,Q), np.complex64)
    coords = [(44, 55), (85, 55), (40, 112), (81, 112)]
    D0 = 10
    for u in range(P):
        for v in range(Q):
            for (ui, vi) in coords + [(P-ui, Q-vi) for (ui, vi) in coords]:
                d = np.sqrt((u - ui)**2 + (v - vi)**2)
                if d < D0:
                    H[u,v] = 0
    return H

def CreateInterferenceFilter(M,N):
    H = np.ones((M,N), np.complex64)
    D0 = 7
    for u in range(M):
        for v in range(N):
            if u not in range(M//2-D0, M//2+D0+1):
                if v in range(N//2-D0, N//2+D0+1):
                    H[u,v] = 0.0
    return H

def CreateMotionFilter(M,N):
    H = np.ones((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(M):
        for v in range(N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi) < 1e-6:
                phi = phi_prev
            RE = T*np.sin(phi)/phi*np.cos(phi)
            IM = T*np.sin(phi)/phi*np.sin(phi)
            H[u,v] = RE + 1j*IM
            phi_prev = phi
    return H

def CreateDeMotionFilter(M,N):
    H = np.ones((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(M):
        for v in range(N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(np.sin(phi)) < 1e-6:
                phi = phi_prev
            RE = phi/(T*np.sin(phi))*np.cos(phi)
            IM = phi/T
            H[u,v] = RE + 1j*IM
            phi_prev = phi
    return H

def CreateWeinerFilter(M,N):
    H = CreateDeMotionFilter(M,N)
    P = H.real**2 + H.imag**2
    K = -0.5
    return H * P / (P + K)

# ----------------- Các hàm xử lý cao cấp -----------------
def RemoveMoire(imgin):
    M, N = imgin.shape
    H = CreateNotchFilter(M, N)
    return FrequencyFilter(imgin, H)

def RemoveInterference(imgin):
    M, N = imgin.shape
    H = CreateInterferenceFilter(M,N)
    return FrequencyFilter(imgin, H)

def CreateMotion(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M,N)
    return FrequencyFilter(imgin, H)

def DeMotion(imgin):
    M, N = imgin.shape
    H = CreateDeMotionFilter(M,N)
    return FrequencyFilter(imgin, H)

def DeMotionWeiner(imgin):
    M, N = imgin.shape
    H = CreateWeinerFilter(M,N)
    return FrequencyFilter(imgin, H)


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
        "RemoveMoire",
        "RemoveInterference",
        "CreateMotion",
        "DeMotion",
        "DeMotionWeiner",
        "DemotionNoise",
    )
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    imgin = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    with col1:
        st.image(imgin, caption="📷 Hình ảnh gốc", use_column_width=True)

    if technique == "Spectrum":
        processed_img = Spectrum(imgin)
    elif technique == "RemoveMoire":
        processed_img = RemoveMoire(imgin)
    elif technique == "RemoveInterference":
        processed_img = RemoveInterference(imgin)
    elif technique == "CreateMotion":
        processed_img = CreateMotion(imgin)
    elif technique == "DeMotion":
        processed_img = DeMotion(imgin)
    elif technique == "DeMotionWeiner":
        processed_img = DeMotionWeiner(imgin)
    elif technique == "DemotionNoise":
        temp = cv2.medianBlur(imgin,7)
        processed_img = DeMotion(temp)
    with col2:
        st.image(processed_img, caption="🛠️ Hình ảnh sau xử lý", use_column_width=True)