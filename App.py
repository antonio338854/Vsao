import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np

# --- Configuração Visual (Tela Cheia) ---
st.set_page_config(page_title="Hand Tracking Pro", page_icon="🖐️", layout="wide")

# CSS para remover as margens e deixar a câmera gigante no celular
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    h1 {text-align: center; font-size: 20px; color: #00ff00;}
    /* Tenta forçar o vídeo a ocupar largura total */
    video { width: 100% !important; height: auto !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🖐️ Detector de Mãos em Tempo Real")
st.caption("Vovô diz: Aponte a câmera para sua mão!")

# --- Configuração do MediaPipe (O Cérebro) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuração da Mão
hands_service = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,     # Detecta até 2 mãos
    model_complexity=0,  # 0 é mais rápido para celular, 1 é mais preciso
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Processador de Vídeo (Onde a mágica acontece) ---
class HandDetectorProcessor:
    def recv(self, frame):
        # 1. Converter frame do WebRTC para formato OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Espelhar a imagem (opcional, bom para selfie, ruim para traseira)
        # img = cv2.flip(img, 1)

        # 3. Converter para RGB (O MediaPipe exige RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 4. Processar detecção
        results = hands_service.process(img_rgb)
        
        # 5. Se achou mãos, desenha o esqueleto
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os pontos e as linhas (ossos)
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # 6. Retorna o frame desenhado para a tela do celular
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Configuração do WebRTC (A Conexão) ---
# Isso ajuda a funcionar no 4G/5G (Servidor STUN do Google)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Componente na Tela ---
webrtc_streamer(
    key="hand-detection",
    mode=WebRtcMode.SENDRECV,
