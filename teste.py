from ultralytics import YOLO
import cv2
import numpy as np

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cos_ang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

# ------------------------------
# Conectando à câmera do celular
# ------------------------------
ip_camera_url = "http://10.56.139.113:4747/video"  # troque pelo IP do seu celular
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("❌ Erro: não foi possível acessar a câmera do celular.")
    exit()
else:
    print("✅ Conectado à câmera do Android!")

model = YOLO("yolov8n-pose.pt")

print("🧍 Analisando postura em tempo real... Pressione 'q' para sair.")
