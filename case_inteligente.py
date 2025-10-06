from ultralytics import YOLO
import cv2
import numpy as np
import time

# ------------------------------
# Função para calcular ângulo
# ------------------------------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cos_ang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

# ------------------------------
# Função para avaliar postura
# ------------------------------
def avaliar_postura(angulo_coluna, inclinacao_cabeca, dif_ombros):
    if angulo_coluna >= 75 and inclinacao_cabeca >= 130 and dif_ombros <= 20:
        return "Postura boa 👍", (0, 255, 0)
    else:
        return "Postura ruim 🚨", (0, 0, 255)

# ------------------------------
# Inicialização
# ------------------------------
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ou URL do celular
if not cap.isOpened():
    print("Erro: não foi possível acessar a câmera.")
    exit()

print("🧍 Pressione 'q' para sair e 'f' para feedback manual.")

# Listas para média do feedback
angulo_coluna_list = []
inclinacao_cabeca_list = []
dif_ombros_list = []
start_time = time.time()
feedback_interval = 10  # segundos
last_feedback_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        nariz = keypoints[0]
        ombro_dir = keypoints[5]
        ombro_esq = keypoints[6]
        quadril_dir = keypoints[11]
        quadril_esq = keypoints[12]
        joelho_dir = keypoints[13]
        joelho_esq = keypoints[14]

        angulo_coluna = calcular_angulo(ombro_esq, quadril_esq, joelho_esq)
        inclinacao_cabeca = calcular_angulo(ombro_esq, nariz, quadril_esq)
        dif_ombros = abs(ombro_esq[1] - ombro_dir[1])

        # Armazenar para média
        angulo_coluna_list.append(angulo_coluna)
        inclinacao_cabeca_list.append(inclinacao_cabeca)
        dif_ombros_list.append(dif_ombros)

        # Feedback atual
        feedback_text, color = avaliar_postura(angulo_coluna, inclinacao_cabeca, dif_ombros)

        # Exibir texto grande no vídeo
        cv2.putText(
            annotated_frame,
            feedback_text,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            color,
            4
        )

    cv2.imshow("Espelho Ergonômico Inteligente", annotated_frame)

    # Feedback automático a cada X segundos (opcional para terminal)
    if time.time() - last_feedback_time >= feedback_interval:
        if angulo_coluna_list:
            media_coluna = np.mean(angulo_coluna_list)
            media_cabeca = np.mean(inclinacao_cabeca_list)
            media_ombros = np.mean(dif_ombros_list)
            feedback_auto, _ = avaliar_postura(media_coluna, media_cabeca, media_ombros)
            print(f"\n📌 Feedback automático (últimos {feedback_interval}s): {feedback_auto}")
        angulo_coluna_list.clear()
        inclinacao_cabeca_list.clear()
        dif_ombros_list.clear()
        last_feedback_time = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Feedback manual
        if angulo_coluna_list:
            media_coluna = np.mean(angulo_coluna_list)
            media_cabeca = np.mean(inclinacao_cabeca_list)
            media_ombros = np.mean(dif_ombros_list)
            feedback_manual, _ = avaliar_postura(media_coluna, media_cabeca, media_ombros)
            print(f"\n📌 Feedback manual: {feedback_manual}")

cap.release()
cv2.destroyAllWindows()
