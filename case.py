from ultralytics import YOLO
import cv2
import numpy as np

# ------------------------------
# Função para calcular ângulo
# ------------------------------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cos_ang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


# ------------------------------
# Carregar modelo YOLOv8 de pose
# ------------------------------
model = YOLO("yolov8n-pose.pt")  # Modelo leve de pose humana

# ------------------------------
# Carregar imagem
# ------------------------------
img_path = "pessoa_sentada.jpg"  # coloque o nome da sua imagem
img = cv2.imread(img_path)

# Executar detecção
results = model(img)
keypoints = results[0].keypoints.xy[0].cpu().numpy()  # coordenadas dos pontos

# Verificar se há pessoa detectada
if keypoints.shape[0] == 0:
    print("Nenhuma pessoa detectada na imagem.")
    exit()

# ------------------------------
# Identificar pontos de interesse
# ------------------------------
# YOLOv8 retorna 17 pontos no formato COCO:
# 0-nose, 5-shoulder direito, 6-shoulder esquerdo, 11-hip direito, 12-hip esquerdo, etc.
nariz = keypoints[0]
ombro_dir = keypoints[5]
ombro_esq = keypoints[6]
quadril_dir = keypoints[11]
quadril_esq = keypoints[12]
joelho_dir = keypoints[13]
joelho_esq = keypoints[14]

# ------------------------------
# Calcular ângulos e alinhamentos
# ------------------------------
angulo_coluna = calcular_angulo(ombro_esq, quadril_esq, joelho_esq)
inclinacao_cabeca = calcular_angulo(ombro_esq, nariz, quadril_esq)
dif_ombros = abs(ombro_esq[1] - ombro_dir[1])

print(f"Ângulo da coluna: {angulo_coluna:.1f}°")
print(f"Inclinação da cabeça: {inclinacao_cabeca:.1f}°")
print(f"Diferença de altura entre ombros: {dif_ombros:.1f}px")

# ------------------------------
# Avaliação ergonômica
# ------------------------------
print("\n🧍 Avaliação da postura:")
if angulo_coluna < 75:
    print("- Coluna inclinada para frente 🚨")
else:
    print("- Coluna ereta 👍")

if inclinacao_cabeca < 130:
    print("- Cabeça inclinada para frente 🚨")
else:
    print("- Cabeça alinhada 👍")

if dif_ombros > 20:
    print("- Ombros desnivelados 🚨")
else:
    print("- Ombros alinhados 👍")

# ------------------------------
# Exibir imagem com pose desenhada
# ------------------------------
annotated_frame = results[0].plot()  # YOLO já desenha a pose
cv2.imshow("Análise de Postura - YOLOv8 Pose", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
