import cv2
import math
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import mediapipe as mp

# --- Configuração MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Função para calcular ângulo entre 3 pontos
def calcular_angulo(p1, p2, p3):
    angulo = math.degrees(
        math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
        math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    )
    angulo = abs(angulo)
    if angulo > 180:
        angulo = 360 - angulo
    return angulo

# Função para desenhar landmarks na imagem
def desenhar_landmarks(imagem, landmarks):
    mp_drawing.draw_landmarks(
        imagem,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
    )

# Extrair ângulos e gerar imagem anotada
def extrair_medidas(img_path, save_out):
    imagem = cv2.imread(img_path)
    h, w, _ = imagem.shape

    rgb_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(rgb_image)

    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    def ponto(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    ombro = ponto(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    quadril = ponto(mp_pose.PoseLandmark.LEFT_HIP.value)
    joelho = ponto(mp_pose.PoseLandmark.LEFT_KNEE.value)
    orelha = ponto(mp_pose.PoseLandmark.LEFT_EAR.value)

    ang_tronco = calcular_angulo(ombro, quadril, joelho)
    ang_pescoco = calcular_angulo(orelha, ombro, quadril)

    # Desenhar landmarks
    desenhar_landmarks(imagem, results.pose_landmarks)
    cv2.imwrite(save_out, imagem)

    return {"tronco": ang_tronco, "pescoco": ang_pescoco}, save_out

# Comparar posturas e gerar PDF
def comparar_posturas(pdf_out="relatorio_postura.pdf"):
    img_antes = "antes.jpg"
    img_depois = "depois.jpg"

    medidas_antes, img_a_out = extrair_medidas(img_antes, "antes_out.jpg")
    medidas_depois, img_d_out = extrair_medidas(img_depois, "depois_out.jpg")

    if not medidas_antes or not medidas_depois:
        print("❌ Não foi possível detectar a postura em alguma das imagens.")
        return

    feedback = []
    if medidas_depois['tronco'] > medidas_antes['tronco']:
        feedback.append("✅ O tronco está mais ereto.")
    else:
        feedback.append("⚠️ O tronco não melhorou.")

    if medidas_depois['pescoco'] < medidas_antes['pescoco']:
        feedback.append("✅ A cabeça está mais alinhada.")
    else:
        feedback.append("⚠️ O pescoço continua inclinado.")

    # Criar PDF
    doc = SimpleDocTemplate(pdf_out)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Relatório de Análise de Postura", styles['Title']))
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Antes -> Tronco: {medidas_antes['tronco']:.1f}° | Pescoço: {medidas_antes['pescoco']:.1f}°", styles['Normal']))
    story.append(Paragraph(f"Depois -> Tronco: {medidas_depois['tronco']:.1f}° | Pescoço: {medidas_depois['pescoco']:.1f}°", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("📋 Feedback de melhoria:", styles['Heading2']))
    for f in feedback:
        story.append(Paragraph(f, styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("📸 Imagem Antes", styles['Heading2']))
    story.append(Image(img_a_out, width=300, height=300))
    story.append(Spacer(1, 20))

    story.append(Paragraph("📸 Imagem Depois", styles['Heading2']))
    story.append(Image(img_d_out, width=300, height=300))

    doc.build(story)
    print(f"✅ Relatório gerado: {pdf_out}")

# Executa
comparar_posturas()
