#  1. Criando o Dataset no Roboflow
1. Acesse: https://roboflow.com
2. Crie um novo projeto:
   - Clique em "Create New Project"
   - Escolha "Object Detection"
   - Nomeie seu projeto (ex.: Peace Signs Detection)
3. Faça upload das imagens:
   - Adicione imagens contendo os objetos que deseja detectar
   - Use imagens variadas (ângulos, iluminação, fundos diferentes)
4. Anote as imagens (Labeling):
   - Desenhe caixas ao redor dos objetos
   - Nomeie cada classe (ex.: peace_sign)
5. Gere versões do dataset:
   - Clique em "Generate"
   - Escolha formato YOLOv8
   - Defina tamanho da imagem (ex.: 640x640)
   - Clique em "Download Dataset" → escolha YOLOv8

#  2. Estrutura do Dataset
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── data.yaml

# Exemplo do arquivo data.yaml:
train: path/to/train/images
val: path/to/valid/images
nc: 1  # número de classes
names: ['peace_sign']

#  3. Instalação das dependências
pip install ultralytics opencv-python

#  4. Treinando o Modelo YOLOv8
from ultralytics import YOLO

# Carrega modelo base
model = YOLO("yolov8n.pt")  # versão nano (leve)

# Treina com seu dataset
model.train(
    data=r"C:\Users\JacksonRodrigues\Downloads\Output\Find peace signs 3.v3-roboflow-instant-2--eval-.yolov8\data.yaml",
    epochs=60,
    imgsz=640
)

#  5. Detecção em Tempo Real com Webcam
import cv2
from ultralytics import YOLO

# Carrega modelo treinado
model = YOLO(r"C:\Users\JacksonRodrigues\Downloads\Output\runs\detect\train9\weights\best.pt")

# Captura da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a predição
    results = model(frame, conf=0.1)
    annotated_frame = results[0].plot()

    # Mostra na tela
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#  Dicas para Melhorar
- Use aumentação de dados no Roboflow (flip, brilho, blur)
- Tenha pelo menos 200-500 imagens para bons resultados
- Ajuste conf no código para evitar falsos positivos
- Teste diferentes modelos: yolov8n.pt (leve), yolov8s.pt (médio)
