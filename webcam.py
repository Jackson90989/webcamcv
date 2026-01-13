from ultralytics import YOLO
# python -m pip install ultralytics opencv-python
#pip install ultralytics opencv-python
# pip install ultralytics opencv-python



import cv2

model = YOLO(r"C:\Users\JacksonRodrigues\Downloads\Output\runs\detect\train9\weights\best.pt")
cap = cv2.VideoCapture(0)  # webcam



while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    results = model(frame, conf=0.1)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
