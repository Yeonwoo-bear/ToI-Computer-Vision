from ultralytics import YOLO
import cv2

# 1. 사전 학습된 YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

# 2. 테스트 이미지 경로
image_path = "data/images/car.jpeg"

# 3. 객체 탐지
results = model(image_path)

# 4. 탐지 결과 가져오기
result = results[0]

# 5. 탐지된 객체 정보 출력
print("\n===== 탐지 결과 =====")

for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])

    print(f"객체: {class_name} | Confidence: {confidence:.2f}")

# 6. Bounding Box가 그려진 이미지
detected_image = result.plot()

# 7. 화면에 출력
cv2.imshow("YOLO Multi Object Detection", detected_image)

cv2.waitKey(0)
cv2.destroyAllWindows()