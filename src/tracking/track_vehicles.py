
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    # 1. 실행 옵션 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--classes", default="vehicle")
    args = parser.parse_args()

    # 2. 웹캠 번호 또는 영상 파일 경로
    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    # 3. YOLO 모델 불러오기
    print("YOLO 모델 불러오는 중...")
    model = YOLO(args.model)

    # 기본 COCO 차량 클래스
    # 윤서 님의 모델이 별도 학습된 경우 all 사용
    vehicle_classes = [2, 3, 5, 7]

    classes = (
        vehicle_classes
        if args.classes == "vehicle"
        else None
    )

    # 4. 카메라 또는 영상 파일 열기
    if isinstance(source, int):
        cap = cv2.VideoCapture(
            source,
            cv2.CAP_AVFOUNDATION
        )
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("카메라 또는 영상 파일을 열 수 없습니다.")
        return

    print("카메라/영상 연결 성공!")
    print("추적 시작! 종료하려면 q를 누르세요.")

    # 5. 카메라 초기 프레임 실패 대비
    failed_frames = 0
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                failed_frames += 1

                # 영상 파일이 끝났으면 종료
                if not isinstance(source, int):
                    print("영상 재생 완료")
                    break

                # 카메라가 준비될 때까지 재시도
                if failed_frames >= 30:
                    print("프레임 읽기 실패: 종료합니다.")
                    break

                time.sleep(0.1)
                continue

            failed_frames = 0
            frame_number += 1

            # 6. YOLO 탐지 + ByteTrack 추적
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.1,
                classes=classes,
                verbose=False
            )

            result = results[0]

            # 7. Bounding Box와 Tracking ID 표시
            annotated = result.plot()

            # 8. 차량별 ID와 중심 좌표 계산
            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # 중심 좌표에 초록색 점 표시
                    cv2.circle(
                        annotated,
                        (center_x, center_y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                    print(
                        f"Frame: {frame_number} | "
                        f"ID: {track_id} | "
                        f"Center: ({center_x}, {center_y})"
                    )

            # 9. 화면에 프레임 번호 표시
            cv2.putText(
                annotated,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # 10. 실시간 화면 출력
            cv2.imshow(
                "Smart Parking - ByteTrack",
                annotated
            )

            # q를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("사용자 종료")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("추적 프로그램 종료")


if __name__ == "__main__":
    main()
