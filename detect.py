import cv2
from ultralytics import YOLO
from pathlib import Path

model_path = Path(__file__).parent / "yolo11s.pt"
model = YOLO(model_path)
camera = cv2.VideoCapture(1)
print("Camera opened:", camera.isOpened())


while True:
    success, frame = camera.read()

    if not success:
        print("Could not read an image.")
        break

    results = model.track(
        frame,
        conf=0.1,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
