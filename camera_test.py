"""Display a live camera feed without loading YOLO."""
import cv2

# Camera numbering varies by computer. Try 0 if 1 is unavailable.
camera = cv2.VideoCapture(1)
print("Camera opened:", camera.isOpened())

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read an image. Check camera permissions and index.")
            break
        cv2.imshow("Camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
