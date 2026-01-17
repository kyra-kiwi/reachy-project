import cv2
import reachy_mini

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam, trying camera 1...")
    cap = cv2.VideoCapture(1)

if cap.isOpened():
    print("Webcam initialized successfully!")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Read ONE frame (your current logic)
    cap.grab()
    ret, frame = cap.retrieve()

    if not ret:
        print("Did not work")
        
    else:
        # 1) Make grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2) Run your face detection on gray
        # Example (keep your own cascade + params):
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 3) Show COLOR frame, not gray
        cv2.imshow("Webcam (color)", frame)

        # Wait for a key so you can see the image and optionally save it
        key = cv2.waitKey(0) & 0xFF
        if key == ord("c"):
            cv2.imwrite("capture_color.png", frame)
            print("Saved capture_color.png")

        cv2.destroyAllWindows()

    cap.release()
else:
    print("Could not open any webcam!")
