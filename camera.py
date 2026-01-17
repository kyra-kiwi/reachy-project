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

    # WARM-UP LOOP
    print("Warming up camera...")
    for i in range(30):  # 30 frames = ~1 second
        cap.read()
    print("Camera ready!")

   # Read ONE frame PROPERLY
    ret, frame = cap.read()

    if not ret:
        print("Did not work")

    else:
        # 1) Make grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2) Run face detection on gray if you want
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 3) Show COLOUR frame
        cv2.imshow("Webcam (color)", frame)

        # 4) Save the COLOUR frame
        print("Frame shape:", frame.shape)
        print("Frame dtype:", frame.dtype)

        success = cv2.imwrite("capture_color.png", frame)
        if success:
            print("✅ Saved capture_color.png")
        else:
            print("❌ Failed to save capture_color.png")

    # Check current directory and list files
    import os
    print("Current directory:", os.getcwd())
    print("Files here:", os.listdir("."))


    # 5) Wait so you can see the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

else:
   print("Could not open any webcam!")