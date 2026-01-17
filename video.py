import cv2
import reachy_mini
import time  # NEW: for timing

# Open webcam (same as before)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam, trying camera 1...")
    cap = cv2.VideoCapture(1)

if cap.isOpened():
    print("Webcam initialized successfully!")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 🔥 WARM UP CAMERA (same as before)
    print("Warming up camera...")
    for i in range(30):
        cap.read()
    print("Camera ready!")

    # NEW: Video recorder setup (like pressing "record")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video format
    video_writer = cv2.VideoWriter('reachy_video.mp4', fourcc, 20.0, (640, 480))

    # NEW: Record for 5 seconds
    RECORD_TIME = 5  # Change this number for different lengths
    start_time = time.time()
    
    print("🎥 Recording for", RECORD_TIME, "seconds... Press 'q' to stop early")
    
    while (time.time() - start_time) < RECORD_TIME:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write each frame to video file
        video_writer.write(frame)
        
        # Show live preview
        cv2.imshow("Recording...", frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Stop recording
    video_writer.release()
    cv2.destroyAllWindows()
    cap.release()
    
    print("✅ Saved reachy_video.mp4!")
else:
    print("Could not open any webcam!")
