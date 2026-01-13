import cv2
import reachy_mini

  # Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam, trying camera 1...")
    cap = cv2.VideoCapture(1)

if cap.isOpened():
   print("Webcam initialized successfully!")
    # Set lower resolution for faster processing
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   # Reduce buffer size to minimize lag (get most recent frame)
   cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

   # Grab and discard buffered frames to get the most recent one
   cap.grab()
   ret, frame = cap.retrieve()
    
   if not ret:
        print('Did not work')
        exit

# Convert to grayscale for face detection
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Display
   cv2.imshow("Grayscale Image", gray)
   cv2.waitKey(500)
   cv2.destroyAllWindows()

else:
   print("Could not open any webcam!")