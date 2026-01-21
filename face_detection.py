"""Demonstrate how to make Reachy Mini look at a point in an image.

When you click on the image, Reachy Mini will look at the point you clicked on.
It uses OpenCV to capture video from a camera and display it, and Reachy Mini's
look_at_image method to make the robot look at the specified point.

Note: The daemon must be running before executing this script.o

Original: https://github.com/pollen-robotics/reachy_mini/blob/develop/examples/look_at_image.py
"""

import argparse

import cv2

import numpy as np

from reachy_mini import ReachyMini


def click(event, x, y, flags, param):
    """Handle mouse click events to get the coordinates of the click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["just_clicked"] = True
        param["x"] = x
        param["y"] = y

def change_brightness(input_frame):
    brightness = 40   # try 20–60
    contrast = 1.2   # 1.0 = no change

    return cv2.convertScaleAbs(input_frame, alpha=contrast, beta=brightness)

def detect_face(input_frame):
    cv2.imshow("Reachy Mini Camera", input_frame)
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much the image size is reduced at each image scale
        minNeighbors=5,    # higher → fewer detections but better quality
        minSize=(30, 30)   # ignore really small faces
    )

    print("Faces detected",faces)

    # 3) Draw rectangles around faces on the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 4) Show the result
    cv2.imshow("Reachy Mini Camera", input_frame)


def main(backend: str) -> None:
    """Show the camera feed from Reachy Mini and make it look at clicked points."""
    state = {"x": 0, "y": 0, "just_clicked": False}

    cv2.namedWindow("Reachy Mini Camera")
    cv2.setMouseCallback("Reachy Mini Camera", click, param=state)

    print("Click on the image to make ReachyMini draw rectangles on faces.")
    print("Press 'q' to quit the camera feed.")
    with ReachyMini(media_backend=backend) as reachy_mini:
        try:
            while True:
                frame = reachy_mini.media.get_frame()

                if frame is None:
                    print("Failed to grab frame.")
                    continue

                #cv2.imshow("Reachy Mini Camera", change_brightness(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting...")
                    break

                if state["just_clicked"]:
                    print("Clicked")
                    detect_face(frame)
                    state["just_clicked"] = False
        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display Reachy Mini's camera feed and make it look at clicked points."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer", "webrtc"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)