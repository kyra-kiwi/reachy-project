"""This code is used to detect new faces in a frame, draw rectangles around them
 and play a sound when a new face is detected.

It uses OpenCV to capture frames from the Reachy's camera and detect faces.

Note: The daemon must be running before running this code. If you don't know how
to start the daemon, look at the README.md file. The daemon is part of the Reachy Mini Control App.

Original references: 
1) https://github.com/pollen-robotics/reachy_mini/blob/develop/examples/look_at_image.py
2) https://github.com/pollen-robotics/reachy_mini/blob/develop/examples/debug/sound_play.py
"""

import argparse
import cv2
from reachy_mini import ReachyMini
import time
import numpy as np
import soundfile as sf
import os
import scipy
import logging

INPUT_FILE = os.path.join("./assets", "wake_up.wav")

def play_sound(mini, audio_file, backend: str):
    """Play a wav file by pushing samples to the audio device."""

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    data, samplerate_in = sf.read(INPUT_FILE, dtype="float32")

    if samplerate_in != mini.media.get_output_audio_samplerate():
        data = scipy.signal.resample(
            data,
            int(
                len(data)
                * (mini.media.get_output_audio_samplerate() / samplerate_in)
            ),
        )
    if data.ndim > 1:  # convert to mono
        data = np.mean(data, axis=1)

    mini.media.start_playing()
    print("Playing audio...")
    # Push samples in chunks
    chunk_size = 1024
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        mini.media.push_audio_sample(chunk)

    time.sleep(1)  # wait a bit to ensure all samples are played
    mini.media.stop_playing()
    print("Playback finished.")

def change_brightness(input_frame):
    brightness = 40   # try 20–60
    contrast = 1.2   # 1.0 = no change

    return cv2.convertScaleAbs(input_frame, alpha=contrast, beta=brightness)

def detect_face(input_frame, face_cascade):
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much the image size is reduced at each image scale
        minNeighbors=5,    # higher → fewer detections but better quality
        minSize=(30, 30)   # ignore really small faces
    )

    print("Faces detected",faces)

    #Draw rectangles around faces on the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Show the result
    cv2.imshow("Reachy Mini Camera", input_frame)

    return faces


def main(backend: str) -> None:
    
    cv2.namedWindow("Reachy Mini Camera")

    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

    face_detected_prev = False  # Track previous state

    with ReachyMini(media_backend=backend) as reachy_mini:
        try:
            while True:
                frame = reachy_mini.media.get_frame()

                if frame is None:
                    print("Failed to grab frame.")
                    continue

                bright = change_brightness(frame)

                faces =detect_face(bright, face_cascade)

                face_detected = len(faces) > 0
                if face_detected and not face_detected_prev:
                    play_sound(reachy_mini, INPUT_FILE, backend)
                face_detected_prev = face_detected

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting...")
                    break

        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Detect faces in a frame and draw rectangles around them."
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