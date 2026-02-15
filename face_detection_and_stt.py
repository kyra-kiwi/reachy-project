"""This code is used to detect new faces in a frame, draw rectangles around them
 and listen for a greeting when a new face is detected. 

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
import whisper
import ollama

logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpcore.http11").setLevel(logging.INFO)
logging.getLogger("httpcore.connection").setLevel(logging.INFO)
logging.getLogger("ollama").setLevel(logging.INFO)

INPUT_FILE = os.path.join("./assets", "wake_up.wav")

system_message = "You are a cute, friendly robot named Cleo. Keep your responses short and helpful with a maximum of 5 sentences. Be concise and conversational."
conversation_history = [{'role': 'system', 'content': system_message}] 

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

    #Draw rectangles around faces on the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Show the result
    cv2.imshow("Reachy Mini Camera", input_frame)

    return faces

def collect_audio_chunk(mini, duration_seconds, current_mode):
    """Collect audio samples for specified duration."""
    audio_samples = []
    t0 = time.time()
    mini.media.start_recording()
    print("Collecting audio chunk...")
    while time.time() - t0 < duration_seconds:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            mini.media.stop_recording()
            return None, current_mode
        key = cv2.waitKey(1) & 0xFF
                
        if key == ord("f"):
            current_mode = 'face'
        elif key == ord("v"):
            current_mode = 'voice'
        elif key == ord("x"):
            conversation_history = [{'role': 'system', 'content': system_message}]
            print("####################################################")
            print("Conversation context cleared (system prompt preserved)")
            print("####################################################")
        elif key == ord("q"):
            mini.media.stop_recording()
            # Default to face detection mode
            current_mode = 'face'
            return None, current_mode

        sample = mini.media.get_audio_sample()
        if sample is not None:
            audio_samples.append(sample)
        else:
            print("No audio data available yet...")
        time.sleep(0.1)  # Small delay to avoid busy waiting
    mini.media.stop_recording()
    if audio_samples:
        return np.concatenate(audio_samples, axis=0), current_mode
    return None, current_mode

def transcribe_audio_chunk_array(audio_data, samplerate, whisper_model):
    """
    Transcribe audio data using Whisper model by passing numpy array directly.
    
    Arguments:
        audio_data: numpy array of audio samples (from collect_audio_chunk)
        samplerate: sample rate of the audio (from mini.media.get_input_audio_samplerate())
        whisper_model: loaded Whisper model instance
    Returns:
        str: Transcribed text, or empty string if transcription fails
    """
    if audio_data is None or len(audio_data) == 0:
        print("No audio data to transcribe.")
        return ""
    
    try:
        print("Transcribing audio chunk...")
        # Ensure audio is mono (1D array) and float32 format
        if audio_data.ndim > 1:
            # Convert stereo/multi-channel to mono by averaging
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure float32 format (Whisper expects float32)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to [-1, 1] range if needed
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            # Normalize to [-1, 1] range
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Pass numpy array directly to Whisper
        # Whisper accepts numpy arrays and will handle resampling if needed
        result = whisper_model.transcribe(
            audio_data,
            fp16=False,  # Important for Mac M-series compatibility
            language="en"
        )
        
        transcribed_text = result["text"].strip()
        return transcribed_text
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def llama(transcribed_text, conversation_history):
    # print(f"LLAMA:{transcribed_text}")
    conversation_history.append({'role': 'user', 'content': transcribed_text})
    response = ollama.chat(model='llama3.2', 
        messages = conversation_history,
        options= {'keep_alive': -1} # Keep the connection alive
        )
    assistant_response = response['message']['content']
    conversation_history.append({'role': 'assistant', 'content': assistant_response})
    
    return assistant_response, conversation_history

def main(backend: str) -> None:
    global conversation_history
    whisper_model = whisper.load_model("base.en")
    cv2.namedWindow("Reachy Mini Camera")

    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

    face_detected_prev = False  # Track previous state
    current_mode = 'face'

    with ReachyMini(media_backend=backend) as reachy_mini:
        try:
            while True:
                 # Check for key presses at the start of each loop iteration
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    print("Exiting...")
                    break
                elif key == ord("f"):
                    # Switch to face detection mode
                    if current_mode != "face":
                        print("Switching to face detection mode...")
                        if current_mode == "voice":
                            reachy_mini.media.stop_recording()  # Stop audio recording
                        current_mode = "face"
                elif key == ord("v"):
                    # Switch to voice/listening mode
                    if current_mode != "voice":
                        print("Switching to voice/listening mode...")
                        reachy_mini.media.start_recording()  # Start audio recording
                        current_mode = "voice"
                
                elif key == ord("x"):
                    # Reset conversation history to system message only
                    conversation_history = [{'role': 'system', 'content': system_message}]
                    print("####################################################")
                    print("Conversation context cleared (system prompt preserved)")
                    print("####################################################")
                
                # Execute based on current mode
                if current_mode == "face":
                    # Face detection mode
                    frame = reachy_mini.media.get_frame()
                    if frame is None:
                        print("Failed to grab frame.")
                        continue

                    bright = change_brightness(frame)
                    faces = detect_face(bright, face_cascade)

                    face_detected = len(faces) > 0
                    if face_detected and not face_detected_prev:
                        print("Face detected!")
                        play_sound(reachy_mini, INPUT_FILE, backend)
                    #else:
                    #   print("No faces detected :(")
                    face_detected_prev = face_detected
                
                elif current_mode == "voice":
                    audio_chunk, current_mode = collect_audio_chunk (reachy_mini, 10, current_mode)
                    if audio_chunk is None:
                        # User pressed 'q' during collection
                        break
                    if audio_chunk is not None:
                        transcribed_text = transcribe_audio_chunk_array(audio_chunk, 
                                reachy_mini.media.get_input_audio_samplerate(), 
                                whisper_model)
                        print(transcribed_text)
                        llama_response, conversation_history = llama(transcribed_text, conversation_history)
                        
                        print(f"LLAMA response: {llama_response}")
                    else:
                        print("No audio chunk available...")
                        break


                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting...")
                    break

        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            if current_mode == "voice":
                reachy_mini.media.stop_recording()
            cv2.destroyAllWindows()
    print("Done!")

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