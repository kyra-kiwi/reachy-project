# reachy-project
- [reachy-project](#reachy-project)
  - [Getting the first sample program running](#getting-the-first-sample-program-running)
  - [Getting the Reachy Mini to look at a point in a frame](#getting-the-reachy-mini-to-look-at-a-point-in-a-frame)
  - [Detect face and play sound](#detect-face-and-play-sound)
  - [Installing Ollama and running models](#installing-ollama-and-running-models)
  - [Useful commands](#useful-commands)
  - [References](#references)


If you want to follow the plan of action and timeline for my project, go to [plan-timeline.md](plan-timeline.md)

## Getting the first sample program running
First of all, install the [Reachy Mini Control App](https://github.com/pollen-robotics/reachy-mini-desktop-app)

I kept the Reachy Mini Control App running in the background and turned the robot on. It was connected to my mac using a USBC cable. I already had `uv` installed in my terminal. To activate the reachy environment, I wrote:

```sh
source reachy_mini_env/bin/activate
```

and got this response: 

```sh
Kyra@mac reachy-project % source reachy_mini_env/bin/activate
(reachy_mini_env) Kyra@mac reachy-project % 
```

I got the sample program from here https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/quickstart.md and ran the following command because I needed the `reachy_mini` library:

```sh
uv pip install "reachy-mini"
```

I ran the python command in the terminal. This is the response I got:

```sh
(reachy_mini_env) Kyra@mac reachy-project % python antenna-wiggle.py
Connected to Reachy Mini! 
Wiggling antennas...
Done!
``` 

## Getting the Reachy Mini to look at a point in a frame

Make sure your environment is active:
```sh
source reachy_mini_env/bin/activate
```

To run the python script, use the command below:

```sh
(reachy_mini_env) Kyra@mac reachy-project % python look-at-point-in-image.py   
Click on the image to make ReachyMini look at that point.
Press 'q' to quit the camera feed.
Exiting...
```

This will open a new window on your laptop showing a live image of what the robot camera sees. Note that most of the code in this program ([look-at-point-in-image.py](look-at-point-in-image.py)) is partly from an example and not written by me.


## Detect face and play sound

This program [face_detection.py](face_detection.py) recognises and draws rectangles around faces. When a new face is detected, the Reachy Mini will play a sound.


## Installing Ollama and running models

To install ollama, I ran the following command:
```sh
brew install ollama

brew services start ollama
```

Now, to pull and run the phi3 model from Microsoft, use the following compands:
```sh
ollama pull phi3:mini

ollama run phi3:mini
```


Once the AI model was running, this is how I said hello to the phi3:mini and its response:
```sh
Kyra@mac ~ % ollama run phi3:mini      
⠸ >>> hello!
Hello there! How can I assist you today?
```

## Useful commands

There were a lot of python files, so I moved the old ones to ([starter-code](starter-code)). This is the command I used, taking ([antenna-wiggle.py](antenna-wiggle.py)) as an example:
```sh
git mv antenna-wiggle.py starter-code/
```

## References

- Official Standard Development Kit (SDK): https://github.com/pollen-robotics/reachy_mini?tab=readme-ov-file
- Python SDK documentation: https://github.com/pollen-robotics/reachy_mini/blob/50923d19d12c13a66baff86cf29ac088d90b07db/docs/SDK/python-sdk.md
- https://github.com/pollen-robotics/reachy_mini_conversation_app
- Mirror test: https://youtu.be/gw2DShGlreQ?si=pziOKiugGwmuNSxI
- https://realpython.com/defining-your-own-python-function/
- https://github.com/dwain-barnes/reachy_mini_conversation_app_local
- https://github.com/pollen-robotics/reachy_mini/blob/develop/examples/debug/sound_play.py
