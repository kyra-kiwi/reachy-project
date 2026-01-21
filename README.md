# reachy-project
- [reachy-project](#reachy-project)
  - [Plan of action](#plan-of-action)
  - [Timeline](#timeline)
  - [Getting a sample program running](#getting-a-sample-program-running)
  - [References](#references)


## Plan of action
- Download app: https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md
- Watch assembly video: https://www.youtube.com/watch?v=PC5Yx950nMY&t=1s
- Assemble robot
- Read and understand documentation from website (example projects)

## Timeline
- 31/12/2025: Created ReadMe
- 01/01/2026: Added tasks on Github and found references
- 03/01/2026: Assembled Reachy and got [Reachy Mini Control app](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini#/download) working
- 04/01/2026: Played with Reachy Mini Control App and ran example program (antenna wiggle) in Visual Studio Code
- 05/01/2026: Got head movement working and made hello2.py (slightly different version)
- 13/01/2026: Got webcam working in grayscale + head movement program
- 17/01/2026: 'Get frame' and 'Get video' in colour programs
- 20/01/2026: 'Look at image' program + brightness control

## Getting a sample program running
First of all, I kept the Reachy Mini Control App running in the background and turned the robot on. It was connected to my mac using a USBC cable. I already had `uv` installed in my terminal. To activate the reachy environment, I wrote:

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
uv pip install "reachy-mini"`
```

I ran the python command in the terminal. This is the response I got:

```sh
(reachy_mini_env) Kyra@mac reachy-project % python hello.py
Connected to Reachy Mini! 
Wiggling antennas...
Done!
``` 

## References

- Official Standard Development Kit (SDK): https://github.com/pollen-robotics/reachy_mini?tab=readme-ov-file
- https://github.com/pollen-robotics/reachy_mini_conversation_app
- Mirror test: https://youtu.be/gw2DShGlreQ?si=pziOKiugGwmuNSxI