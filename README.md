# reachy-project
- [reachy-project](#reachy-project)
  - [Getting a sample program running](#getting-a-sample-program-running)
  - [References](#references)

If you want to follow the plan of action and timeline for my project, go to [plan-timeline.md](plan-timeline.md)

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