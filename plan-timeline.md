## Plan of action
- [x] Download app: https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md
- [x] Watch assembly video: https://www.youtube.com/watch?v=PC5Yx950nMY&t=1s
- [x] Assemble robot
- [x] Read and understand documentation from website (example projects)
- [x] Basic antenna wiggle program: [antenna-wiggle.py](antenna-wiggle.py)
- [x] Head movement program: [head_movement.py](head_movement.py)
- [x] Camera/audio tasks before starting with robot
    - [x] 'Get frame' program from Mac webcam
    - [x] 'Get video' program from Mac webcam
    - [ ] Get and process audio from Mac mic -- won't do, already doing with Reachy
- [x] Reachy Mini Camera
    - [x] 'Get frame' program from Reachy Mini camera
    - [x] 'Get video' program from Reachy Mini camera
    - [x] 'Look at point in image' program using Reachy Mini
    - [x] Face detection program
- [x] Add greeting when face detected
- [x] Add ability to respond to speech
- [ ] Reach Mini Audio
    - [x] Record audio from Reachy Mini
    - [x] Convert audio to text
    - [x] Use LLM to process/respond to text
    - [ ] Convert response text to audio
    - [x] Play back audio on Reachy Mini


## Timeline
- 31/12/2025: Created ReadMe
- 01/01/2026: Added tasks on Github and found references
- 03/01/2026: Assembled Reachy and got [Reachy Mini Control app](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini#/download) working
- 04/01/2026: Played with Reachy Mini Control App and ran example program (antenna wiggle) in Visual Studio Code
- 05/01/2026: Got head movement working and made hello2.py (slightly different version)
- 13/01/2026: Got webcam working in grayscale + head movement program
- 17/01/2026: 'Get frame' and 'Get video' in colour programs
- 20/01/2026: 'Look at image' program + brightness control
- 21/01/2026: Updated tasks and readme + organised files + face detection program (rectangle drawing not working)
- 28/01/2026: Fixed face detection program so rectangles are drawn on click
- 01/02/2026: Made face detection continuous + started using Cursor
Cursor
- 04/02/2026: Updated readme + instructions for Cursor
- 06/02/2026: Got Phi:3 model working
- 08/02/2026: Added llama model (as well as Phi:3) + got Reachy sound playback working + reorganised files + updated readme + got Reachy recording working
- 13/02/2026: Got some audio detection working
- 14/02/2026: Added keys to change modes since video and audio can't work at the same time in a single while loop
- 15/02/2026: Got llama responses working for transcribed text + added coloured prints
