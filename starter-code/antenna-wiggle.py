from reachy_mini import ReachyMini

# Connect to the running daemon
with ReachyMini() as cutie:
    print("Connected to Reachy Mini! ")
    
    # Wiggle antennas
    print("Wiggling antennas...")
    cutie.goto_target(antennas=[0.5, -0.5], duration=1.0)
    cutie.goto_target(antennas=[-0.5, 0.5], duration=1.0)
    cutie.goto_target(antennas=[0, 0], duration=1.0)

    print("Done!")
