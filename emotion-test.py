from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
recorded_moves = RecordedMoves(EMOTIONS_DATASET)
move_name = "reprimand2"

with ReachyMini() as reachy_mini:
    move = recorded_moves.get(move_name)
    reachy_mini.play_move(move, initial_goto_duration=1.0)