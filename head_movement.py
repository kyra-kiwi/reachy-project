
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # ... inside with ReachyMini() as mini:
    mini.goto_target(head=create_head_pose(yaw=-10, pitch=20))