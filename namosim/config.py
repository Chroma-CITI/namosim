import os

DISPLAY_WINDOW = os.environ.get("NAMO_DEACTIVATE_TKINTER", "") == ""
DEACTIVATE_RVIZ = (
    os.environ.get("NAMO_DEACTIVATE_RVIZ", "") != ""
    or os.environ.get("ROS_DISTRO", "") == ""
)
