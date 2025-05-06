import os

DISPLAY_WINDOW = os.environ.get("NAMO_NO_DISPLAY_WINDOW", "") == ""
DEACTIVATE_RVIZ = os.environ.get("NAMO_DEACTIVATE_RVIZ", "") != ""
