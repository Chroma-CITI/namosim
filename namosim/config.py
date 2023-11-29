import os

DISPLAY_WINDOW = os.environ.get("NAMO_NO_DISPLAY_WINDOW", "") == ""
THINK_IN_PARALLEL = os.environ.get("NAMO_NO_THINK_PARALLEL", "") == ""
