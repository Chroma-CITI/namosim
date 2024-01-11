class Input:
    def __init__(self):
        self.key_pressed: str | None = None

    def update_key_pressed(self, key: str | None):
        self.key_pressed = key

    def clear(self):
        self.key_pressed = None
