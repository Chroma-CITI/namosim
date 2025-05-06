class Input:
    def __init__(self):
        self.key_pressed: str | None = None

    def handle_key_press(self, key: str | None):
        self.key_pressed = key

    def handle_key_release(self, key: str | None):
        if key == self.key_pressed:
            self.key_pressed = None

    def clear(self):
        self.key_pressed = None
