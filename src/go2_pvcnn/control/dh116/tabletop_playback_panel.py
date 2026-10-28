"""Small Isaac window for interactive policy inference, never training control."""


class PlaybackPanel:
    def __init__(self):
        import omni.ui as ui
        self.paused = False
        self.replay_requested = False
        self.window = ui.Window('Playback Controls', width=360, height=180)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label('DH116 PPO | Close - Hold - Lift', height=24)
                self.status = ui.Label('Running', height=24)
                with ui.HStack(spacing=8, height=38):
                    self.pause_button = ui.Button('Pause', clicked_fn=self.toggle_pause)
                    ui.Button('Replay', clicked_fn=self.request_replay)
                ui.Label('Replay resets the scene and reruns this policy.', word_wrap=True)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.text = 'Resume' if self.paused else 'Pause'

    def request_replay(self):
        self.replay_requested = True

    def take_replay(self):
        requested = self.replay_requested
        self.replay_requested = False
        return requested

    def update(self, phase, steps):
        self.status.text = f'{"Paused" if self.paused else "Running"} | {phase} | step {steps}'

    def close(self):
        self.window.destroy()
