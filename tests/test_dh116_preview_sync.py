from go2_pvcnn.control.dh116.preview_sync import (
    commit_static_pose,
    preview_simulation_cfg_kwargs,
)
from go2_pvcnn.control.dh116.visual_pose_sync import pose_quaternion_wxyz


class _FakeSimulation:
    def __init__(self):
        self.playing = False
        self.events = []

    def play(self):
        self.playing = True
        self.events.append("play")

    def forward(self):
        assert self.playing, "kinematic forwarding must happen while timeline is playing"
        self.events.append("forward")

    def step(self, render=True):
        assert self.playing, "the preview tick must run while timeline is playing"
        assert render is True
        self.events.append("step")

    def pause(self):
        self.playing = False
        self.events.append("pause")

    def render(self):
        self.events.append("render")


def test_static_pose_commits_visual_kinematics_before_pause():
    sim = _FakeSimulation()

    commit_static_pose(sim)

    assert sim.events == ["play", "step", "forward", "pause", "render"]


def test_static_pose_preview_disables_fabric_only_for_visual_sync():
    assert preview_simulation_cfg_kwargs() == {"use_fabric": False}


def test_visual_pose_sync_preserves_isaac_wxyz_quaternion_order():
    pose = [0.0, 0.0, 0.0, 0.9, 0.1, 0.2, 0.3]
    assert pose_quaternion_wxyz(pose) == (0.9, 0.1, 0.2, 0.3)
