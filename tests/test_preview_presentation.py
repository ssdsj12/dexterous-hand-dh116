from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

class PresentationGlueTests(unittest.TestCase):
    def test_publish_before_capture_and_use_live_identity(self):
        from go2_pvcnn.control.dh116.tabletop_preview_presentation import PreviewPresentation
        events = []
        class Display:
            def publish_snapshot(self, snapshot, *, read_clock):
                events.append(("mesh", snapshot["identity"]["physics_step"]))
        class Camera:
            def capture(self):
                events.append(("rgb",)); return {"left_rgb":"left", "right_rgb":"right"}
        clock = {"physics_step": 10}
        presentation = PreviewPresentation(
            capture_snapshot=lambda row: {"identity": dict(clock)},
            display=Display(), camera=Camera(), read_clock=lambda: dict(clock),
            sink=lambda rgb, snapshot, terminal: events.append(("sink", snapshot["identity"]["physics_step"], terminal)))
        presentation(None, {"preview_terminal_reason": "guard"})
        self.assertEqual(events, [("mesh",10), ("rgb",), ("sink",10,"guard")])
    def test_clock_change_after_capture_fails_before_sink(self):
        from go2_pvcnn.control.dh116.tabletop_preview_presentation import PreviewPresentation
        clock=[0]; events=[]
        class Display:
            def publish_snapshot(self, snapshot, *, read_clock): pass
        class Camera:
            def capture(self): clock[0]+=1; return {}
        presenter=PreviewPresentation(capture_snapshot=lambda row:{},display=Display(),camera=Camera(),
                                     read_clock=lambda:clock[0],sink=lambda *args: events.append("sink"))
        with self.assertRaisesRegex(RuntimeError, "clock"):
            presenter(None,{})
        self.assertEqual(events,[])
