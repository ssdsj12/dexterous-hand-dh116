from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

class PreviewLoopTests(unittest.TestCase):
    def test_publishes_reset_and_each_control_after_journal_end(self):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_preview
        events = []
        class Episode:
            def pre_action_state(self):
                return {"clock": len([x for x in events if x == "control"])}
            def control(self, action):
                events.append("control")
                return {"termination_reason": None}
        class Journal:
            def begin(self, pre, action): events.append("begin")
            def end(self, result): events.append("end")
        def observer(pre):
            events.append(("observe", pre["clock"]))
        result = run_preview(Episode(), Journal(), observer, actions=[0, 0])
        self.assertEqual(events, [("observe", 0), "begin", "control", "end",
                                  ("observe", 1), "begin", "control", "end",
                                  ("observe", 2)])
        self.assertEqual(result["ppo_updates"], 0)

    def test_stops_on_terminal_before_another_action(self):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_preview
        calls = []
        class Episode:
            def pre_action_state(self): return {}
            def control(self, action):
                calls.append(action)
                return {"termination_reason": "guard"}
        class Journal:
            def begin(self, pre, action): pass
            def end(self, result): pass
        result = run_preview(Episode(), Journal(), lambda pre: None, actions=[0, 1])
        self.assertEqual(calls, [0])
        self.assertEqual(result["termination_reason"], "guard")

    def test_observer_cannot_advance_clock(self):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_preview
        clock = [0]
        class Episode:
            def pre_action_state(self): return {}
        def observer(pre): clock[0] += 1
        with self.assertRaisesRegex(RuntimeError, "physical clock"):
            run_preview(Episode(), None, observer, actions=[], read_clock=lambda: clock[0])



class OwnedSceneTests(unittest.TestCase):
    def test_same_rod_reader_captures_every_physical_row_before_presentation(self):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_owned_scene
        events = []
        ticks = []
        clock = {"step": 0}
        class Rod:
            rope = object()
            _base = type("Base", (), {"nodes": "seed"})()
            def clock(self): return dict(clock, physics_step=clock["step"])
        rod = Rod()
        class Episode:
            def pre_action_state(self): return {"step": clock["step"]}
            def control(self, action):
                for _ in range(2):
                    clock["step"] += 1
                    record({"kind": "physical", "exchange": {"rod": dict(clock), "policy_step": action}})
                return {"termination_reason": None}
        class Owners:
            resources = {"runtime": rod, "dual_episode": Episode()}
            def keep(self, name, value): self.resources[name] = value; return value
        class Trace:
            def watch(self, *args, **kwargs): events.append("watch")
        class Journal:
            def record(self, row): events.append(("journal", row["kind"]))
            def begin(self, pre, action): events.append("begin")
            def end(self, result): events.append("end")
        class Reader:
            def capture_reset(self, row, **kwargs):
                events.append("reset_capture"); return dict(step=0)
            def capture(self, row, **kwargs):
                events.append(("capture", row["step"])); return dict(row)
        def builder(sim, owners, **kwargs):
            nonlocal record
            record = kwargs["record"]
            owners.resources["dual_episode"]._record = record
            record({"kind": "reset", "reset": {"rod_reset": {"reset_id": 1, "step": 0}}})
        record = None
        class Sim:
            current_time_step_index = 0
            current_time = 0.0
        owners = Owners()
        def observe(pre, exchange):
            ticks.append((exchange["preview_phase"], exchange["preview_policy_tick"]))
            events.append(("observer", exchange["rod"]["step"]))
        result = run_owned_scene(Sim(), owners, Trace(), builder=builder,
            reader_factory=lambda nodes, *, device: Reader(), backend=object(),
            journal=Journal(), observer=observe,
            build_kwargs={"episode_uid": "test"}, device="cuda:0", actions=[0])
        self.assertEqual(events, [("journal", "reset"), "watch", "reset_capture",
                                  ("observer", 0), "begin", ("journal", "physical"), ("capture", 1),
                                  ("journal", "physical"), ("capture", 2), "end", ("observer", 2)])
        self.assertEqual(result["ppo_updates"], 0)
        self.assertIsNot(owners.resources["dual_episode"]._record, record)
        self.assertEqual(ticks, [("reset", 0), ("physical", 0)])
        self.assertEqual(result["physics_rows"], 2)



    def test_terminal_with_new_physical_row_is_published_once(self):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_preview
        observations = []
        class Episode:
            def pre_action_state(self): return {}
            def control(self, action): return {"termination_reason": "guard"}
        class Journal:
            def begin(self, pre, action): pass
            def end(self, result): pass
        run_preview(Episode(), Journal(), observations.append, actions=[0],
                    publish_initial=False, on_terminal=lambda result: observations.append(result["termination_reason"]))
        self.assertEqual(observations, ["guard"])


    def _terminal_case(self, *, terminal_has_row):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_owned_scene
        events, clock = [], {"step": 0}
        class Sim:
            current_time_step_index = 0
            current_time = 0.0
        class Rod:
            rope = object()
            _base = type("Base", (), {"nodes": "seed"})()
            def clock(self): return dict(clock, physics_step=clock["step"])
        rod = Rod()
        class Episode:
            def pre_action_state(self): return {}
            def control(self, action):
                if action == 0 or terminal_has_row:
                    clock["step"] += 1
                    self._record({"kind": "physical", "exchange": {"rod": dict(clock), "policy_step": action}})
                return {"termination_reason": "guard" if action else None}
        episode = Episode()
        class Owners:
            resources = {"runtime": rod, "dual_episode": episode}
            def keep(self, name, value): self.resources[name] = value; return value
        class Trace:
            def watch(self, *a, **k): pass
        class Journal:
            def record(self, row): pass
            def begin(self, pre, action): pass
            def end(self, result): pass
        class Reader:
            def capture_reset(self, row, **kw): return {}
            def capture(self, row, **kw): return dict(row)
        def builder(sim, owners, **kw):
            episode._record = kw["record"]
            episode._record({"kind": "reset", "reset": {"rod_reset": {"reset_id": 1, "step": 0}}})
        run_owned_scene(Sim(), Owners(), Trace(), builder=builder, reader_factory=lambda n, *, device: Reader(),
                        backend=object(), journal=Journal(),
                        observer=lambda pre, row: events.append((row["rod"]["step"], row.get("preview_terminal_reason"))),
                        build_kwargs={"episode_uid": "test"}, device="cuda:0", actions=[0,1])
        return events

    def test_terminal_row_published_with_reason(self):
        self.assertEqual(self._terminal_case(terminal_has_row=True), [(0,None),(1,None),(2,"guard")])

    def test_terminal_without_row_never_reuses_previous_frame(self):
        self.assertEqual(self._terminal_case(terminal_has_row=False), [(0,None),(1,None)])

class BoundaryIdentityTests(unittest.TestCase):
    def test_before_control_receives_pre_action_boundary(self):
        from go2_pvcnn.control.dh116.tabletop_physical_preview import run_preview
        events=[]
        class Episode:
            def pre_action_state(self): return {"step":2}
            def control(self, action): events.append("control"); return {"termination_reason":None}
        class Journal:
            def begin(self, pre, action): events.append("journal_begin")
            def end(self, result): pass
        run_preview(Episode(),Journal(),lambda pre:None,actions=[0],publish_initial=False,
                    before_control=lambda pre,action:events.append(("before",pre["step"])))
        self.assertEqual(events,["journal_begin",("before",2),"control"])
