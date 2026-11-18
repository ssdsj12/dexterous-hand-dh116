"""Bounded zero-training control loop for an already reset physical episode.

Observers borrow owned endpoint data and must not advance SDK or rod clocks.
This module does not launch Isaac, construct a scene, or grant native authority.
"""
from copy import deepcopy


def run_preview(episode, journal, observer, *, actions, read_clock=None, publish_initial=True, on_terminal=None, before_control=None):
    """Run bounded controls, publishing reset and completed control boundaries."""
    try:
        def observe(pre):
            before = deepcopy(read_clock()) if read_clock is not None else None
            observer(pre)
            if read_clock is not None and read_clock() != before:
                raise RuntimeError("preview observer advanced or changed the physical clock")

        pre = episode.pre_action_state()
        if publish_initial:
            observe(pre)
        count = 0
        terminal = None
        for action in actions:
            journal.begin(pre, action)
            if before_control is not None:
                before_control(pre, action)
            result = episode.control(action)
            journal.end(result)
            count += 1
            terminal = result["termination_reason"]
            # A terminal episode is closed and cannot provide another pre-action state.
            if terminal is not None:
                if on_terminal is not None:
                    before = deepcopy(read_clock()) if read_clock is not None else None
                    on_terminal(result)
                    if read_clock is not None and read_clock() != before:
                        raise RuntimeError("preview observer advanced or changed the physical clock")
                break
            pre = episode.pre_action_state()
            observe(pre)
        return dict(control_intervals=count, termination_reason=terminal, ppo_updates=0)
    finally:
        episode = journal = observer = actions = read_clock = on_terminal = before_control = observe = None



def run_owned_scene(sim, owners, trace, *, builder, reader_factory, backend,
                    journal, observer, build_kwargs, device, actions):
    """Use the existing fresh dual-hand builder and same-rod endpoint reader.

    The caller owns the App, singleton lock, lifecycle, backend binding and
    independent execution budget. The reader sees every actual physical row,
    including rows within a control interval. Failures do not roll back physics.
    """
    if "record" in build_kwargs:
        raise ValueError("preview owns the physical record callback")
    state = {"reset": [], "rod": None, "reader": None, "packet": None, "generation": 0, "published": 0}

    def record(row):
        # Write the physical exchange before readback; physical rows flush but are not fsynced here.
        journal.record(row)
        if row["kind"] == "reset":
            state["reset"].append(deepcopy(row["reset"]["rod_reset"]))
        elif row["kind"] == "physical":
            if row["exchange"]["policy_step"] != state["policy_tick"]:
                raise ValueError("preview policy tick differs from physical exchange")
            rod = state["rod"]
            state["packet"] = state["reader"].capture(
                row["exchange"]["rod"], rope=rod.rope, backend=backend,
                sim=sim, read_clock=rod.clock)
            state["exchange"] = deepcopy(row["exchange"])
            state["exchange"]["material_director"] = deepcopy(state["packet"])
            state["exchange"]["preview_phase"] = "physical"
            state["generation"] += 1

    try:
        builder(sim, owners, record=record, **build_kwargs)
        rod = state["rod"] = owners.resources["runtime"]
        episode = owners.resources["dual_episode"]
        if len(state["reset"]) != 1:
            raise ValueError("one actual rod reset record required")
        reader = owners.keep("material_director_reader", reader_factory(rod._base.nodes, device=device))
        state["reader"] = reader
        trace.watch("material_director_reader", reader, owner="preview endpoint readback",
                    release="owned scope after synchronize")
        reset = state["reset"][0]
        state["packet"] = reader.capture_reset(
            reset, material_reference_uid=build_kwargs["episode_uid"] + ":material-reset-" + str(reset["reset_id"]),
            rope=rod.rope, backend=backend, sim=sim, read_clock=rod.clock)
        state["exchange"] = dict(rod=deepcopy(reset), material_director=deepcopy(state["packet"]), preview_phase="reset")
        state["start"] = rod.clock()["physics_step"]
        state["policy_tick"] = 0
        state["action_index"] = -1

        def clock():
            return (deepcopy(rod.clock()), sim.current_time_step_index, float(sim.current_time))

        def begin(pre, action):
            state["start"] = rod.clock()["physics_step"]
            state["action_index"] += 1
            state["policy_tick"] = state["action_index"]

        def publish(pre):
            exchange = deepcopy(state["exchange"])
            exchange["preview_physics_start"] = state["start"]
            exchange["preview_policy_tick"] = state["policy_tick"]
            observer(pre, exchange)
            state["published"] = state["generation"]

        def terminal(result):
            if state["generation"] > state["published"]:
                exchange = deepcopy(state["exchange"])
                exchange["preview_terminal_reason"] = result["termination_reason"]
                exchange["preview_physics_start"] = state["start"]
                exchange["preview_policy_tick"] = state["policy_tick"]
                observer(None, exchange)
                state["published"] = state["generation"]

        result = run_preview(episode, journal, publish, actions=actions, read_clock=clock,
                             publish_initial=True, on_terminal=terminal, before_control=begin)
        result["physics_rows"] = state["generation"]
        return result
    finally:
        try:
            episode = owners.resources.get("dual_episode")
            if episode is not None and getattr(episode, "_record", None) is record:
                episode._record = journal.record
        finally:
            state.clear()
            # A debugger/SDK may retain completed frames. Drop only our own
            # aliases here; never materialize or clear somebody else's f_locals.
            sim = owners = trace = builder = reader_factory = backend = journal = observer = actions = None
            rod = reader = episode = clock = begin = publish = terminal = record = None
