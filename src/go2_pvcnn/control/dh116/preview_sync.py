"""Helpers for synchronizing static Isaac pose previews."""


def commit_static_pose(sim):
    """Flush a written kinematic pose to USD before pausing the timeline.

    IsaacLab only updates articulation link transforms from the physics view
    while the timeline is playing.  Rendering after pausing therefore leaves
    static previews at the asset's default pose even though the PhysX buffers
    contain the requested joints.
    """
    sim.play()
    sim.step(render=True)
    sim.pause()


def preview_simulation_cfg_kwargs():
    """Return the preview-only SimulationCfg overrides.

    Fabric remains enabled in the real physics runners.  The visible static
    replay uses USD-backed transforms so the reviewed pose is actually shown.
    """
    return {"use_fabric": False}
