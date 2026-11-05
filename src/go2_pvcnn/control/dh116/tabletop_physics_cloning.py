"""Choose the physics initialization path without changing authored scene data."""


def clone_physics_environments(cloner, sim, paths, mode='replicate'):
    """Clone before ``sim.reset()`` and preserve cross-environment isolation.

    ``replicate`` uses PhysX replication with environment IDs. ``usd_parse``
    leaves parsing to PhysX and explicitly authors collision groups, following
    IsaacLab InteractiveScene's fallback when replication IDs are unavailable.
    Both paths retain USD inheritance; geometry and material/solver attributes
    are untouched. The caller receives the original cloner return object.
    """
    if mode not in ('replicate', 'usd_parse'):
        raise ValueError(f'unknown physics cloning mode: {mode!r}')
    if not paths:
        raise ValueError('physics cloning requires at least one environment path')
    replicate = mode == 'replicate'
    result = cloner.clone(source_prim_path=paths[0], prim_paths=paths,
        replicate_physics=replicate, copy_from_source=False, enable_env_ids=replicate)
    if not replicate:
        cloner.filter_collisions(sim.get_physics_context().prim_path,
            '/World/collisions', paths, global_paths=[])
    return result
