"""Runtime collision filtering for the DH116 hand-only grasp probe."""
from __future__ import annotations

from pxr import Usd, UsdPhysics


# DH116 distal pad bodies remain collidable: thumb14 and finger23/33/43/53.
# The intermediate phalanges stay visible and articulated but do not create
# broad mesh contacts with the cable in this simulation-only probe.
NONPAD_LINKS = (
    "finger11_link", "finger12_link", "finger13_link",
    "finger21_link", "finger22_link",
    "finger31_link", "finger32_link",
    "finger41_link", "finger42_link",
    "finger51_link", "finger52_link",
)


def disable_nonpad_collisions(stage: Usd.Stage, hand_root: str) -> tuple[str, ...]:
    """Disable collision shapes under non-pad hand links on this in-memory stage.

    Source USDs are not modified.  Collision references are de-instanced in
    this stage only so the per-link ``collisionEnabled`` overrides are
    authorable.  Missing links are skipped to support hand asset variants.
    """
    root = stage.GetPrimAtPath(hand_root)
    if not root or not root.IsValid():
        raise ValueError(f"hand root does not exist: {hand_root}")
    disabled = []
    for link_name in NONPAD_LINKS:
        branch = stage.GetPrimAtPath(f"{hand_root}/{link_name}/collisions")
        if not branch or not branch.IsValid():
            continue
        if branch.IsInstance():
            branch.SetInstanceable(False)
        for prim in Usd.PrimRange(branch, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(prim)
                if api.GetCollisionEnabledAttr().Get():
                    api.GetCollisionEnabledAttr().Set(False)
                    disabled.append(str(prim.GetPath()))
    return tuple(sorted(disabled))


__all__ = ["NONPAD_LINKS", "disable_nonpad_collisions"]
