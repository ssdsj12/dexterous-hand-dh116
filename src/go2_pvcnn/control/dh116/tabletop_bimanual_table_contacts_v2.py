"""Static original table descendant discovery; no new contact capability.

The frozen v1 expects CollisionAPI on the Table root. Installed CuboidCfg authors
an Xform root and a /geometry/mesh collider. This successor resolves the real
USD hierarchy, retaining v1 identity/clock checks and its explicit static NA.
It never changes USD, creates contact views, enables readback, or steps physics.
"""
import hashlib
from pathlib import Path

from . import tabletop_bimanual_table_contacts as legacy

VERSION = "dh116-bimanual-static-table-descendants-v2"
LEGACY_SHA = "60579a928e7304842e80f9689f89b842bf710e4409ad9c712b68b21a4f6090a9"
SHAPES = Path("/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/spawners/shapes/shapes.py")
SHAPES_SHA = "053227b88000f63f707f6719d81b342bfa59c04ba78126e4e8b6f3760c7df4ca"


def resolve_static_table_colliders(stage, table_root_path):
    """Return owned strings/scalars for every enabled collider below the root.

    Every collider's complete ancestor chain is inspected. Any RigidBodyAPI is
    rejected, including a disabled body: this package only handles the original
    static table and cannot silently convert a different scene into static NA.
    """
    root = prim = ancestor = None
    try:
        from pxr import Usd, UsdPhysics
        root = stage.GetPrimAtPath(table_root_path)
        if not root or not root.IsActive():
            raise ValueError("actual active original table root required")
        colliders, disabled = [], []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                disabled.append(path)
                continue
            rigid = []
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid.append(str(ancestor.GetPath()))
                ancestor = ancestor.GetParent()
            if rigid:
                raise ValueError("successor requires original static table; rigid ancestors: "+repr(rigid))
            colliders.append(dict(path=path, rigid_body_ancestors=rigid))
        if not colliders:
            raise ValueError("actual enabled original table collider required")
        colliders.sort(key=lambda row: row["path"])
        return dict(table_root_path=str(root.GetPath()),
                    collider_paths=[row["path"] for row in colliders],
                    colliders=colliders, disabled_collider_paths=sorted(disabled),
                    table_is_static=True, live_physx_shape_query_performed=False)
    finally:
        stage = root = prim = ancestor = None


def open_bimanual_table_contacts(stage, hands, prepared, *, read_clock):
    """Same caller-owned source as v1; only the original static table is accepted.

    The existing v1 reader returns NA from the empty views before its root-only
    collision check. Metadata records the actual child paths, not filter support.
    No process-global monkeypatch or synthetic stage/hand proxy is used.
    """
    hand = view = None
    try:
        from .linear_coupling import JOINTS
        from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
        before = legacy._clock(read_clock)
        pins = legacy._source_pins()
        for path, sha in ((Path(legacy.__file__), LEGACY_SHA), (SHAPES, SHAPES_SHA)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("static table source implementation changed: "+str(path))
            pins[str(path)] = sha
        if len(hands) != 2 or hands[0] is hands[1] or hands[0].root_physx_view is hands[1].root_physx_view:
            raise ValueError("two distinct original mounted hands required")
        if hands[0]._physics_sim_view is not hands[1]._physics_sim_view:
            raise ValueError("hands must use one actual root simulation view")
        table = prepared["prefix"]+"/Table"
        resolved = resolve_static_table_colliders(stage, table)
        paths = {}
        for side, hand in zip(("left", "right"), hands):
            names = list(hand.body_names)
            view = hand.root_physx_view
            root = prepared["prefix"]+"/"+side+"Hand"
            expected = list(prepared["hands"][side]["source_colliders"])
            if (len(expected) != 17 or len(set(expected)) != 17 or len(names) != 23 or len(set(names)) != 23
                    or set(hand.joint_names) != set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
                    or len(hand.joint_names) != 17 or not set(expected) <= set(names)
                    or list(view.prim_paths) not in ([root], [root+"/wrist_fixed"])
                    or view.link_paths != [[root+"/"+name for name in names]]):
                raise ValueError("actual original 17DOF/17 body source mapping differs")
            paths[side] = [view.link_paths[0][names.index(name)] for name in expected]
        metadata = dict(version=VERSION, table_is_static=True,
            capability_unavailable="static GPU table filter unsupported; empty buffers cannot prove absence",
            created_clock=before, source_sha256=pins, paths=paths, table_path=table,
            table_colliders=resolved, missing_reporting=None, capacity=legacy.CAPACITY)
        if legacy._clock(read_clock) != before:
            raise ValueError("clock changed while opening contact sources")
        return legacy.BimanualTableContacts(stage, (), metadata, before["physics_step"])
    finally:
        stage = hands = prepared = read_clock = hand = view = None
