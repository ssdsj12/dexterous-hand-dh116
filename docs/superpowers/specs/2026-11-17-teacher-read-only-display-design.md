# Read-only live Teacher display

User approved this design in the current conversation on 2026-11-17 UTC.
The actual original USD hands, articulation state, collision, controllers,
checkpoint actions and DeformX solver remain unchanged. Only the original visual
geometry is hidden from rendering. Independent display-only body Xforms contain
exact copies of visual geometry, no rigid-body/collision/articulation APIs, no
joints, and no physics attributes or relationships. Their poses follow measured
SDK body-link world positions and wxyz quaternions every displayed frame.

Use the published GitHub visual_pose_sync row-vector composition convention.
Never author measured poses onto the running physical body prims.
Verify every copied geometry's provenance and each resulting display transform.
Fail before exposing the viewport if any display link exceeds 0.1mm position
error or 0.001rad rotation error. Verify physics-clock invariance during sync.
Record per-link identities and errors; never grant grasp success from a picture.
Three existing models and their reset contracts remain separately source-bound.
One WebRTC button cycles Pinch/Tripod/Power views.

CPU USD tests precede a separately authorized bounded native check. Keep native
failure, normal exit and actual grasp qualification separate. No PPO is started.

All opinions use an owned anonymous session sublayer. Asset RootLayer bytes and the user's EditTarget remain unchanged. Closing removes only the owned layer; concurrent user edits remain. The viewport is hidden before creating the physical scene.
