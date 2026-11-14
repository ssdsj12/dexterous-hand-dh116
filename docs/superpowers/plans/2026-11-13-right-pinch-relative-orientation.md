# Independent right palm/rod relative Reset orientation

Base b3ba69674e748aa917bfbd9f00a9568da0cf6d7e. Last V4 has actual source-hull/capsule overlap proven by feasible-distance upper witnesses. Existing rod-axis roll preserves continuous rod/hand relative geometry and cannot by itself fix that interference.

Add only opt-in fixed world X/Z orientation around right grasp pivot(.43,-.2,.305), each within0.05rad. Preserve earlier seven-variable physical box and original delta/velocity/limits. This changes Reset posture, not action amplitude. Do not alter source hulls, SDF or rod nodes.

RED tests for rotation composition/pivot, bounds, retained extra angles in every path sample and generic nine-variable direction. Reuse Geometry.measure/path and existing bounded max-min LP; old4/6/7D paths unchanged. Six passes maximum,246endpoint measurements,161path only if all endpoints pass,120s CPU,0native/0PPO. No automatic second batch.

Only full FK/nonlinear constraints may pass. Fresh authored-root consistency/self-collision/physical Reset remain separate gates even if counterfactual path passes. Publish test/report/readiness/protection evidence through Windows; PR2Draft.
