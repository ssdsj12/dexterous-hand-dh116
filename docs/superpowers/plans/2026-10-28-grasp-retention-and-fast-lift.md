# Retain the object and lift five times faster

User explicitly requests real grasp (object moves with the fingers, no sustained sliding) and five times current lift speed. Current coordinated speed is0.01m/s; change to0.05m/s. Increase wrist solver cap from0.03 to0.05m/s, retaining0.06m/s actual transport guard and0.003m tracking guard. Do not fake object attachment or lift success. Current-held pose and instantaneous bilateral admission remain.

Task1 controller: add lift-only grip-loss recovery based on measured relative slip and contact loss, not repeating close admission on every sample. Retain applied pose; use bounded stronger force feedback to recover grip (requested target stayswithin2–12N/hand). Lift pauses for detected persistent grip loss/sliding while feedback tightens; resumes on recovered bilateral contact and no growing slip. Preserve mechanical/severe/force safety and original physical success criteria. Explicit contract and meaningful regression tests; use same physical case set.

Task2 observable demonstration: teacher monitor publishes per-environment lift target, actual/target palm worldZ, contact center relative to object, current loads, force targets, phase and reasons for lift pause. Invalid contact points are null, never fabricated. GUI draws non-colliding markers at valid force-weighted contact centers for the visible first environment. Provide a safe polled stop request to save trace once; no repeated SIGINT during save.

Task3 review and real Isaac: one process at a time; fresh8case validation, latestcode, compare actual hands/object displacement and contact slip; do not call empty-hand motion a successful grasp. No commit/push requested.
