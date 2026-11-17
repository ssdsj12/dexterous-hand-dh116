# Segment geometry to measured R0 approach

Base69d39a96dda9bd3b3aa9c66c64dc82d00f42ec7b. CPU-only source adapter and disk replay of already completed234-step run. No native/PPO execution.

1. RED tests: legacy adapter rejects segment version; independent adapter consumes same source without relabelling raw records; stale/changed identities, SDF, radius and geometry rejected.
2. Add independent SegmentMeasuredPinchApproach reusing original measured/reset calculation with SegmentClosureGeometry validator. Default/old sources unchanged.
3. Stream SHA-pinned234-step ledger, same real reset baseline,29pre-action/control intervals. Consume actual common/action/provider using original PhysicalR0Diagnostic plus explicit existing applicability reducer. All missing measurements stay missing. Retain R0 terms and blocked reasons.
4. Bound disk-only replay wall600s, preserve partial evidence on failure. No new physical rows, stable/slip/contact labels or trainedmodel.
5. CPU regressions, protection528+2, report/readiness, Ubuntucommit→Windowspublish→remoteSHA/DraftPR verification.
