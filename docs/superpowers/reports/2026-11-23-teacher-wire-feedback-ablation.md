# Wire posture and force response on identical observations

The CPU diagnostic evaluates one frozen feedback Teacher four times on each identical, source-bound pre-action state. The base privileged input is identical in every query. Only the optional palm-local position and tangent, or the preceding control window's feedback, change. The tool computes bounded 24D deterministic means, retains signed per-axis differences and reports absolute maxima for each hand's finger, wrist translation and wrist rotation commands. It reuses the canonical surface-motion24 slices and records their semantic digest and the input and policy digests.

| Input comparison | What the difference measures |
| --- | --- |
| Geometry only minus feedback absent | Effect of adding the palm-frame geometry branch |
| Same valid pads with counterfactual zero force minus geometry only | Effect of force availability flags |
| Measured geometry and force minus the same valid pads with counterfactual zero force | Effect of the measured reaction values with unchanged validity |

Unavailable values remain NaN in model-facing tensors. The fabricated zero is confined to the numerical comparison. The full measurement is bound to its control token, pre-action and observation digests, and prior completed window. This sequential decomposition is conditional on the given state and mask. The deltas are neither issued actuator commands nor measurements of physical tracking or successful grasping.

The focused CPU tests exercise Pinch, Tripod and Power using diagnostic weights with independently selected geometry, force-validity and force-value inputs. They reject a stale prior window. The archived Tripod playback validates the source identities of eight decisions and eight bootstraps, retaining one first-decision missing window and 15 measured zero-load windows. All four actions coincide with the read-only base checkpoint at zero residual initialization. The archive is a short diagnostic and holds no positive-load policy decisions or eligible experts.

The independent training catalog contains 11 training specs, two validation specs and eight held-out test specs formed by 9.2/10.5 mm diameters, 475/525 mm lengths and 0.7/1.3 elastic modulus multipliers. The latest common pregrasp covers 104 static held-out layouts per grasp. Those results establish predicted reset clearances. A frozen closed-loop evaluation still needs independent native resets, actual dual-hand load and hand-side reaction delivery. Power requires whole-hand enclosure with palm and five fingers. Training and checkpoint selection use the training and validation splits; the held-out trial identities stay in evaluation.

The next comparison starts on training wires with the original safety gates and the same hardware-facing commands. Verify bilateral opposition and lift the actual rod clear of both supports before measuring 30 mm net Lift, a continuous five-second Hold and at most 0.5 mm Slip. On the frozen holdouts, report each grasp and specification, worst-case success, force and posture corrections, and support load. The audit can diagnose which observations change a proposed action, while the paired native runs determine whether those changes help. No Isaac instance, optimizer update or controller was invoked by this package.

Antigravity verified the source digests, unchanged validity in the zero-force counterfactual, and correct 24D hand/channel mapping. The author independently checked the source action contract and the latest common pregrasp report. Keep the containing report and scope when consuming individual per-state rows.

The combined regression passed 249 CPU tests in 438.09 seconds. The output and command digests are in the readiness contract. The protection audit matched 528 source files and three persistent Teacher checkpoints. The two older diagnostic paths in the registry are absent from /tmp, so their contents could not be reverified.
