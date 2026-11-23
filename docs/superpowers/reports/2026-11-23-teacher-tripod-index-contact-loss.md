# Tripod index unloading and flexible-wire generalization

## Archived contact evidence

The original supported Tripod run ended naturally and never cleared both supports. The audit reads its receipt-bound native files. It follows the previously loaded material node between adjacent SDK steps and evaluates the locked left index pad SDF at all four combinations of measured rod positions and measured link poses. The sum of the two motion contributions is the observed endpoint gap change.

- Across 69 adjacent first-loss pairs, all 69 same-node endpoint gaps crossed from negative to positive. The median gap went from −8.08 µm to +20.28 µm. Each loaded step involved one source node, and all 69 nodes stayed inside the four SDF sampling lattices.
- Left index pad motion contributed a median +31.45 µm to the gap. Rod motion contributed a median −3.12 µm. Separating measured palm motion and palm-relative finger motion gives medians of −0.188 µm and +31.61 µm, respectively. The wrist target stayed fixed. The palm origin still moved by up to 7.54 µm and the index link origin by up to 13.81 µm.
- The original left index close increment was issued on 16 loss transitions. All 69 recorded finger21 and finger22 positions decreased. Median finger21 target minus measured q was 0.001503 rad. The measured World force and torque routing for the index body matched the preceding SDK packet on all 69 transitions.
- These are measured endpoint associations from one supported wire and one physical run. They leave the within-step force/pose order, actuator response and contact compliance unresolved. Test finger21/22 tracking under measured load with the existing wrist target fixed before selecting a closing correction. There is no new native hold result from this audit.

## Cross-specification acceptance

Reuse the existing eight held-out wire specifications. The diameter endpoints are 9.2 mm and 10.5 mm, length endpoints 475 mm and 525 mm, elastic modulus multipliers 0.7 and 1.3. Group all initial conditions and replays by physical-specification identity. Freeze one policy per grasp before testing. Keep the final minimum of 100 independent native initial conditions per grasp, and record per-specification and worst-case results. Keep independent placement coverage visible. The current CPU placement screen cleared 24/104 Pinch, 72/104 Tripod and 0/104 Power rows. Power still needs a separate viable initial pose for wrap contact, and the placement screening failures remain in the reported denominator.

Evaluate the same initial conditions with the frozen baseline, measured palm-local geometry corrections, and geometry plus measured pad/palm reactions. Log wrist pose corrections, 24D issued actions, per-pad reaction and actual q/qd, support clearance and slip. The controller should use palm-frame contact geometry and calibrated reaction with validity masks. Keep held-out specification identifiers out of Actor inputs. Require measured opposing contact and a stable fixed-wrist hold before lifting. Power keeps the palm and five-finger wrap requirement. Final native success still requires net Lift ≥30 mm, unsupported Hold ≥5 s and Slip ≤0.5 mm. The two fixed RGB cameras retain their existing calibration and synchronization contract.

The combined CPU run passed 101 tests. This package ran zero new Isaac steps and zero PPO updates. Teacher admission and generalization remain pending.

The [readiness record](../../contracts/dh116_teacher_tripod_index_contact_loss_v1.json) contains the reproduction command, source and result SHA-256 values.
