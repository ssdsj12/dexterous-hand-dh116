# Native Power rod suspension with independent supports

The counterfactual parked both original hands 105 mm away from their contact initial poses. Everything governing the original rod, table, supports and hand safety stayed fixed. One native run completed 240 SDK steps, including two initialization steps, and 2,856 DeformX substeps. The paired wrist and hand ledger recorded zero hand–rod pairs and the table carried no contact. Each support had exactly one loaded pair in every solver substep.

| Quantity | Negative-Y support | Positive-Y support |
| --- | ---: | ---: |
| Peak aggregate force | 0.156464 N | 0.156513 N |
| Mean upward force in final 50 ms | 0.137665 N | 0.137657 N |
| Solver substeps with support load | 2,856/2,856 | 2,856/2,856 |

The final minimum centerline height was 347.067 mm (17.933 mm below Reset). Peak rod speed in the final 50 ms was 0.015997 m/s. The test verifies two-sided suspension without hand support across this interval. It does not establish static equilibrium or a grasp. The aligned contact replay and aggregate checks gave zero pair force reconciliation residual. The parked comparison supports the explanation that the previously observed loss of positive-Y support followed the incidental left-pad force.

The instance returned from App.close and outer and GDB inferior both exited naturally 0 in 64.237 seconds. Recipe SHA-256 `ffea4b5975fa2369775896fc355e0d3cc8c89a14d98404e0d070b4e77e091551`, outer receipt SHA-256 `8c0e73409e3a4ef001d1f9807439c14689c538be0d74f944208ab3ea0b4c9620`. Each complete exchange and support reducer row was independently revalidated from the hashed artifact list. Use `python -B -m scripts.verify_dh116_power_supported_task --settle-without-hand-contact` with the pinned files and independent run directory. For next Power/Tripod entry geometry, use an actual settled solver curve and preserve its nonzero velocity/rest-length state.
