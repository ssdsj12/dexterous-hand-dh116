# Right-owned rod stations and radial surface references
Base8fe74cf793fd77f50cc3415da7334695c318aad7. Prior reflected targets put thumb/index at material5/6 while actual right source surfaces are nearest4/3. Do not impose left material ordering.
Derive station indices and X-side from right USD/FK/SDF at actual SDK234 root and q. Keep unchanged41-node Reset rod, originaltable/limits/couplings. Construct three explicit radial references (normalZ0,-0.2,-0.42), independent target normals not measured forces. No left normal or joint mirroring. Each root must be proper; fit right3joints within originalbounds, folded remaining1.15rad.
Budget:3referencecases,60nfev percase,768FK/120sCPU,noIsaac. All17source+5cooked geometries rechecked. Ifnonepass,retainfailures; ifanyendpointpasses, prepare separate full closure/reset audit. No automaticnativequalification.

After all three reference-only candidates failed, add a distinct constrained CPU solve with exact all-shape table/rod inequalities in the residual: two frozen starts,1200FK,55nfev/start,120s. No native budget. Passing endpoints require separate whole-path qualification; local numerical stagnation is not infeasibility.
