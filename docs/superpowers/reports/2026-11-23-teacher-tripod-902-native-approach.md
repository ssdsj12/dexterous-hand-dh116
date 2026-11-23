# Tripod 902 native precontact result

The approved single run reproduced the original supported Tripod path through SDK 552 and executed two new control windows. It ended at SDK 568 and exited naturally. Both the launcher and GDB recorded exit 0 in 288.53 s. No retry, forced exit, camera capture, model query or PPO update occurred.

| Quantity | Measured result |
| --- | ---: |
| Reset / initialization SDK steps | 1 / 2 |
| Parked SDK rows | 238 |
| Controlled SDK rows | 328 in 41 windows |
| New candidate SDK rows | 16 |
| DeformX solver substeps | 6,792 |
| Interlocks verified | 240, 552, 568 |
| Archived and candidate 24D command mismatches | 0 |
| Hand or table contact during the new tail | 0 |
| Left / right peak support force in the new tail | 0.15755 / 0.15770 N |
| Largest rod node motion from SDK 552 in the new tail | 0.08950 mm |
| Maximum right wrist position tracking error in the new tail | 0.02011 mm |
| Maximum right active joint tracking error in the new tail | 0.0010175 rad |

The external verifier hashed all 639 captured artifact files, checked contiguous SDK rows and matched each executed 24D request to the frozen schedule. It checked the same SDK, rod and measured hand geometry clock on every controlled row. The verified 552 prefix bound the exact rod, pose and issued-wrench checkpoint before any new command. Both supports retained positive normal load on every solver substep of the two new windows. The left and right hand meshes recorded no pairs during the run.

The measured 568 state is now a source for the next contact proposal. The run qualifies an approach with the rod still supported. Bilateral contact, opposing load, fixed-wrist hold, support-free lift, final slip and Teacher admission require separate evidence.

## Execution and evidence

Worker entry `bash scripts/python.sh -B -m scripts.verify_dh116_tripod_902_approach`. The Ubuntu run used detached execution code `f3bd58446be8828e0899aaf3bbd426aae61c51db`, the original Isaac environment, GPU 2, one environment, the SHA-bound recipe, and the previously verified GDB wrapper. Raw physics and debugger files remain on Ubuntu.

| Evidence | SHA-256 |
| --- | --- |
| Approved freeze | `c7e3fd4631a78ca0dadd2c8d66641a5727121e00724ea9a38c700bfda5000dc6` |
| Outer receipt | `3f404d60345a420e2d0e48141344f25b3eaa23c2531f871b4a8cc0c7faa385ca` |
| Independent post-exit reduction | `40348bc531db55c752ad57f5a06e0fb3084a68aaf16b5f81df7ad510e6a28a63` |
| SDK 568 measured control row | `1bc8dad63782345e3502afa72649c6a960b3cbd7a43d645168a38db4c666730f` |
| Shared result | `6f4731b9b81eaa5439cc463fd5832b1d0d24ad8bdc7e9e36c5947e10c95578ae` |
| Candidate tracking reduction | `8cf1052699b7365f511845f16dbe59bd2ea4cc64a4b2fffde82095c728d94922` |
