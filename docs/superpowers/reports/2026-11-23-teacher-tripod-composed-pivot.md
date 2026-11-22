# Tripod composed wrist increment preview

The independent, opt-in `--compose-pivot-preview` mode keeps the same 9 mm supported Tripod approach and original 24D wrist and finger command paths. A selected left hull that fails the original norm-summed rod-distance preview can receive a second check during a pure wrist pivot. The original table and support margins must still pass. Subsequent hulls, non-selected hulls, the right hand and finger motion keep the existing checks.

The second check reuses the original joint-command and wrist projection functions and declines any clamped increment. It evaluates five points of the same world translation and intrinsic XYZ command about the measured palm. Each point uses the actual measured rod and convex hull, the existing observed displacements, and a bound of one eighth of the full commanded point travel for the gaps between samples. The next measured post-step geometry, pair, force and joint guards still run at the original rates. The forecast depends on the measured movement allowance and does not certify future deformation.

The archived SDK 3101 frame reproduced the old preview failure with a 33.64 µm commanded travel sum. The new production preview function admitted the same candidate for the measured thumb source and archived thumb cooked hulls, with a minimum 29.24 µm reserve beyond the unchanged acceptance boundary. Regression tests also rejected the opposite translation, a clipped workspace command and an unsafe subsequent non-selected hull. No action was issued by the archive replay.

The final CPU suite passed 136 tests. The single native budget stays at 3338 SDK steps, 239 control windows and 40032 solver substeps, with one environment on GPU 2 and zero policy queries or PPO updates. The native qualification gate still needs measured opposing loads, release of the supports and the original grasp-specific lift and hold criteria.

## Native result on a 9 mm wire

The single Isaac/DeformX run consumed 3298 SDK steps and 39552 solver substeps in 744.81 seconds and exited naturally with code 0. The same-episode pivot advanced through the old refusal. The composed check was needed on 93 SDK steps and retained a minimum 18.42 µm predicted margin. The left thumb had positive load in all 12 solver substeps during every one of the 1024 pivot steps.

The left index closure advanced the SDK joint by 0.04946 rad. The thumb first unloaded at SDK 3204, ten steps into index closure. The left index first made contact at SDK 3292 and delivered a 0.03092 N reaction through the native composer at 3293. Its peak normal load was 0.04473 N. At SDK 3298 the thumb clearance was +0.06406 mm. There were zero samples with opposing thumb/index loads and zero Teacher or PPO steps. The right hand remained stationary. No Lift or generalization was qualified.

The next control trial should preserve thumb load during index closure using the same completed-interval geometry and measured reaction. Additional wrist/finger increments must satisfy the existing guard and be measured again before attempting middle-finger contact or Lift. This run covers only one wire specification.

Frozen recipe `77576a5247ffe198f2a6ed9d0991a3bee3fa3d22d9150a225744b4fbbaaa0db9`. Outer receipt `8e14856abb32d7c6bdc6ba06e88847cca90905f92d0a61f0cde5826d05de0a4c`. The offline audit independently verified all 6331 artifact digests, same-episode timing and reaction transport.
