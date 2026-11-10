"""Create a reward-only experiment checkpoint; never run Isaac or alter input."""
import argparse

from go2_pvcnn.control.dh116.tabletop_vision_reward_derivation import derive_reward_experiment_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', choices=['R1', 'R2', 'R3', 'R4', 'R5'], required=True)
    args = parser.parse_args()
    report = derive_reward_experiment_checkpoint(args.input, args.output, args.profile)
    print('DERIVED_REWARD_CHECKPOINT', report['checkpoint_sha256'])


if __name__ == '__main__': main()
