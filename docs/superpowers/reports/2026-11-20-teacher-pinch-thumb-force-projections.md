# Pinch left thumb force projection

The audit reads the two preserved native ledgers by exact SHA. It uses the same consecutive SDK ticks 299–362 and all twelve solver substeps within each tick. There are 768 left index/thumb pairs per trial, always on rod nodes 34/35. One pair is required per pad and substep. Each measured force is projected along the line between contact points. The normal contribution and the remainder are reported separately.

| Trial | Thumb inward force P05 (mN) | Minimum (mN) | Signed opposition failures |
| --- | ---: | ---: | ---: |
| Fixed wrist | 45.03 | 18.15 | 0/768 |
| Longitudinal wrist adjustment | 14.89 | −13.16 | 9/768 |

In all nine failures, the measured thumb normal projected inward and the tangential contribution outweighed it. The source samples therefore localize the lost opposition to the net thumb force direction in this comparison, with both contact channels still present. The comparison combines the response to the tested seating and preload schedule. It does not identify an isolated coefficient or show sustained bilateral force closure.

Replay command: `python -B -m scripts.audit_dh116_pinch_thumb_projections --fixed <fixed physical.jsonl> --longitudinal <adjusted physical.jsonl> --output <new JSON file>`. The CPU regression passed 232 tests. This package launched no Isaac or PPO jobs. Original per-episode qualification remains unchanged. The next controlled candidate should keep both wrists fixed during closure and require measured opposed reactions before allowing lift.
