# TensorBoard force direction labels

User requests upward/downward force annotations in TensorBoard. Add per-live-sample positive/negative parts of the signed bilateral force before rollout averaging; both plotted magnitudes nonnegative, units N. Include positive and negative peak magnitudes, signed net mean, and object weight reference. Continue historical telemetry fields unchanged.

Implement GPU accumulators and clear together with existing rollout statistics; test opposing hands and positive/negative samples, inactive outliers and reset. Add Chinese labels under force_N and English directional fields under vertical_force. No reward/control/observation/checkpoint contract changes. Preserve current checkpoint and counters through a single-instance process restart; verify real event records and policies/optimizer continuity. Old v126 logs retained. No deletion or fresh random training required for instrumentation.
