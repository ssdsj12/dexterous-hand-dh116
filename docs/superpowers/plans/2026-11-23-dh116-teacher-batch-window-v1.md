# DH116 Teacher batch window implementation plan

The R21 design reuses the original 480 Hz exchange and PPOTrainer. A window owns a shared absolute SDK boundary. Each lane retains its own episode, Reset and relative solver clock.

- Add a current-source CPU clock audit alongside the historical clock-consumer gate. Bind the checked source inventory, invoke the shipped readers at SDK 240, 258, 2402 and 9600, preserve nominal rod and SDK stamps, and mark native clock acceptance pending.
- Add a batch control-window entry point. The existing direct diagnostic closes its entire group on any stop. The batch entry point commits only a full trustworthy window, retires lanes with trusted window-end events, then permits a same-owner zero-step Reset of only those lanes. Exceptions and hardware guards stop the group before the commit callback.
- Check two-lane GAE with the existing PPOTrainer, terminal and timeout endpoints, mismatched Reset ownership and an injected early guard. Run the focused CPU suites before publication. Native training still requires real per-lane source and reward wiring and the separate two-lane native experiment.
