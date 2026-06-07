# Validation Note

This archive was created after `perf_3gpu --np 6 --gpus 0,1,2` reported:

- Baseline `WP computing time`: `3.491814 s`
- Candidate `WP computing time`: `1.549196 s`
- WP-based speedup: `2.254x`
- Output comparison: passed

After a timing audit, this archive should be treated as a provisional scheduling experiment, not a confirmed 2.0x CUDA performance milestone.

Reason:

- The candidate changed the MPI decomposition from `np=3` to `np=6`.
- In this code path, `WP computing time` is measured on the root rank and does not reliably represent whole-job runtime when the number of MPI ranks changes.
- Whole-job metrics did not confirm a 2.0x speedup:
  - Baseline: `Gradient TIME all = 4.135657 s`, elapsed wall clock `0:08.39`
  - Confirmed 1.5x archive: `Gradient TIME all = 2.530737 s`, elapsed wall clock `0:06.85`
  - `np=6` best run: `Gradient TIME all = 2.559642 s`, elapsed wall clock `0:07.68`
  - `np=6` solo recheck: `Gradient TIME all = 2.984629 s`, elapsed wall clock `0:07.63`

Conclusion:

- The `np=6` scheduling idea is numerically valid and worth keeping as an experiment.
- It is not currently accepted as a confirmed 2.0x milestone.
- Future scheduling changes must compare `Gradient TIME all` and `/usr/bin/time -v` elapsed wall time, not only `WP computing time`.
