# Original Vs Current Best Status

## Decision

```text
direct_original_vs_current_table = unavailable
```

I did not run an original-vs-current benchmark because the repository does not
currently contain a clean, rebuildable "very first original" source snapshot.

Existing project notes say:

```text
orig_code matches current runnable source hashes for key files
```

Therefore `orig_code` cannot be used as proof of an unmodified original
baseline.

## What Was Checked

- Local git branches and tags related to `orig`, `original`, `baseline`, and `current`.
- Local directories including `archives/`, `remote_work/`, `overnight_20260607/`, and `feedback/`.
- Project notes in `AGENTS.md`, `AGENT_LOG.md`, and `docs/`.

Findings:

- `archives/speedups/2.0x_20260603_033651_rank6_3gpu_concurrent_shots/` is a
  provisional scheduling experiment, not a clean original source baseline.
- `remote_work/current/` and `overnight_20260607/` are intermediate optimized
  or synced states, not the first original source.
- `orig_code` is explicitly recorded as matching the current runnable source
  for key files and must not be used as an original baseline.

## Current Best

```text
candidate                     current_best_v_pml_len16
WP speedup vs zmem             1.222023x
Gradient speedup vs zmem       1.206588x
max rel L2                    6.384336e-07
```

## Estimated Total Speedup

If the earlier project-level `1.8x` anchor is accepted, and if
`zmem_reference` is accepted as `1.049300x` WP over `current_best_reference`,
then:

```text
estimated WP speedup vs first original
  = 1.8 * 1.049300 * 1.222023
  = 2.308x

estimated Gradient speedup vs first original
  = 1.8 * 1.046865 * 1.206588
  = 2.274x
```

This should be reported as an estimate, not a formal direct measurement.

## Reopen Condition

Only create a direct original-vs-current table after one of these appears:

- A git commit, tag, tarball, or directory that can be proven to be the true
  original source.
- A rebuildable original binary plus input/output hashes and environment notes.

When available, rerun on the same RTX 5090 platform:

- `smoke_1gpu`
- `correctness` if the original can produce matching outputs
- `perf_1gpu_6shots` x 3

Then generate:

```text
docs/original_vs_current_best_<date>.md
reports/original_vs_current_best_<date>/summary.json
```
