# Tactics2D agent guide

This file is the canonical repository guidance for AI coding agents. Tool-specific
instruction files should point here instead of duplicating these rules.

## Scope and compatibility

- Preserve the public API unless the task explicitly calls for a breaking change.
- Keep changes focused. Do not rewrite unrelated code or generated artifacts.
- Support Python 3.10 and later, as declared in `pyproject.toml`.
- Treat optional dependencies as optional. Tests that require unavailable datasets,
  checkpoints, services, or extras should skip with a clear reason.

## Domain invariants

- Before changing geometry or trajectory code, confirm the module's coordinate system,
  angle convention, physical units, and frame or timestamp semantics from the local API
  and tests. Do not silently mix degrees and radians, seconds and milliseconds, or local
  and global coordinates. State units in the public documentation of new fields.
- Preserve the orientation relationship among lane boundaries, centerlines, and travel
  direction. Keep trajectory frames ordered and unique.
- Do not mutate caller-owned trajectories, arrays, or geometry objects unless the API
  explicitly documents in-place behavior.
- Do not silently repair invalid geometry when the intended shape cannot be recovered
  reliably. Raise an error that identifies the failing input instead.

## Public API and optional dependencies

- When adding a public class or function, update the nearest package `__init__.py` export
  when appropriate. Keep public parameter names, defaults, return types, and exception
  behavior stable unless a breaking change is explicitly requested.
- For an intentional breaking change, update affected tests, reference documentation,
  examples, and the changelog together.
- Importing the base package must not require optional libraries, datasets, checkpoints,
  or services. Delay heavyweight optional imports until their feature is used and raise
  an actionable error when the dependency or asset is unavailable.

## Python conventions

- Format Python and notebooks with Black at a line length of 100 and sort imports with
  isort's Black profile. Let `pyproject.toml` and `.pre-commit-config.yaml` define the
  executable formatting and lint rules.
- Within `tactics2d.behavior`, do not use generic filenames that collide with top-level
  domain modules or core types, such as `trajectory.py`, `state.py`, or `map.py`. Name
  files after their specific responsibility, such as `trajectory_processing.py` or
  `decision_state.py`.
- New Python files should begin with the current-year copyright notice and SPDX line,
  followed by a concise module docstring:

  ```python
  # Copyright (C) YYYY, Tactics2D Authors. Released under the GNU GPLv3.
  # SPDX-License-Identifier: GPL-3.0-or-later

  """Concise module description."""
  ```

- Document public interfaces and non-obvious behavior. Include parameter meaning,
  units, return values, and directly raised exceptions when they are not evident from
  the signature.
- Comments should capture useful constraints, invariants, units, or necessary design
  rationale. Do not restate the code or preserve change history in source comments.

## Tests

- Add or update focused pytest coverage for behavior changes and bug fixes.
- Prefer readable arrange-act-assert tests with direct assertions. Use fixtures for
  shared setup rather than hidden global state.
- Apply existing pytest markers when a test belongs to a marked subsystem or has
  integration/environment requirements. Ordinary unit tests do not need a marker.
- Run the smallest relevant test selection first. Expand to the broader suite when the
  change has cross-module impact.
- Run the full suite with an explicit test path, such as `python -m pytest tests`, rather
  than bare `pytest` from the repository root. Bare discovery can recurse through the
  dataset symlinks under `tactics2d/data` and spend a long time collecting no tests.
- For faster local full-suite runs, use up to four file-level workers (for example,
  `pytest tests -n 4` when pytest-xdist is installed) and set `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` to avoid nested thread oversubscription.
  If pytest-xdist is unavailable, split disjoint test modules across processes instead.
- Unit tests must not access the network, download assets, or depend on machine-specific
  absolute paths. Use `pytest.importorskip` or an explicit skip for optional resources.
- Seed randomized tests. Use justified numeric tolerances for floating-point and
  geometric results rather than exact equality.
- A bug fix should include a regression test that fails for the original behavior.
- Keep transient test output under ignored runtime or temporary directories; do not
  commit downloaded datasets, model checkpoints, caches, or generated build output.

## Native extensions

- Changes to C++ extensions must preserve the corresponding Python behavior, including
  accepted inputs, returned values, exceptions, and documented numerical tolerances.
- Keep native code portable across the operating systems and Python versions supported
  by the project. Test both native and fallback paths when both exist.

## Documentation and notebooks

- When a public interface or example call changes, find and update every affected
  notebook under `docs/` as part of the same change.
- Re-run affected documentation notebooks from a clean kernel and verify that all cells
  complete in order with current outputs. Do not treat a text-only notebook edit as
  sufficient validation.
- If an affected notebook cannot run because an optional dataset, checkpoint, service,
  or other external resource is unavailable, report that explicitly and avoid claiming
  the notebook was validated.

## Changelog and validation

- Add a concise entry under the appropriate `CHANGELOG.md` `[Unreleased]` section for
  user-visible features, fixes, removals, or compatibility changes. Pure refactors,
  tests, and internal tooling changes normally do not need an entry.
- Before handing off a code change, run the relevant tests and the applicable
  pre-commit checks. Report any checks that could not run and why.
- Do not claim success based only on formatting or static inspection when behavior can
  be exercised with a focused test.
