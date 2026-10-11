# Local preservation area

This directory's contents, except this README, are intentionally excluded from Git.
No archived source, cached HTML, model weights or notebooks are distributed with the release.

On the maintainer's machine:

- `legacy-20261011/` preserves the original source, scraper/research tools, tests,
  configuration trees, notebooks, GX10 Docker helpers and development documentation.
- `local-preservation/pre-release-20261011/` holds a verified Git history bundle,
  an original tracked/untracked file SHA-256 inventory and copies of files changed
  by the cleanup. This is the recovery point for the former repository state.
- Existing `archive/data/`, raw `data/corpus/` and superseded result directories remain
  locally available; they are ignored rather than deleted.

The public supported implementation lives in `src/chandassu`, with `train.py` and
`inference.py` as entry points. Complete final evidence lives in `results/v1`.
