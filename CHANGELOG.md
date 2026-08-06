# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- docs: human-voice cleanup pass — removed AI-assistant framing and phantom file references (papers/status/analyses/notebooks/data)
- docs: frontier AUC/Brier rescore — §14.11 addendum (2026-08-06): per-fold AUC/Brier in `frontier_results_*.csv`; ausprivauto0405 "DOMINATED" retracted to calibration tie + ranking edge; verdict reframed to best risk-ranking model (TabPFN AUC #1 of 9 methods on all 6 classification datasets); stale-verdict reconciliation across regime/portfolio/papers + wiki reframe (commits 4e7912c, 9037b26, cea967d, ced3ac9)

### Added
- `LICENSE` file (MIT) — addresses wiki issue #1
- `CHANGELOG.md` (this file) — addresses wiki issue #10
- `tests/` directory with a smoke test — addresses wiki issue #5
- Wiki at https://github.com/IFoA-ADSWP/TabPFN/wiki (Issues disabled, backlog lives in the wiki)
- Backlog tracking: `TASKS.md` (the planned `docs/MAINTENANCE_BACKLOG.md` was never created)
- `scripts/infra/import_backlog_to_github.py` (bulk import when Issues are re-enabled)
- `scripts/infra/push_wiki.sh` (sync the backlog to the wiki)

### Changed
- `.gitignore` now excludes `.venv*/`, `outputs/archive/`, `*.pkl`, `*.tabpfn_fit`, `**/.DS_Store`, `**/catboost_info/`
- README install instructions now point to the working `requirements.txt` file

### Removed
- `.venv/` and `.venv312/` untracked from git (628+ files)

### Fixed
- Wiki: GitHub Issues disabled, so maintenance backlog lives in the wiki

## [0.0.0] — pre-release

Initial state. No versioned release. Pre-licence, pre-changelog, pre-tests.
