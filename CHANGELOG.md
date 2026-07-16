# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` file (MIT) — addresses wiki issue #1
- `CHANGELOG.md` (this file) — addresses wiki issue #10
- `tests/` directory with a smoke test — addresses wiki issue #5
- Wiki at https://github.com/IFoA-ADSWP/TabPFN/wiki (Issues disabled, backlog lives in the wiki)
- `docs/MAINTENANCE_BACKLOG.md` (canonical source for the wiki)
- `scripts/import_backlog_to_github.py` (bulk import when Issues are re-enabled)
- `scripts/push_wiki.sh` (sync `docs/MAINTENANCE_BACKLOG.md` to the wiki)

### Changed
- `.gitignore` now excludes `.venv*/`, `outputs/archive/`, `*.pkl`, `*.tabpfn_fit`, `**/.DS_Store`, `**/catboost_info/`
- README install instructions still reference the broken `pip install -r requirements.txt` — will be fixed when `pyproject.toml` lands

### Removed
- `.venv/` and `.venv312/` untracked from git (628+ files)

### Fixed
- Wiki: GitHub Issues disabled, so maintenance backlog lives in the wiki

## [0.0.0] — pre-release

Initial state. No versioned release. Pre-licence, pre-changelog, pre-tests.
