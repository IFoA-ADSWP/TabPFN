#!/usr/bin/env python3
"""Import the maintenance backlog into GitHub Issues.

Each `### Issue N — Title` section in `docs/MAINTENANCE_BACKLOG.md` becomes a
GitHub issue. Run this once Issues are enabled on the repo.

Usage:
    python scripts/import_backlog_to_github.py

Requires:
    - gh CLI installed and authenticated
    - `gh repo set-default IFoA-ADSWP/TabPFN` (or pass --repo)
    - Issues enabled on the target repo

The script is idempotent on the backlog file (re-running reads it again) but
not on GitHub — already-created issues will be duplicated. Use --dry-run first.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_BACKLOG = Path(__file__).resolve().parents[1] / "docs" / "MAINTENANCE_BACKLOG.md"
DEFAULT_REPO = "IFoA-ADSWP/TabPFN"

# Mapping from Tier to GitHub label.
TIER_LABELS = {
    "T1": ["documentation", "good first issue"],
    "T2": ["enhancement"],
    "T3": ["enhancement"],
}


def parse_backlog(md_path: Path) -> list[dict]:
    """Return a list of {number, title, tier, effort, body} dicts.

    Tier is sourced from the summary table at the top of the file.
    Effort is sourced from each issue's body (`**Effort.** ...`).
    """
    text = md_path.read_text()

    # First, extract tiers from the summary table.
    # Table rows look like: `| 1 | Add LICENSE file | T1 | 5 min | ⬜ Not started |`
    tier_by_number: dict[int, str] = {}
    table_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|.*?\|\s*(T[123])\s*\|", re.MULTILINE
    )
    for m in table_pattern.finditer(text):
        tier_by_number[int(m.group(1))] = m.group(2)

    # Then, extract issues from `### Issue N — Title` sections.
    issues: list[dict] = []
    section_pattern = re.compile(
        r"^### Issue (\d+) — (.+?)$\n(.+?)(?=^### Issue \d+ — |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for match in section_pattern.finditer(text):
        number = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3).strip()

        effort_match = re.search(r"\*\*Effort\.\*\*\s+(.+?)(?:\.|$)", body)

        issues.append(
            {
                "number": number,
                "title": title,
                "tier": tier_by_number.get(number, "T2"),
                "effort": effort_match.group(1).strip() if effort_match else "?",
                "body": body,
            }
        )

    return issues


def create_issue(repo: str, issue: dict, dry_run: bool = False) -> None:
    title = f"[{issue['tier']}] {issue['title']}"
    labels = ",".join(TIER_LABELS.get(issue["tier"], []))
    cmd = [
        "gh", "issue", "create",
        "--repo", repo,
        "--title", title,
        "--body", issue["body"],
    ]
    if labels:
        cmd += ["--label", labels]

    if dry_run:
        print(f"[DRY-RUN] Would create: {title} (labels: {labels})")
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR creating issue {issue['number']}: {result.stderr}", file=sys.stderr)
    else:
        # gh prints the issue URL on stdout
        url = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "?"
        print(f"[{issue['number']}] Created: {title} → {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import maintenance backlog to GitHub Issues.")
    parser.add_argument("--backlog", type=Path, default=DEFAULT_BACKLOG)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target repo (owner/name)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created")
    parser.add_argument("--from", dest="start", type=int, default=1, help="Start from issue number")
    parser.add_argument("--only", type=int, nargs="*", help="Only create these issue numbers")
    args = parser.parse_args()

    if not args.backlog.exists():
        print(f"Backlog file not found: {args.backlog}", file=sys.stderr)
        return 1

    issues = parse_backlog(args.backlog)
    print(f"Parsed {len(issues)} issues from {args.backlog}")

    if args.only:
        issues = [i for i in issues if i["number"] in args.only]

    for issue in issues:
        if issue["number"] < args.start:
            continue
        create_issue(args.repo, issue, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
