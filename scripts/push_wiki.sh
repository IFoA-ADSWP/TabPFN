#!/usr/bin/env bash
# Push the maintenance backlog content to the GitHub Wiki.
#
# PREREQUISITE: The wiki must be initialized. To do this:
#   1. Go to https://github.com/IFoA-ADSWP/TabPFN/wiki
#   2. Click "Create the first page"
#   3. Title: "Home", body: anything (e.g., "init")
#   4. Save
# After that, run this script.
#
# The wiki repo is at: https://github.com/IFoA-ADSWP/TabPFN.wiki.git

set -euo pipefail

REPO="IFoA-ADSWP/TabPFN"
WIKI_DIR="${1:-/tmp/tabpfn-wiki}"
CONTENT_DIR="$(cd "$(dirname "$0")/.." && pwd)/.wiki-content"

echo "==> Cloning $REPO.wiki.git into $WIKI_DIR"
if [[ -d "$WIKI_DIR" ]]; then
  echo "    (directory exists, will pull instead)"
  (cd "$WIKI_DIR" && git pull)
else
  gh repo clone "$REPO.wiki" "$WIKI_DIR"
fi

echo "==> Syncing content from $CONTENT_DIR"
rsync -av --delete \
  --exclude='.git' \
  --exclude='Home.md' \
  "$CONTENT_DIR/" "$WIKI_DIR/"

cd "$WIKI_DIR"

# Stage and commit
git add -A
if git diff --cached --quiet; then
  echo "==> No changes to commit"
else
  git -c user.name="scotthawes" -c user.email="scottlhawes@gmail.com" \
    commit -m "docs(wiki): sync maintenance backlog from docs/MAINTENANCE_BACKLOG.md

- Home: landing page with quick links
- Maintenance-Backlog: full 36-issue backlog
- Tier-1-Critical / Tier-2-Important / Tier-3-Polish: split by priority
- Production-Readiness-Audit: original audit findings
- _Sidebar / _Footer: navigation"
  git push
  echo "==> Pushed to $REPO.wiki"
fi

echo "==> Done. View at: https://github.com/$REPO/wiki"
