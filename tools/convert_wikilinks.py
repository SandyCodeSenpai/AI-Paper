"""Convert Obsidian-style [[wikilinks]] in wiki/*.md to standard markdown links.

GitHub doesn't render [[wikilinks]] — they show up as literal text. Standard
[label](relative/path.md) links render in both Obsidian and GitHub.

Run: python3 tools/convert_wikilinks.py
"""

import os
import re
import sys
from pathlib import Path

WIKI = Path(__file__).resolve().parent.parent / "wiki"

# Matches [[target]] and [[target|alias]]. Targets do not contain ] or |.
WIKILINK_RE = re.compile(r"\[\[([^\]\|]+?)(?:\|([^\]]+))?\]\]")


def resolve_target(target: str) -> Path | None:
    """Resolve a wikilink target to a file under wiki/. Returns None if broken."""
    target = target.strip()
    candidates = [
        WIKI / f"{target}.md",
        WIKI / target / "README.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def convert_file(md_path: Path) -> tuple[int, list[str]]:
    """Convert all wikilinks in md_path. Returns (n_changes, broken_targets)."""
    content = md_path.read_text(encoding="utf-8")
    broken: list[str] = []
    n_changes = 0

    def repl(m: re.Match) -> str:
        nonlocal n_changes
        target_str = m.group(1)
        alias = m.group(2)
        target_path = resolve_target(target_str)
        if target_path is None:
            broken.append(target_str)
            return m.group(0)  # leave broken links as-is for visibility
        rel = os.path.relpath(target_path, md_path.parent)
        text = alias if alias else target_str
        n_changes += 1
        return f"[{text}]({rel})"

    new_content = WIKILINK_RE.sub(repl, content)
    if new_content != content:
        md_path.write_text(new_content, encoding="utf-8")
    return n_changes, broken


def main() -> int:
    if not WIKI.exists():
        print(f"No wiki at {WIKI}", file=sys.stderr)
        return 1

    total_changes = 0
    all_broken: list[tuple[Path, list[str]]] = []

    for md_path in sorted(WIKI.rglob("*.md")):
        n, broken = convert_file(md_path)
        rel = md_path.relative_to(WIKI.parent)
        if n or broken:
            print(f"{rel}: {n} link(s) converted" + (f", {len(broken)} broken" if broken else ""))
        if broken:
            all_broken.append((md_path, broken))
        total_changes += n

    print(f"\nTotal: {total_changes} wikilinks converted across the wiki.")

    if all_broken:
        print("\nBroken links (target file does not exist; left unchanged):")
        for path, broken in all_broken:
            for b in broken:
                print(f"  {path.relative_to(WIKI.parent)}: [[{b}]]")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
