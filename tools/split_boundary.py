"""Repository-split boundary: which tracked files go to GABES main and which to
the private archive, and what in main still points at archive-side files.

docs/repo_restructure/README.md holds the plan and the user decisions; this file
is its machine-readable boundary (S2). Rules are ordered; the first match wins.

    python tools/split_boundary.py report              # counts + violations
    python tools/split_boundary.py report --json out.json
    python tools/split_boundary.py checkout DIR        # main-only shared clone at HEAD

A pattern ending in "/**" matches everything below that folder; "*" never
crosses a "/".
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN, ARCHIVE = "main", "archive"

# Ordered (pattern, side, reason). Keep reasons short; the hub explains them.
RULES: list[tuple[str, str, str]] = [
    # Grand Challenge code, tests, skill and campaign tools.
    ("tests/quantum/**", ARCHIVE, "GC test suite"),
    ("tools/thermal_campaign_*.py", ARCHIVE, "GC campaign tools (sealed controllers import them)"),
    ("analysis/grand_challenge/**", ARCHIVE, "GC analysis code and references"),
    (".agents/skills/grand-challenge-head-manager/**", ARCHIVE, "GC head-manager skill"),
    (".agents/skills/gabes-weekly-review/**", ARCHIVE, "review automation writes docs/daily_report"),
    # Tests that verify dated records rather than product behaviour.
    ("tests/test_fwm_chi_validation.py", ARCHIVE, "reads dated FWM calibration records"),
    ("tests/test_fwm_literature_validation.py", ARCHIVE, "reads dated FWM calibration records"),
    ("tests/test_fwm_stark_validation.py", ARCHIVE, "reads dated FWM calibration records"),
    ("tests/test_fwm_zeeman_validation.py", ARCHIVE, "reads dated FWM calibration records"),
    ("tests/test_fwm_stark_cache.py", ARCHIVE, "reads dated FWM calibration records"),
    ("tests/test_fwm_zeeman_cache.py", ARCHIVE, "reads dated FWM calibration records"),
    ("tests/test_fwm_gain_hotfix.py", ARCHIVE, "retired gain-hotfix replay of dated records (user 2026-11-09)"),
    ("tests/test_fwm_tpd_gain_hotfix.py", ARCHIVE, "retired gain-hotfix replay of dated records (user 2026-11-09)"),
    ("tests/test_fwm_docs_consistency.py", ARCHIVE, "analytic-reconstruction audit artifacts"),
    ("tests/test_publication_records.py", ARCHIVE, "edition manifest and archived paper"),
    ("tests/test_fwm_measured_gain_records.py", ARCHIVE, "2026-10-16 measured-gain evidence identity"),
    # Reusable pipelines kept in main until S5 moves them to gabes_lab/pipelines.
    ("analysis/rydberg_cell_heating/*", MAIN, "reusable heating workflow (top level only)"),
    ("analysis/squeezing/*.py", MAIN, "reusable squeezing scanners"),
    ("analysis/squeezing/low_pump_power/*.py", MAIN, "reusable squeezing scanner"),
    ("analysis/squeezing/variable_audit/*.py", MAIN, "reusable squeezing scanner"),
    ("analysis/**", ARCHIVE, "dated analysis records"),
    # Documents: user/developer docs stay; dated research and reports go.
    ("docs/ui_redesign/**", MAIN, "UI design hub"),
    ("docs/Userguide/**", MAIN, "user guide source"),
    ("docs/repo_restructure/**", MAIN, "split hub"),
    ("docs/papers/**", MAIN, "latest scientific editions (R4)"),
    ("docs/SABES/design_rules.md", MAIN, "SABES design rules"),
    ("docs/SABES/optical_table_ui_plan.md", MAIN, "SABES table UI plan"),
    ("docs/checklist.json", MAIN, "work registry"),
    ("docs/**", ARCHIVE, "dated research, reports and evidence"),
    ("output/**", ARCHIVE, "generated outputs"),
    ("archive/**", ARCHIVE, "retired artifacts and migration records"),
    # Product code, data, references and tooling.
    ("gabes/**", MAIN, "engine"),
    ("gabes_ui/**", MAIN, "UI package"),
    ("assets/**", MAIN, "UI assets"),
    ("sabes/**", MAIN, "SABES model and UI parts"),
    ("static/**", MAIN, "Streamlit static files"),
    (".streamlit/**", MAIN, "Streamlit config"),
    ("references/**", MAIN, "literature databases used by models and tests"),
    ("tools/**", MAIN, "developer tools"),
    ("tests/**", MAIN, "product tests"),
    (".claude/**", MAIN, "Claude Code project skills and preview launch config"),
    (".agents/**", MAIN, "agent skills"),
    (".impeccable/**", MAIN, "UI critique records"),
    ("*", MAIN, "root files"),
]

ARCHIVE_PREFIXES = sorted({p[:-3] for p, side, _ in RULES if side == ARCHIVE and p.endswith("/**")},
                          key=len, reverse=True)
PATH_TOKEN = re.compile(r"(?<![\w/.-])((?:analysis|docs|output|archive|tests/quantum|tools)/[\w ./()%,+-]*[\w)])")
IMPORT = re.compile(r"^\s*(?:from|import)\s+((?:analysis|tools|tests)(?:\.\w+)*)", re.M)
LINK = re.compile(r"(?:\]\(|href=\"|src=\")([^)\"#?\s]+)")


def _regex(pattern: str) -> re.Pattern:
    if pattern.endswith("/**"):
        return re.compile(re.escape(pattern[:-3]) + r"/.*\Z")
    out = ""
    for ch in pattern:
        out += "[^/]*" if ch == "*" else ("[^/]" if ch == "?" else re.escape(ch))
    return re.compile(out + r"\Z")


COMPILED = [(_regex(p), p, side, why) for p, side, why in RULES]


def classify(path: str) -> tuple[str, str]:
    for rx, pattern, side, _ in COMPILED:
        if rx.match(path):
            return side, pattern
    raise LookupError(path)


def tracked() -> list[str]:
    out = subprocess.run(["git", "-c", "core.quotepath=false", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def side_of_reference(ref: str, sides: dict[str, str]) -> str | None:
    ref = ref.strip().strip("`'\"").replace("%20", " ").lstrip("./")
    if not ref:
        return None
    if ref in sides:
        return sides[ref]
    prefix = ref.rstrip("/") + "/"
    under = {sides[p] for p in sides if p.startswith(prefix)}
    if under:
        return ARCHIVE if under == {ARCHIVE} else (MAIN if MAIN in under else None)
    try:
        return classify(ref)[0]
    except LookupError:
        return None


def module_side(module: str, sides: dict[str, str]) -> str | None:
    base = module.replace(".", "/")
    for candidate in (base + ".py", base + "/__init__.py"):
        if candidate in sides:
            return sides[candidate]
    return side_of_reference(base, sides)


def scan(files: list[str], sides: dict[str, str]) -> list[dict]:
    findings = []
    for path in files:
        if sides[path] != MAIN or not path.endswith((".py", ".md", ".html", ".json", ".toml", ".txt", ".ini")):
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if path.endswith(".py"):
            for m in IMPORT.finditer(text):
                if module_side(m.group(1), sides) == ARCHIVE:
                    findings.append(dict(file=path, kind="import", target=m.group(1),
                                         line=text.count("\n", 0, m.start()) + 1))
        refs = {(m.group(1), m.start()) for m in PATH_TOKEN.finditer(text)}
        if path.endswith((".md", ".html")):
            dirname = Path(path).parent
            for m in LINK.finditer(text):
                link = m.group(1)
                if "://" in link or link.startswith(("mailto:", "data:", "#")):
                    continue
                resolved = (dirname / link.replace("%20", " ")).as_posix()
                parts = []
                for part in resolved.split("/"):
                    if part == "..":
                        if parts:
                            parts.pop()
                    elif part not in ("", "."):
                        parts.append(part)
                refs.add(("/".join(parts), m.start()))
        for ref, pos in sorted(refs):
            if side_of_reference(ref, sides) == ARCHIVE:
                kind = "link" if path.endswith((".md", ".html")) else (
                    "data" if path.endswith(".json") else "string")
                findings.append(dict(file=path, kind=kind, target=ref, line=text.count("\n", 0, pos) + 1))
    unique = {(f["file"], f["kind"], f["target"], f["line"]): f for f in findings}
    return list(unique.values())


def report(args) -> int:
    files = tracked()
    sides, hits = {}, Counter()
    for path in files:
        side, pattern = classify(path)
        sides[path] = side
        hits[pattern] += 1
    unused = [p for p, *_ in RULES if hits[p] == 0 and p not in ("docs/papers/**",)]
    sizes = Counter()
    for path, side in sides.items():
        try:
            sizes[side] += (ROOT / path).stat().st_size
        except OSError:
            pass
    findings = scan(files, sides)
    by_kind = Counter(f["kind"] for f in findings)
    by_file = Counter(f["file"] for f in findings)
    print(f"tracked {len(files)}: main {sum(s == MAIN for s in sides.values())} "
          f"({sizes[MAIN] / 1e6:.1f} MB), archive {sum(s == ARCHIVE for s in sides.values())} "
          f"({sizes[ARCHIVE] / 1e6:.1f} MB)")
    if unused:
        print("rules matching nothing:", ", ".join(unused))
    print("main -> archive references:", dict(by_kind))
    for f, n in by_file.most_common(40):
        print(f"  {n:4d}  {f}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip(),
            rules=[dict(pattern=p, side=s, reason=w, files=hits[p]) for p, s, w in RULES],
            sides=sides, findings=sorted(findings, key=lambda f: (f["file"], f["line"]))),
            ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def checkout(args) -> int:
    files = tracked()
    main_files = [p for p in files if classify(p)[0] == MAIN]
    target = Path(args.dir).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    # A shared clone borrows this repository's objects without touching its
    # .git/worktrees (which a synced drive can hold locked).
    subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(ROOT), str(target)],
                   check=True)
    patterns = "\n".join("/" + p.replace("\\", "/") for p in main_files) + "\n"
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"], cwd=target,
                   input=patterns.encode("utf-8"), check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", head], cwd=target, check=True)
    print(f"main-only checkout: {target} at {head[:7]} ({len(main_files)} files)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--json")
    c = sub.add_parser("checkout")
    c.add_argument("dir")
    args = parser.parse_args(argv)
    return {"report": report, "checkout": checkout}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
