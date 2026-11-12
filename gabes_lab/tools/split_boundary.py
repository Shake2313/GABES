"""Repository-split boundary: which tracked files go to GABES main and which to
the private archive, and what in main still points at archive-side files.

docs/repo_restructure/README.md holds the plan and the user decisions; this file
is its machine-readable boundary (S2). Rules are ordered; the first match wins.

    python -m gabes_lab.tools.split_boundary report              # counts + violations
    python -m gabes_lab.tools.split_boundary report --json out.json
    python -m gabes_lab.tools.split_boundary checkout DIR        # main-only shared clone at HEAD
    python -m gabes_lab.tools.split_boundary links [--apply]     # rewrite main doc links to the archive
    python -m gabes_lab.tools.split_boundary history-paths REPO OUT   # filter-repo path list
    python -m gabes_lab.tools.split_boundary manifest ...        # Split Manifest (S3)

A pattern ending in "/**" matches everything below that folder; "*" never
crosses a "/".
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from gabes_lab.blocks.paths import REPO_ROOT

ROOT = REPO_ROOT
MAIN, ARCHIVE = "main", "archive"
# User decisions R1/R9 (2026-11-09): the private archive and the tag that
# pins every pre-split path.
ARCHIVE_REPO = "Shake2313/GABES-archive"
SPLIT_TAG = "pre-split-2026-11-09"
# Main docs whose archive paths are deliberate source citations, not links to follow.
CITATION_DOCS = ("docs/repo_restructure/", "tests/fixtures/", "docs/papers/")

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
    ("gabes_lab/**", MAIN, "development package: blocks, pipelines, tools (S5)"),
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


def resolve_link(doc: str, link: str) -> str | None:
    """Repository path of a relative link written in `doc`, or None for URLs/anchors."""
    if "://" in link or link.startswith(("mailto:", "data:", "#", "/")):
        return None
    resolved = (Path(doc).parent / link.replace("%20", " ")).as_posix()
    parts: list[str] = []
    for part in resolved.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts) or None


def archive_url(path: str, sides: dict[str, str]) -> str:
    kind = "blob" if path in sides else "tree"
    return f"https://github.com/{ARCHIVE_REPO}/{kind}/{SPLIT_TAG}/{quote(path, safe='/')}"


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
            for m in LINK.finditer(text):
                resolved = resolve_link(path, m.group(1))
                if resolved:
                    refs.add((resolved, m.start()))
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


def links(args) -> int:
    """Rewrite main-doc links that point at archive-side files to archive URLs."""
    files = tracked()
    sides = {p: classify(p)[0] for p in files}
    changed, dangling = Counter(), []
    for path in files:
        if sides[path] != MAIN or not path.endswith((".md", ".html")) or path.startswith(CITATION_DOCS):
            continue
        text = (ROOT / path).read_bytes().decode("utf-8")
        edits = []
        for m in LINK.finditer(text):
            target = resolve_link(path, m.group(1))
            if target and side_of_reference(target, sides) == ARCHIVE:
                if target not in sides and not any(p.startswith(target + "/") for p in sides):
                    dangling.append((path, target))  # local-only output; no archive URL exists
                    continue
                edits.append((m.start(1), m.end(1), archive_url(target, sides)))
        if not edits:
            continue
        changed[path] = len(edits)
        if args.apply:
            out = text
            for start, end, url in reversed(edits):
                out = out[:start] + url + out[end:]
            (ROOT / path).write_bytes(out.encode("utf-8"))
        elif args.verbose:
            for start, end, url in edits:
                print(f"  {path}: {text[start:end]} -> {url}")
    for path, n in changed.most_common():
        print(f"{n:4d}  {path}")
    print(f"{'rewrote' if args.apply else 'would rewrite'} {sum(changed.values())} links in {len(changed)} files")
    if dangling:
        print(f"{len(dangling)} links to untracked local outputs left unchanged:")
        for path, target in dangling:
            print(f"  {path}: {target}")
    return 0


def history_paths(args) -> int:
    """filter-repo path list: main by rule over the whole history, plus earlier names of HEAD main files."""
    repo = Path(args.repo)

    def git(*cmd):
        return subprocess.run(["git", "-c", "core.quotepath=false", *cmd], cwd=repo,
                              capture_output=True, check=True).stdout.decode("utf-8", "replace")

    head_main = {p for p in git("ls-tree", "-r", "--name-only", args.ref).splitlines()
                 if classify(p)[0] == MAIN}
    seen, renames = set(), collections.defaultdict(set)
    for block in git("log", args.ref, "-M", "--name-status", "--format=%x00%H").split("\0"):
        for line in [l for l in block.splitlines() if l.strip()][1:]:
            parts = line.split("\t")
            if parts[0][:1] in "RC":
                seen.update(parts[1:3])
                if parts[0][:1] == "R":
                    renames[parts[2]].add(parts[1])
            else:
                seen.add(parts[-1])
    by_rule = set()
    for p in seen:
        try:
            if classify(p)[0] == MAIN:
                by_rule.add(p)
        except LookupError:
            pass
    lineage, stack = set(), list(head_main)
    while stack:
        for old in renames.get(stack.pop(), ()):
            if old not in lineage and old not in head_main:
                lineage.add(old)
                stack.append(old)
    keep = sorted(by_rule | head_main | lineage)
    Path(args.out).write_text("".join(p + "\n" for p in keep), encoding="utf-8", newline="\n")
    print(json.dumps(dict(historical_paths=len(seen), head_main=len(head_main), kept=len(keep),
                          by_rule=len(by_rule), lineage_only=len(lineage - by_rule))))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(args) -> int:
    """Split Manifest: what went where, from which commit, and the old->new commit map."""
    def rev(repo, ref):
        return subprocess.run(["git", "rev-parse", ref], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()

    source_head = rev(ROOT, args.source_ref)
    files = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "--name-only", source_head],
                           cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").splitlines()
    sides = {p: classify(p)[0] for p in files}
    archive_list = sorted(p for p, s in sides.items() if s == ARCHIVE)
    main_list = sorted(p for p, s in sides.items() if s == MAIN)
    pairs = []
    for line in Path(args.commit_map).read_text(encoding="utf-8").splitlines()[1:]:
        old, new = line.split()
        pairs.append([old, None if set(new) == {"0"} else new])
    filter_repo = subprocess.run([sys.executable, "-m", "git_filter_repo", "--version"],
                                 capture_output=True, text=True).stdout.strip()
    record = dict(
        schema="gabes-split-manifest-v1",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hub="docs/repo_restructure/README.md",
        archive_repo=ARCHIVE_REPO, main_repo=args.main_repo_name, split_tag=SPLIT_TAG,
        source_head=source_head, main_head=rev(args.main_repo, "HEAD"),
        tool=dict(boundary="gabes_lab/tools/split_boundary.py", boundary_sha256=_sha256(Path(__file__)),
                  git_filter_repo=filter_repo),
        rules=[dict(pattern=p, side=s, reason=w) for p, s, w in RULES],
        filter_paths=dict(file=Path(args.paths).name, sha256=_sha256(Path(args.paths)),
                          count=len(Path(args.paths).read_text(encoding="utf-8").splitlines())),
        source_tree=dict(main_files=len(main_list), archive_files=len(archive_list),
                         main_list_sha256=hashlib.sha256("\n".join(main_list).encode()).hexdigest(),
                         archive_list_sha256=hashlib.sha256("\n".join(archive_list).encode()).hexdigest()),
        commit_map=dict(rewritten=sum(1 for _, n in pairs if n), dropped=sum(1 for _, n in pairs if not n),
                        pairs=pairs),
        provenance_rule=(f"Paths under archive-side folders cited by main code or data resolve in "
                         f"{ARCHIVE_REPO} at {SPLIT_TAG} or later."),
    )
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                              newline="\n")
    print(f"manifest: {args.out} ({record['commit_map']['rewritten']} rewritten, "
          f"{record['commit_map']['dropped']} dropped commits)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--json")
    c = sub.add_parser("checkout")
    c.add_argument("dir")
    lk = sub.add_parser("links")
    lk.add_argument("--apply", action="store_true")
    lk.add_argument("--verbose", action="store_true")
    hp = sub.add_parser("history-paths")
    hp.add_argument("repo")
    hp.add_argument("out")
    hp.add_argument("--ref", default="main")
    mf = sub.add_parser("manifest")
    mf.add_argument("--main-repo", required=True, help="filtered main repository (after filter-repo)")
    mf.add_argument("--commit-map", required=True, help="filter-repo commit-map file")
    mf.add_argument("--paths", required=True, help="path list given to filter-repo")
    mf.add_argument("--out", required=True)
    mf.add_argument("--source-ref", default="HEAD")
    mf.add_argument("--main-repo-name", default="Shake2313/GABES")
    args = parser.parse_args(argv)
    return {"report": report, "checkout": checkout, "links": links, "history-paths": history_paths,
            "manifest": manifest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
