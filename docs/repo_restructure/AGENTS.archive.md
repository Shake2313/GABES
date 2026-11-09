# AGENTS.md (GABES-archive)

⟦AI⟧ Draft for the archive repository root, to install at S4 of the repository
split (docs/repo_restructure/README.md); the Grand Challenge parts need Codex
review. Instructions for every coding agent here (Codex, Claude Code, others);
`CLAUDE.md` only imports it; `docs/grand_challenge/AGENTS.md` adds GC rules.

## What this repository is

The private research archive of GABES: Grand Challenge (GC) work and evidence,
dated analysis records, reports, daily reviews and generated outputs, with the
full history of the project before the split. The engine, UI, product tests and
user documentation live in the public repository `Shake2313/GABES` (main).

- **Engine:** `engine/` is a git submodule pinned to one GABES main commit.
  Pytest runs with `pythonpath = engine`, so `import gabes` resolves there.
  Bump the pin in its own commit, name the main commit and why, and run the GC
  checks the bump affects.
- **Pre-split evidence:** everything recorded before the split resolves at tag
  `pre-split-2026-11-09`, where the original paths and engine exist. Re-verify
  sealed evidence there; archive HEAD need not run pre-split validators
  unchanged (user decision R8, 2026-11-09).
- **New evidence** records provenance as `{repo, commit, path}` for every
  source it depends on, main or archive.

## Where things live

- **Grand Challenge research turns** → `docs/grand_challenge/AGENTS.md`, then
  `.agents/skills/grand-challenge-head-manager/SKILL.md` and its durable
  hypermemory, then `blueprint.md` for the living master design/current card and
  `NEXT_SESSION.md`. See **Grand Challenge Head Manager** below.
- **Lean in Grand Challenge** → `docs/grand_challenge/lean_workflow.md`: an evolving
  working default for selected formalization tasks. Propose better equivalent
  methods to the user, revise the guide within the authorized scope, and append its
  bottom revision history. Lean is not a blanket requirement for GC work.
- **Scientific publication policy** → `docs/grand_challenge/publication_policy.md`;
  current artifacts → `docs/grand_challenge/current_publications.json`.
  Squeezing report owns experiment agreement/frontier exploration; analytic
  reconstruction owns theory/precision; quotient structure owns the explicit
  logical and order-theoretic organization. Each latest document must stand
  alone. Preserve substantive content when revising; document every
  omission/correction in Grand Challenge coverage records. Development logs and
  version differences belong only in `docs/grand_challenge/`, not the three
  scientific documents. This policy continues after Grand Challenge completion.
  After publishing a new edition here, replace its byte copy in main
  `docs/papers/` in one main commit.
- **Daily and weekly reviews** → `docs/daily_report/`, written by the
  `gabes-weekly-review` automation (`.agents/skills/gabes-weekly-review/`), which
  reviews the engine at the pinned main commit.
- **Dated analysis records** → `analysis/<topic>_<date>/`; generated outputs →
  `output/`. Results too large for GitHub (over 100 MB) are Release assets with a
  sha256 record beside their path (example:
  `analysis/squeezing/operating_point_20261014/coarse.jsonl.external.json`).
- **Repository split record** → `docs/repo_restructure/` and the Split Manifest.

## Sealed records

Sealed records stay intact: reports, cache records, declarations, source ZIPs,
archived controllers, figures, logs and validators keep their bytes and paths.
New results need distinct outputs; failed gates stay recorded. Do not edit
sealed scripts or receipts to route them through current code; new scripts use
the current engine from the start.

## Change-scoped validation

User policy, 2026-10-23: reduce costly regression runs for minor patches, omit
unrelated GC checks, and consolidate validation across development turns.

- ⟦AI⟧ Choose checks from the changed behavior, its inputs and the pinned engine.
  GC suites/audits and expensive research campaigns are separate scopes; omit
  them when their model, engine pin, inputs, validators and evidence contracts
  are unaffected. An engine-pin bump that touches shared solvers may require GC
  checks. Existing registered scientific acceptance criteria remain intact when
  claiming their milestones; routine maintenance does not rerun sealed campaigns.
- ⟦AI⟧ Report commands, outcomes and intentional omissions briefly. Focused
  success is not a full-suite PASS. Record deferred required validation before
  handoff.

## Conventions

- Instructions and reviews between agents: English. Final user reports and summary
  Markdown: Korean, `caveman` full style. Drop filler; keep equations,
  conditions, errors and validation scope. Existing records need no bulk compression.
- UI labels, units and markdown tables use unicode (°C, µT, µW, Γ, Ω, 2π).
  Matplotlib axis strings stay ASCII.

## Internal Text Provenance

User policy, 2026-10-13: keep supplementary verification text brief. Separate
direct user requests and registered project goals from AI-added interpretations,
recommendations and validation notes. Prefix those additions with `⟦AI⟧`, or
place them in one section whose heading has that marker. Preserve the marker,
source and adoption status when summarizing or transferring them into memory.
An unmarked historical sentence does not establish human authorship.

Keep essential mathematical assumptions, units and result scope with the claim.
Consolidate repeated limitations once.
Details: [internal provenance policy](docs/grand_challenge/internal_provenance.md).

## Text encoding on Korean Windows

PowerShell's default code page here is CP949, not UTF-8, so non-ASCII text breaks.

- Edit text with Python, not PowerShell, and pass `encoding="utf-8"` to every
  read and write.
- Do not pass Korean or other non-ASCII text as a shell argument or an inline
  string. Write it to a file first and read that file from Python.
- Edit Office files (PPTX/DOCX/XLSX) only with python-pptx, python-docx or
  openpyxl, never through shell text pipes.
- Prefer bash to PowerShell. If PowerShell is unavoidable, first run
  `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8; chcp 65001`.

## Branch Policy

- Work on the current branch by default.
- Do not create, switch to, or open new git/Codex branches unless the user
  explicitly asks for it, or there is a clear technical blocker that requires an
  isolated branch.
- If a temporary branch is truly necessary, explain why, keep it narrowly
  scoped, commit or stash the work before leaving it, and close/delete the
  branch once the work is merged or no longer needed.

## Grand Challenge Head Manager

For every Grand Challenge session, read and apply
[Grand Challenge Head Manager](.agents/skills/grand-challenge-head-manager/SKILL.md)
at start, relevant checkpoints and closure. Use its persistent linked memory
to check long-term direction, closed work, failed approaches and repetition.
Apply the latest user request; old AI proposals are working guidance, not new
approval requirements. The same blueprint remains the master design. End every
research turn with a principal-investigator review; update the same blueprint,
close bounded work packages, and state the single next question. Sealed failures
and official milestone criteria remain intact.
