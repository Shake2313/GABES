# AGENTS.md (GABES-archive)

⟦AI⟧ Draft for the archive repository root, to install at S4 of the repository
split. Instructions for coding agents in the archive. `CLAUDE.md` imports
these rules. `docs/grand_challenge/AGENTS.md` adds GC rules.

## What this repository is

The private research archive of GABES: Grand Challenge (GC) work and evidence,
dated analysis records, reports, daily reviews and generated outputs, with the
full history of the project before the split. The engine, UI, product tests and
user documentation live in the public repository `Shake2313/GABES` (main).

- **Engine.** `engine/` is a Git submodule pinned to the main commit
  recorded in the Split Manifest at S4. The archive pytest configuration
  sets `pythonpath = engine`. Check that `gabes` imports and validators
  that hash live engine sources both resolve to this checkout. Record
  the archive revision and engine commit with validation. Bump the pin
  in its own commit and run the affected GC checks.
- **Historical replay (R8).** `pre-split-2026-11-09` (`0c95124`) records the
  S1 baseline. Use the revision and layout recorded for each historical
  result. Some validators also require the original checkout path.
- **Verification from a different checkout path.** Use the archive
  revision that includes the portable reader (introduced in `473017f`)
  to verify the same pinned pilot-shadow inputs. From the archive root,
  run `python -m analysis.grand_challenge.variance_reduction_ledger_reader`.
- **New evidence** records provenance as `{repo, commit, path}` for every
  source it depends on, main or archive.

## Where things live

- **Grand Challenge registry** → `docs/grand_challenge/checklist.json` (the
  `grand_challenges` section and GC items; user decision R12). The product
  registry is `engine/docs/checklist.json`, read-only here at the pinned commit.
- **Grand Challenge research turns** → `docs/grand_challenge/AGENTS.md`, then
  `.agents/skills/grand-challenge-head-manager/SKILL.md` and its durable
  hypermemory, then `blueprint.md` for the living master design/current card and
  `NEXT_SESSION.md`. See **Grand Challenge Head Manager** below.
- **GC registry (R12)** → `docs/grand_challenge/checklist.json` owns GC
  milestones and tasks. Use `engine/docs/checklist.json` for product
  items at the pinned main revision.
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
  `docs/papers/` and update the edition, archive reference and hash in
  `docs/papers/README.md` in the same main commit.
- **Daily and weekly reviews** → `docs/daily_report/`. Move the existing
  `gabes-weekly-review` automation after verifying the archive checkout
  and pinned `engine/`. Preserve 05:00 Asia/Seoul, the weekday assignments
  and model settings. The archive holds the review workflow at
  `.agents/skills/gabes-weekly-review/`.
- **Dated analysis records** → `analysis/<topic>_<date>/`; generated outputs →
  `output/`. Results too large for GitHub (over 100 MB) are Release assets with a
  sha256 record beside their path (example:
  `analysis/squeezing/operating_point_20261014/coarse.jsonl.external.json`).
- **Repository split record** → the Split Manifest and the main split hub
  at `engine/docs/repo_restructure/README.md`.

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
