---
name: gabes-ui-design
description: Front-end work in THIS GABES repository - the Streamlit app (streamlit_app.py), the gabes_ui package, its assets, and the SABES optical-table screen (gabes_ui/sabes). Use it before changing layout, styling, components, labels, plots shown in the app, or anything a user sees in GABES or SABES, including small UI fixes. Not for physics or scheme numerics in gabes/.
---

# GABES UI design

The app is scheme-driven: `streamlit_app.py` and `gabes_ui` draw whatever a
scheme declares (`param_schema`) and returns (`observables`). They hold no
physics. Keep it that way.

## Read first

1. `gabes_ui/DESIGN.md` - package map, rules, verification.
2. `docs/ui_redesign/README.md` - current state and work rules; then the
   current stage in `plan.md` and the newest `log.md` entry.
3. For the SABES screen: `docs/SABES/design_rules.md`.

## Before writing code

- Look for an existing building block in `gabes_ui` (tokens in `theme`, number
  formatting in `format`, unit parsing in `units`, ScrubField in `scrub`, rail
  assembly in `controls`, plot cards in `plotcard`, component runtime in
  `component_runtime`). Extend it with a parameter before adding a new one.
- Colours and fonts come from `gabes_ui.theme` tokens only. Asset files are
  read through `gabes_ui.paths` (`ASSETS`, `UI_ASSETS`, `asset_text`).
- A new on-screen element must replace at least one existing element or show
  only conditionally (decision D4).
- UI labels and units use unicode; matplotlib axis strings stay ASCII.

## After the change

- Run the UI tests that cover the change (`tests/test_ui_*.py`,
  `tests/test_sabes_*.py`, component tests) and look at the page through the
  `gabes` launch configuration. For visible changes run the ui_audit
  comparison described in `gabes_ui/DESIGN.md`.
- Record a design decision as a new entry in `docs/ui_redesign/decisions.md`
  and the session in `docs/ui_redesign/log.md` (newest first).
- The working tree is shared with other agents: commit only your own files.
