# NanoReason dashboard

Updated 18 October 2026. Scope: the original small-language-model application.

## Direction

The requested design-taste-frontend, gpt-taste, and minimalist-ui guidance is
adapted to a working data application. Marketing-only sections, stock imagery,
scroll-pinned storytelling and framework migration are omitted.
The existing Solve / Verify / Runs navigation, brand mark, backend routes and
model behavior remain intact. Design dials: variance 5, motion 3, density 5.

The visual system uses self-hosted Geist, warm neutral surfaces, charcoal
controls, a muted olive accent, and semantic green/red/amber feedback. One
12px panel radius and 6px control radius establish the shape hierarchy.
The desktop sidebar becomes horizontal tabs below 800px; inputs and outputs
collapse into one column. Tables scroll within their own region.

The design selection used Python Random(18) among two compatible workspace
compositions and selected the asymmetric workspace. This only chose visual
layout; measured data and statistical decisions are never randomized.

## Screens and behavior

- Solve: adjacent problem composer and streamed response. Three ordinary math
  examples optionally fill known answers. Explicit model/demo/unavailable
  status, copy response, Ctrl/Cmd+Enter, and a final arithmetic/reward report.
  No scripted trap prompt is offered as a real-model example.
- Verify: editable reasoning with the original deliberately incorrect sample;
  arithmetic checks and answer correctness are distinguished.
  Word-diversity feedback is a heuristic, not a claim of model collapse.
- Runs: charts and percentages derived from the runs API, expandable artifact
  provenance, JSON downloads, Wilson intervals, percentage-point deltas and
  the actual paired test. It does not imply the trained model beats the base.

Changing inputs invalidates stale reports. Active requests lock their inputs,
prevent duplicate requests, preserve partial output on stream failure and
restore controls for retry. Empty data and failed requests are separate states.

Theme defaults to the operating system and can be toggled and saved locally.
Tabs support arrows, Home/End and URL fragments. GSAP performs short panel and
result reveals; reduced-motion disables these effects. Assets are locally
served with pinned versions, license notices and hashes under static/vendor.

## Validation

On 18 October 2026:
- Existing Python regression suite: 143 passed.
- Browser script: 13 checks passed using Playwright 1.62.1 / Chrome.
- All three screens fit 1440, 1024, 768, 390 and 320px viewport widths.
- Real GPU inference through the UI: original GRPO adapter returned 35 for
  20 + 15, with a complete response and arithmetic analysis in 3.5 seconds.
  This is an integration smoke test, not a new benchmark.
- axe-core 4.10.3 reported zero automated WCAG A/AA violations on all three
  screens in both light and dark mode. This is not an accessibility certification
  and does not replace assistive-technology testing.
- No uncaught browser JavaScript errors.

Local screenshots and machine-readable checks are under
artifacts/slm/dashboard/. These artifacts are gitignored. No training,
checkpoint changes, public publishing or new benchmark run was performed.

## Reproduce browser checks

Start the local app with the existing SLM runbook. Make Playwright 1.62.1
available to Node (the desktop runtime already includes it) and have Chrome
installed, then run from the repository root:

    node tests/dashboard.browser.cjs
    node tests/dashboard.browser.cjs --live

The default uses actual verification and evaluation APIs and intercepts selected
requests to test failure/empty states; it does not invoke model generation.
--live adds one real inference against the already-running model. The test
expects the project's saved base/SFT/GRPO evaluation artifacts to be available.
Do not launch a second GPU model for these checks.

Formatting was performed with Prettier 3.6.2. Browser tooling and accessibility
packages used for local checks are kept in ignored artifacts, not shipped
to dashboard users.
