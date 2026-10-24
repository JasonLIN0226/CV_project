# Grouped memory-failure casebook

Open `index.html` locally. The HTML embeds its images and data and works offline; GitHub's file view does not execute it. This package reorganizes existing development evidence, with **no new model, encoder, policy replay or simulator calls**. It does not alter the frozen confirmation results.

- **A: content failure + success control.** One input-condition button changes both cards. The primary 000→100 comparison changes content while keeping timestamp/action/filler/recent/reader fixed within each case. The two different scenes are not a controlled cross-scene pair. All RGB is true observation/future; feature predictions are numerical, never decoded images.
- **B: WR startup.** A slider follows the original time axis in archived cold/prefix traces. First cue copy deletion at t=8 differs from complete cue loss at t=10. A common four-frame prefix is a diagnostic control, not oracle access or an equal-total-cost policy win. It rescues all 62 early cues, but only 41 improve feature MSE; 21 worsen. The selected example is a post-hoc illustration, not a new performance estimate.
- **Auxiliary timestamp ablation.** Two content-fixed comparisons keep action fixed and change only timestamp. They show content/time interaction in one selected case, not a universal old-memory penalty.
- **Reading.** Worth Remembering → PaFu-KV → Gated DeltaNet (for the PM connection). The [Chinese guide](../../../docs/CASE_STUDY_READING_GUIDE_ZH.md) contains exact sections and case-specific questions.

The original [20261022 casebook](../visual-casebook-20261022/index.html), saved factorial rows, and WR startup results remain the sources. `audit.json` records source checks; `wr/provenance.json` records WR trace and exact RGB hashes; `manifest.json` covers this entire new package except itself.

`presentation_audit.json` distinguishes source-JavaScript checks from visual validation. All 4 reader modes and 18 WR times were evaluated with minimal DOM stubs; all 3 static scientific figures were visually inspected. **Browser visual/layout/interaction QA was not completed:** browser URL security policy rejected automatic `file://` navigation. No alternative browser surface or bypass was used.

To rebuild from the already published WR package:

```sh
.venv/bin/python experiments/phase2/render_memory_casebook.py
```

The WR exporter additionally needs the existing local RGB cache, whose hashes and location are in `wr/provenance.json`. Rebuilding the viewer does not execute that exporter or rerun any scientific experiment. The existing masked-cost frozen confirmation remains negative against its predeclared criterion.
