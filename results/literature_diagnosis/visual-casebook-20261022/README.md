# Gate real-RGB visual casebook

Open [the self-contained interactive gallery](index.html) or read [the report](../../../docs/GATE_VISUAL_CASEBOOK.md).

Three previously fixed examples show a content failure, an age interaction, and a success control. All pictures come from cached RGB observations or true futures; no predicted image was generated or decoded. Model forecasts appear only as existing feature MSE and paired-direction alpha values.

The broader context is 27 cue-harmed parents among 72 comparable parents, drawn from 192 previously observed development episodes. In a matched correct/wrong cue test, correct cue is worse in 21/27 original failures and better in 48/72 overall. These three selected examples are illustrations, not typicality or new-scene confirmation.

- `index.html`: offline player, time jumps, exact numeric comparisons and embedded downloads; no network dependencies.
- `figures/`: three static PNG/PDF evidence plates and three exact-palette real-trajectory GIFs.
- `frames/`: 45 native 96×96 source frames; PNG pixels match the original RGB bank exactly.
- `cases.json`, `rgb_frames.json`: timestamps, original frame-bank indices, source pixel hashes and fixed numerical rows.
- `audit.json`, `render_audit.json`: image provenance/roundtrip verification and desktop/mobile browser checks.
- `source/`, `manifest.json`: export source and file hashes.

The renderer does not run a model, encoder or simulator. The wrong-cue and hybrid conditions describe predictor input interventions; the displayed videos remain the actual original trajectories.
