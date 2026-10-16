# Third-party provenance

Original third-party ownership and notices remain applicable. This publication does not assign a blanket license to external code or model checkpoints.

| Dependency | Pinned identity | Publication treatment |
|---|---|---|
| I-JEPA encoder | `facebook/ijepa_vith14_1k`, revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e` | Checkpoint is external; identity retained in protocols |
| FAR historical reference | `https://github.com/sony/far.git`, commit `2963105c0f1db25214202d2c089cdf33041270f3` | Pristine checkout omitted; local adapters, experimental source snapshots and results retained |
| SPACE historical reference | `https://github.com/GilliaN-HJ/SPACE.git`, commit `5a073bcc303f006834bfb6ae0010c11747ded615` | Pristine checkout omitted; local adapters and records retained |
| Set Transformer components | Attribution and license in the frozen repair source folders | Original `SET_TRANSFORMER_LICENSE.txt` files retained |

Fetch upstream references only when reproducing that particular historical experiment. They are not needed for the CPU-only scalar verification or report rebuilding. No FAR/SPACE reproduction is restarted by this publication.
