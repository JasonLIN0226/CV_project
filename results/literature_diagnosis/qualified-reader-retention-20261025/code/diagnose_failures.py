"""Post-hoc diagnosis (labelled as such) of the two failed predictions."""
import json, sys
from collections import Counter, defaultdict
import numpy as np

scored = json.load(open(sys.argv[1]))
term_records = {r["id"]: r for r in json.load(open(sys.argv[2]))}
agg_records = {r["id"]: r for r in json.load(open(sys.argv[3]))}
AGG = sys.argv[4]

for split, recs in (("existing_development", term_records), ("fresh_diagnostic", agg_records)):
    for reader in ("multihead", "item_attention"):
        rows = scored["splits"][split][reader]["rows"]
        print(f"\n######## {split} / {reader}")
        # ---------- D: medoid failures ----------
        tab = Counter(); tot = Counter()
        for r in rows:
            if r["task"] != "D":
                continue
            rec = recs[r["id"]]
            roles, axes, pay = rec["roles"], rec["semantic_axes"], rec["slot_payload"]
            dj = roles.index("D")
            choice = r["choices"]["medoid"]
            # which useful cue shares the decoy's axis?
            same_axis_cue = [j for j in range(3) if j != dj and axes[j] == axes[dj]]
            sa = same_axis_cue[0] if same_axis_cue else None
            match = (pay[sa] == pay[dj]) if sa is not None else None
            key = ("decoy axis matches a cue" if sa is not None else "no match",
                   "decoy bit == that cue's bit" if match else "decoy bit != cue bit")
            tot[key] += 1
            if choice != dj:
                tab[(key, "deleted same-axis cue" if choice == sa else "deleted other cue")] += 1
        print("D medoid failures by decoy relation:")
        for k in sorted(tot):
            print("  ", k, "n=", tot[k], {kk[1]: v for kk, v in tab.items() if kk[0] == k})
        # ---------- S: full-sensitivity failures ----------
        base = f"{AGG}/run/evaluation/{split}/{reader}/"
        qrows = json.load(open(base + "reader_qualification_rows.json"))
        q = defaultdict(dict)
        for x in qrows:
            q[x["id"]][x["arm"]] = x
        fs = Counter(); gap_ok = []; gap_bad = []
        orc = Counter()
        for r in rows:
            if r["task"] != "S":
                continue
            rec = recs[r["id"]]
            single = [j for j, t in enumerate(rec["tags"]) if rec["tags"].count(t) == 1][0]
            pair_correct = q[r["id"]]["native_%d" % single]["strict_correct"]
            full_correct = q[r["id"]]["full"]["strict_correct"]
            lost = r["full_sensitivity_lost"] > 0
            fs[(pair_correct, full_correct, lost)] += 1
            # semantic margin of the full forecast: distance to wrong prototype - to right prototype
            d = q[r["id"]]["full"]["distances"]; t = q[r["id"]]["full"]["target_class"]
            margin = d[1 - t] - d[t]
            sens = np.array(r["sensitivity"])
            (gap_bad if lost else gap_ok).append(margin)
            # oracle losses: is the pair forecast class-correct?
            orc[(pair_correct, r["oracle_lost"] > 0)] += 1
        print("S full-sensitivity (pair K2 correct, full K3 correct, lost):", dict(sorted(fs.items())))
        print("S full-forecast semantic margin: kept-correct median %.4f | lost median %.4f" % (np.median(gap_ok), np.median(gap_bad)))
        print("S oracle (pair K2 correct, oracle lost):", dict(sorted(orc.items())))
        # scale: sensitivity values vs prototype separation
        sens_pair = []; sens_mixed = []; sep = []
        for r in rows:
            if r["task"] != "S":
                continue
            rec = recs[r["id"]]
            single = [j for j, t in enumerate(rec["tags"]) if rec["tags"].count(t) == 1][0]
            s = r["sensitivity"]
            sens_pair.append(s[single]); sens_mixed += [s[j] for j in range(3) if j != single]
            d = q[r["id"]]["full"]["distances"]; sep.append(abs(d[0] - d[1]))
        print("S sensitivity: delete-singleton median %.5f, delete-pair-member median %.5f; |d0-d1| of full median %.5f"
              % (np.median(sens_pair), np.median(sens_mixed), np.median(sep)))
