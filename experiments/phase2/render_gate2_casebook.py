"""Render archived Gate2 predictions and RGB; no encoder/model/experiment calls."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLES = ("benign_duplicate", "distant_style_duplicate")
BITS = ((0, 0), (0, 1), (1, 0), (1, 1))


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def render(run, out):
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    summary = read(run / "summary.json")
    if "summary" not in summary:
        raise ValueError("This casebook requires the completed reader evaluation")
    records = read(run / "data/records.json")
    meta = {r["id"]: r for r in records}
    decisions = read(run / "terminal_decisions.json")
    rows = read(run / "evaluation_rows.json")
    rowmap = {(r["id"], r["arm"]): r for r in rows}
    rowindex = {(r["id"], r["arm"]): i for i, r in enumerate(rows)}
    with np.load(run / "data/unique_rgb.npz") as f:
        rgb = f["rgb"]
    with np.load(run / "data/unique_features.npz") as f:
        z = f["normalized"]
    with np.load(run / "evaluation_predictions.npz") as f:
        predictions = f["predictions"]
    paired = defaultdict(dict)
    for d in decisions:
        paired[(d["family"], tuple(d["bits"]))][d["condition"]] = d
    eligible = []
    for key, pair in paired.items():
        b, f = (pair[s]["policies"]["visual_diversity"] for s in STYLES)
        if b["semantic_correct"] and b["feature_class_correct"] and not f["semantic_correct"] and not f["feature_class_correct"]:
            eligible.append(key)
    if not eligible:
        raise ValueError("No case satisfies the declared failure/control rule; do not substitute a different rule")
    selected = sorted(eligible)[0]
    pair = paired[selected]
    family = selected[0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})

    # Population context precedes selected examples.
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"width_ratios": [1, 1.35]})
    arms = ["AB", "AprimeB", "AAprime", "Ablank", "Bblank", "blank2"]
    labels = ["AB", "A′B", "AA′", "A+blank", "B+blank", "2 blanks"]
    values = [summary["summary"]["all"]["arms"][a]["feature_class_accuracy"]*100 for a in arms]
    ax[0].bar(labels, values, color=["#237a67"]*2+["#9ba7b5"]*4)
    for i, value in enumerate(values):
        ax[0].text(i, value+2, f"{value:.1f}%", ha="center", fontsize=10)
    ax[0].set(ylim=(0, 115), ylabel="Unique nearest-future feature class (%)", title="Reader can use both independent clues")
    ax[0].tick_params(axis="x", rotation=20)
    policies = ["fifo", "terminal_uniform", "visual_diversity", "reader_attention", "oracle"]
    x = np.arange(len(policies))
    for si, (style, color) in enumerate(zip(STYLES, ["#237a67", "#d18b42"])):
        s = summary["summary"][style]
        values = [s["policies"][p]["mse"] if p != "oracle" else s["oracle_mse"] for p in policies]
        ax[1].bar(x+(si-.5)*.35, values, .35, color=color, label="Normal copy" if si == 0 else "Changed copy style")
    ax[1].set_xticks(x, ["FIFO", "Uniform", "Diversity", "Attention", "Local oracle"], rotation=15)
    ax[1].set(ylabel="Mean normalized feature MSE (h=3,6)", title="Native terminal deletion: all 192 episodes")
    ax[1].legend(frameon=False, fontsize=10)
    fig.suptitle("Gate2 development pilot: qualified reader, remaining retention failures", fontsize=15)
    fig.text(.5, .01, "24 paired-style families · 48 quartets · 192 episodes · one reader seed · descriptive means, no confirmation claim", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(figures / "population.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(17, 6.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.55]})
    details = []
    role_name = {"Aprime": "A′", "A": "A", "B": "B"}
    for ri, style in enumerate(STYLES):
        d = pair[style]; r = meta[d["id"]]
        q = r["query"]; drop = d["choices"]["visual_diversity"]
        for j, t in enumerate(r["evidence"]):
            ax = axes[ri, j]
            ax.imshow(rgb[r["rgb_indices"][t]])
            axis = r["semantic_axes"][j]
            sign = r["bits"][0 if axis == "a" else 1]
            direction = ("right" if sign else "left") if axis == "a" else ("down" if sign else "up")
            ax.set_title(f"{role_name[r['scene']['roles'][j]]}: {direction}  (t={t})", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color("#b6423e" if j == drop else "#237a67"); spine.set_linewidth(3)
            ax.set_xlabel("Diversity deletes" if j == drop else "Diversity keeps", color="#b6423e" if j == drop else "#237a67", fontsize=10)
        truth_ax = axes[ri, 3]
        truth_ax.imshow(rgb[r["rgb_indices"][q+6]])
        truth_ax.set_title("Actual future, h=6", fontsize=11)
        nr = rowmap[(r["id"], "native_drop_"+str(drop))]
        sibling = next(x for x in records if x["quartet"] == r["quartet"] and x["bits"] == nr["predicted_bits"])
        label_ax = axes[ri, 4]
        label_ax.imshow(rgb[sibling["rgb_indices"][q+6]])
        label_ax.set_title("Nearest GT candidate\n(feature label only)", fontsize=10)
        for a in (truth_ax, label_ax):
            a.set_xticks([]); a.set_yticks([])
        truth_ax.set_xlabel("bits "+str(r["bits"]))
        label_ax.set_xlabel("bits "+str(nr["predicted_bits"]), color="#237a67" if nr["feature_class_correct"] else "#b6423e")
        bar = axes[ri, 5]
        names = ["Drop "+role_name[a] for a in r["scene"]["roles"]]
        bars = bar.barh(range(3), d["losses"], color=["#b6423e" if i not in d["semantic_correct_delete_set"] else "#237a67" for i in range(3)])
        bars[drop].set_edgecolor("#111827"); bars[drop].set_linewidth(2.3)
        bar.set_yticks(range(3), names); bar.invert_yaxis()
        bar.set_xlim(0, max(max(pair[s]["losses"]) for s in STYLES)*1.4)
        for i, value in enumerate(d["losses"]):
            bar.text(value+.005, i, f"{value:.4f}", va="center", fontsize=10)
        bar.set_title("Same parent: all 3 deletions", fontsize=10)
        bar.set_xlabel("Feature MSE; outline = diversity", fontsize=9)
        axes[ri, 0].text(-.05, 1.35, "SUCCESS CONTROL: normal A′" if ri == 0 else "FAILURE: only A′ style changes", transform=axes[ri, 0].transAxes, fontweight="bold", fontsize=12, color="#237a67" if ri == 0 else "#b6423e")
        cand = z[np.asarray(r["rgb_indices"])[r["evidence"]]].astype(np.float64)
        unit = cand/np.linalg.norm(cand, axis=1, keepdims=True)
        distance = 1-np.clip(unit@unit.T, -1, 1)
        details.append({"style": style, "id": r["id"], "bits": r["bits"], "roles": r["scene"]["roles"],
                        "diversity": d["policies"]["visual_diversity"], "attention": d["policies"]["reader_attention"],
                        "cosine_distances": distance.tolist(), "attention_weights": d["attention"], "native_losses": d["losses"]})
    fig.suptitle("Same scene, actions, clue meanings and future; only the duplicate view changes appearance", fontsize=15, y=.99)
    fig.text(.5, .015, "The fifth column displays an actual sibling target selected by a feature classifier. It is NOT a rendered model prediction.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .92), h_pad=3)
    fig.savefig(figures / "paired_case.png", dpi=180)
    plt.close(fig)

    # Verify indistinguishability after deleting the sole complementary cue.
    family_records = [r for r in records if r["family"] == family and r["condition"] == STYLES[1]]
    duplicated_axis = family_records[0]["scene"]["duplicate_axis"]
    singleton_axis = 1 if duplicated_axis == "a" else 0
    missing = next(i for i, role in enumerate(family_records[0]["scene"]["roles"]) if role == "B")
    collisions = []
    for first in family_records:
        if first["bits"][singleton_axis] != 0:
            continue
        flipped = first["bits"].copy(); flipped[singleton_axis] = 1
        second = next(r for r in family_records if r["bits"] == flipped)
        pa = predictions[rowindex[(first["id"], f"native_drop_{missing}")]]
        pb = predictions[rowindex[(second["id"], f"native_drop_{missing}")]]
        collisions.append({"bits_a": first["bits"], "bits_b": second["bits"],
                           "saved_predictions_identical": bool(np.array_equal(pa, pb)),
                           "maximum_prediction_difference": float(np.max(np.abs(pa-pb)))})
    selection = {"rule": "Among matched (family,bits) pairs with distant diversity semantic+feature-class failure and benign diversity semantic+feature-class success, select lexicographically first (family,bits), not maximum loss/regret.",
                 "eligible_episode_pairs": len(eligible), "eligible_families": len({k[0] for k in eligible}),
                 "all_matched_episode_pairs": len(paired), "selected_family": family, "selected_bits": list(selected[1]),
                 "details": details, "missing_B_prediction_collisions": collisions,
                 "source_hashes": {name: sha(run/name) for name in ("summary.json", "terminal_decisions.json", "evaluation_rows.json", "evaluation_predictions.npz", "data/unique_rgb.npz")},
                 "model_calls": 0, "new_experiment": False, "render_source_sha256": sha(Path(__file__))}
    (out / "case_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    table_rows = []
    for bits in BITS:
        for style in STYLES:
            d = paired[(family, bits)][style]; p = d["policies"]["visual_diversity"]
            pred = rowmap[(d["id"], "native_drop_"+str(p["drop"]))]["predicted_bits"]
            table_rows.append(f"<tr><td>{bits}</td><td>{'普通副本' if style == STYLES[0] else '改外观副本'}</td><td>{'互补两条' if p['semantic_correct'] else '重复同一条'}</td><td>{pred}</td><td>{p['mse']:.6f}</td><td>{min(d['losses']):.6f}</td></tr>")
    document = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>Gate2：重复外观与互补信息</title>
<style>body{{font-family:system-ui,sans-serif;margin:32px auto;max-width:1250px;padding:0 22px;color:#172333;line-height:1.7}}h1{{font-size:30px}}h2{{margin-top:34px;font-size:23px}}img{{width:100%;height:auto}}.note{{background:#f1f5f8;border-left:4px solid #58718b;padding:13px 18px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce2e8;text-align:left}}code{{font-size:12px}}.small{{font-size:13px;color:#506070}}</style>
<h1>这次确实丢掉了有用信息：重复外观挤掉互补线索</h1>
<p>Gate2 把两个独立方向分别藏在历史 cue 中：一个决定左右，一个决定上下。K=2 足以各留一条；候选有三张，其中两张讲的是同一个方向。A 与 A′ 是重复信息，B 是另一条独有信息。角色 A 不固定代表横向。</p>
<div class="note">全部 24 项 reader 检查通过；互补 AB 的特征分类为 192/192，A′B 为 191/192，分别翻转两个 cue 后预测也都转向对应 sibling（192/192）。因此这个终端任务里的保留差距，已有“预测器能利用正确证据”的支持。这里没有训练新保留方法，也没有独立确认方法收益。</div>
<img src="figures/population.png" alt="全部开发场景的reader与保留策略结果">
<p>普通副本时，diversity 保住两条不同信息的比例为 96/96；改变副本外观后降到 48/96。对应特征分类从 96/96 降到 72/96。注意：这 48 次信息丢失全部集中在<strong>重复的是纵向 cue</strong>的48条 episode；重复横向 cue 的48条没有该失败。这是明确但很窄的受控现象，不能概括成所有外观变化都会击败 diversity。</p>
<h2>一组配对案例：只有 A′ 的画法变了</h2>
<p>这对样本的真实答案都是“向左、向上”。三张 cue 的时间依次是 t=9、10、11；query 为15。上下文、动作、cue 含义与未来完全相同，只有第一张 A′ 从普通样式换成彩色大卡片。绿框表示 diversity 保留，红框表示删除。</p>
<img src="figures/paired_case.png" alt="失败与配对成功对照的实际观测、未来和三种删除损失">
<p>普通样式下，diversity 删除一张重复的“向上”，保住“向左＋向上”；改样式后，它保住外观差别大的两张“向上”，反而删除唯一的“向左”。这对样本的 MSE 从 <strong>{pair[STYLES[0]]['policies']['visual_diversity']['mse']:.6f}</strong> 变为 <strong>{pair[STYLES[1]]['policies']['visual_diversity']['mse']:.6f}</strong>；同一候选集合里另删一张重复 cue，MSE 可以回到 <strong>{min(pair[STYLES[1]]['losses']):.6f}</strong>。</p>
<p>Attention 的表现正好提醒我们不能据单例选赢家：它在普通样式中删 B、MSE {pair[STYLES[0]]['policies']['reader_attention']['mse']:.6f}；改样式后删 A′、MSE {pair[STYLES[1]]['policies']['reader_attention']['mse']:.6f}。两类 proxy 在不同外观下会犯不同错误。</p>
<p>普通样式的 diversity 虽然保住两条互补信息、分类也正确，仍不是精确 MSE 最优：保存较早或较晚的同内容副本，reader 的数值输出会有差别。因此要把“丢了独有信息”和“语义正确但读出误差略有不同”分开。</p>
<div class="note"><strong>图中的“最近 GT 候选”不是模型生成的视频帧。</strong>它展示预测特征最接近哪一个真实 sibling 未来，用于解释四分类标签。模型实际输出是特征向量。</div>
<h2>同一 family 的全部四种答案</h2>
<p>不能只看失败视频。下面保留这组所有四个 sibling：如果 B 被删掉，左右不同的两个样本会向 reader 提供相同剩余信息。归档预测核查显示，两个 y 分组内删除 B 后的预测逐元素完全相同，最大差异都是0；猜对其中一个并不表示恢复了丢失的方向。</p>
<table><tr><th>真实 bits（x,y）</th><th>样式</th><th>diversity 保留的信息</th><th>最近未来类别</th><th>MSE</th><th>局部 oracle MSE</th></tr>{''.join(table_rows)}</table>
<h2>这组例子支持什么，还没支持什么</h2>
<p>它支持“视觉差异最大”与“信息互补”不等价：有限槽位可能被同一事实的不同外观占据。它也支持在这个固定 reader、固定终端候选集合中存在实际预测损失，而不仅是 cue 标签被删。它尚未证明我们的 scorer 能学会解决，更没有证明这个任务覆盖真实视频的困难。</p>
<p>另外，早期 q−6 的静止预测仍不可靠：MSE 为 {summary['preuse_static_diagnostic']['by_condition']['all']['mse']:.6f}，复用上一帧的因果参照是0。当前证据限定在 query 时的终端选择，不能顺势宣称整个 early-time reader 或长轨迹标签已经校准。</p>
<h2>选择与来源</h2><p class="small">按明确失败类型筛出 {len(eligible)}/{len(paired)} 个配对 episode（{len({k[0] for k in eligible})} 个 family），再按 family ID、bits 字典序选第一组；没有按最大损失或最大 regret 选。完整总体为24个配对style family、48个quartet、192条episode，一个训练seed。该图册只重绘归档观测和计算结果，未调用模型。</p>
<p class="small">Family：<code>{family}</code>。<a href="case_selection.json">选择规则、具体ID、距离、attention、源文件哈希</a>。数据来自 A100 Gate2 开发 pilot；数值属于新任务自己的归一化特征空间，不与旧 Gate 的 MSE 绝对值横向比较。</p></html>"""
    (out / "index.html").write_text(document)
    print(json.dumps({"output": str(out), "selected_family": family, "eligible_pairs": len(eligible), "model_calls": 0}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    render(args.run, args.output)
