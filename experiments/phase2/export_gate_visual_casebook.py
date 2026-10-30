"""Export real cached RGB case studies and existing feature-space diagnostics.

No model, encoder, or simulator is run. RGB frames are indexed from the verified
frame bank; future pictures are ground truth, never decoded model predictions.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
DISPLAY_ARMS = ((0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 1))
ARM_EN = ('Original noncue', 'Correct content only', 'Original correct cue', 'Matched wrong cue')
ARM_ZH = ('原始 noncue', '仅换正确内容', '原始正确 cue', '同时间/动作的错误 cue')
CASES = {
    'content_harm': {
        'title': '01 · 内容失败：看见线索，方向仍然错',
        'english': 'Content failure: correct cue, wrong feature-space direction',
        'tag': 'ID · 内容伤害最大的既有失败案例',
        'note': '即使保持 noncue 的时间和动作，只换成正确 cue 内容，误差仍上升。实际正确 cue 获得很高 attention，却比同条件的错误 cue 更差。',
    },
    'age_contribution': {
        'title': '02 · 年龄参与：内容与时间发生交互',
        'english': 'Age interaction: timing amplifies this content failure',
        'tag': 'ID · 年龄贡献最大的既有失败案例',
        'note': '这个案例的年龄项确实放大错误；但仅换内容也会变差，不能全部归因于 cue 更旧。',
    },
    'correct_cue_success': {
        'title': '03 · 成功对照：同一模型也能正确使用 cue',
        'english': 'Success control: a correct cue helps after a long gap',
        'tag': 'Long gap · 正确 cue 优于错误 cue 的成功对照',
        'note': '正确 cue 降低误差，换成同时间、同动作的错误 cue 后误差明显增加。失败模式并非所有场景都存在。',
    },
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def uri(path, kind): return 'data:'+kind+';base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')


def verified(data, relative, manifest):
    path = (data/relative).resolve()
    if not path.is_relative_to(data.resolve()) or sha(path) != manifest['artifact_sha256'][relative]:
        raise ValueError('Fresh artifact hash/path mismatch: '+relative)
    return path


def select_rows(card):
    keyed = {(r['content'], r['age'], r['past_action']): r for r in card['arms']}
    if len(keyed) != 12: raise ValueError('Incomplete fixed factorial case')
    return [keyed[arm] for arm in DISPLAY_ARMS]


def make_figure(path, kind, card, original, sibling, rgb, selected):
    q, cue = card['query'], card['cue_time']
    times = [cue, q, q+3, q+6]
    fig = plt.figure(figsize=(13.2, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(3, 4, height_ratios=(1, 1, 1.35), left=.11, right=.98, top=.87,
                           bottom=.13, wspace=.12, hspace=.40)
    for row, record in enumerate((original, sibling)):
        for col, t in enumerate(times):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(rgb[record['rgb_indices'][t]], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(('Historical cue', 'Query: same RGB', 'TRUE future h3', 'TRUE future h6')[col]+f'\nt={t}', fontsize=10)
            if col == 0:
                ax.set_ylabel('Case truth' if row == 0 else 'Sibling truth', fontsize=10, labelpad=14)
            for spine in ax.spines.values(): spine.set_color('#adb6c0')
    mse = fig.add_subplot(grid[2, :2]); alpha = fig.add_subplot(grid[2, 2:])
    colors = ['#7b8794', '#b38a3e', '#3d6f9d', '#9a70a0']
    positions = np.arange(4)
    mse.barh(positions, [r['mse'] for r in selected], color=colors, height=.62)
    for i, r in enumerate(selected): mse.text(r['mse']+.003, i, f"{r['mse']:.6f}", va='center', fontsize=9)
    mse.set_yticks(positions, ARM_EN, fontsize=9); mse.invert_yaxis()
    mse.set_xlim(0, max(r['mse'] for r in selected)*1.30)
    mse.set_xlabel('Feature MSE: mean of h3 and h6 (lower is better)', fontsize=9)
    mse.xaxis.set_major_locator(MaxNLocator(4))
    for i, r in enumerate(selected):
        alpha.plot(r['contrast_amplitude'], [i-.1, i+.1], color=colors[i], lw=1.3)
        alpha.scatter(r['contrast_amplitude'][0], i-.1, color=colors[i], marker='o', s=30)
        alpha.scatter(r['contrast_amplitude'][1], i+.1, color=colors[i], marker='s', s=30)
    alpha.axvline(0, color='#7a8793', lw=1)
    alpha.axvline(1, color='#467c65', lw=1, ls='--')
    alpha.axvline(-1, color='#ad655c', lw=1, ls='--')
    alpha.set_yticks(positions, ['']*4); alpha.invert_yaxis()
    alpha.set_xlabel('Feature-space α: −1 sibling, 0 midpoint, +1 case truth\nCircle=h3; square=h6. No image decoding.', fontsize=9)
    alpha.xaxis.set_major_locator(MaxNLocator(5))
    for ax in (mse, alpha): ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(CASES[kind]['english'], x=.11, ha='left', y=.98, fontsize=15, weight='semibold')
    fig.text(.11, .925, f"Episode {card['episode_id']}  |  shared filler t={card['filler_time']}  |  query t={q}", fontsize=10, color='#53606c')
    fig.text(.11, .035, 'All pictures are exact cached RGB observations / true futures. The tables and plots use existing frozen predictions.\n'
             'Only three preselected examples: 27/72 comparable parents were cue-harmed. Wrong-cue content tests and hybrid inputs are evaluator-only.',
             fontsize=9, color='#53606c')
    fig.savefig(path, dpi=180); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)


def make_gif(path, card, original, sibling, rgb):
    scale = 3; side = 96*scale; gap = 20
    width, height = 2*side+3*gap, side+112
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 14)
    frames, durations = [], []
    for t in range(original['frames']):
        image = Image.new('RGB', (width, height), (248, 250, 252)); draw = ImageDraw.Draw(image); draw.fontmode = '1'
        stage = 'OBSERVED HISTORY' if t < card['query'] else 'QUERY' if t == card['query'] else 'TRUE FUTURE - evaluation only'
        draw.text((gap, 10), f't={t}  |  {stage}', font=font, fill=(25, 36, 49))
        draw.text((gap, 40), 'Case: real RGB', font=small, fill=(25, 36, 49))
        draw.text((2*gap+side, 40), 'Sibling: real RGB', font=small, fill=(25, 36, 49))
        for column, record in enumerate((original, sibling)):
            frame = Image.fromarray(rgb[record['rgb_indices'][t]]).resize((side, side), Image.Resampling.NEAREST)
            image.paste(frame, (gap+column*(side+gap), 64))
        draw.text((gap, 64+side+12), 'Ground truth only. No predicted or generated video frames.', font=small, fill=(25, 36, 49))
        frames.append(image)
        durations.append(800 if t == card['cue_time'] else 650 if t == card['query'] else 500 if t in card['target_times'] else 110)
    colors = np.unique(np.concatenate([np.asarray(f).reshape(-1, 3) for f in frames]), axis=0)
    if len(colors) > 256: raise AssertionError('GIF exact palette would lose RGB colors')
    palette = np.zeros((256, 3), np.uint8); palette[:len(colors)] = colors
    palette_image = Image.new('P', (1, 1)); palette_image.putpalette(palette.reshape(-1).tolist())
    indexed = [f.quantize(palette=palette_image, dither=Image.Dither.NONE) for f in frames]
    for original_frame, encoded in zip(frames, indexed):
        if not np.array_equal(np.asarray(original_frame), np.asarray(encoded.convert('RGB'))):
            raise AssertionError('GIF palette changed an RGB value')
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations, loop=0, disposal=2, optimize=False)
    with Image.open(path) as animation:
        decoded = [np.asarray(f.convert('RGB')).copy() for f in ImageSequence.Iterator(animation)]
    if len(decoded) != len(frames) or any(not np.array_equal(a, np.asarray(b)) for a, b in zip(decoded, frames)):
        raise AssertionError('GIF roundtrip changed a complete rendered frame')
    return {'frames': len(frames), 'palette_colors': len(colors), 'full_rgb_roundtrip_exact': True}


def make_html(output, cards, frame_uri):
    sections = []
    payload = []
    for kind, card in cards.items():
        source = CASES[kind]; selected = card['display_rows']; q = card['query']
        table = []
        for label, r in zip(ARM_ZH, selected):
            content_source = f"该案例 cue t={card['cue_time']}" if r['content'] == 1 else f"sibling cue t={card['cue_time']}" if r['content'] == 2 else f"noncue t={card['noncue_time']}"
            clock = card['cue_time'] if r['age'] else card['noncue_time']
            action_start = card['cue_time'] if r['past_action'] else card['noncue_time']
            table.append(f'<tr><th>{label}</th><td>{content_source}<small>年龄按 t={clock}；动作从 t={action_start} 累积</small></td>'
                         f'<td class="num">{r["mse"]:.6f}</td><td class="num">{r["contrast_amplitude"][0]:+.4f}</td>'
                         f'<td class="num">{r["contrast_amplitude"][1]:+.4f}</td></tr>')
        figure_uri = uri(output/card['figure_png'], 'image/png')
        gif_uri = uri(output/card['animation_gif'], 'image/gif')
        sections.append(f'''<section class="case" id="{kind}"><div class="case-head"><p class="tag">{source['tag']}</p>
<h2>{source['title']}</h2><p>{source['note']}</p><code>{card['episode_id']}</code></div>
<div class="player" data-case="{kind}"><div class="play-top"><strong class="clock"></strong><span class="stage"></span></div>
<div class="paired"><figure><figcaption>该案例的真实轨迹</figcaption><img class="actual" alt="该案例真实 RGB 帧"></figure>
<figure><figcaption>相反 cue 的 sibling 真实轨迹</figcaption><img class="sibling" alt="配对场景真实 RGB 帧"></figure></div>
<div class="controls"><button class="play">播放</button><input class="slider" aria-label="{source['title']} 时间" type="range" min="0" max="{card['frames']-1}" value="{card['cue_time']}"></div>
<div class="jumps"><button data-time="{card['cue_time']}">线索 t={card['cue_time']}</button><button data-time="{q}">query t={q}</button>
<button data-time="{q+1}">首次分叉 t={q+1}</button><button data-time="{q+3}">h3</button><button data-time="{q+6}">h6</button></div>
<p class="caption">顶部斜线是历史线索；两个 recent window 的 RGB 相同。query 之后才出现真实轨道分叉。当前第一动作 a₀={card['first_query_action']:+.6f}；这三例均为该案例上轨、sibling 下轨。</p></div>
<div class="table-wrap"><table><thead><tr><th>预测器输入条件</th><th>交换槽内容 / 时序</th><th>MSE ↓</th><th>α h3</th><th>α h6</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p class="caption">共享 filler t={card['filler_time']}、recent context 和未来 controls 全部固定。表中的混合输入只交换预测器的输入槽；上方播放器始终展示原始真实轨迹，没有合成混合视频。</p>
<details><summary>静态证据图与可下载动画</summary><img class="evidence" src="{figure_uri}" alt="真实 RGB 和既有预测数值的静态证据图">
<p class="downloads"><a download="{kind}.png" href="{figure_uri}">下载 PNG</a><a download="{kind}.gif" href="{gif_uri}">下载真实轨迹 GIF</a></p></details></section>''')
        payload.append({k: card[k] for k in ('case', 'query', 'cue_time', 'frames', 'actual_rgb_indices', 'sibling_rgb_indices')})
    data_json = json.dumps({'cases': payload, 'frames': frame_uri}, ensure_ascii=False).replace('</', '<\\/')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gate 真实 RGB 案例画廊</title><style>
:root{color-scheme:light;--ink:#1d2b3b;--muted:#5d6b79;--blue:#386a93;--line:#dae2e9;--bg:#f3f6f8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1100px;margin:auto;padding:40px 26px 70px}h1{font-size:36px;line-height:1.3;letter-spacing:-.7px;margin:12px 0 20px}h2{font-size:24px;line-height:1.4;margin:8px 0 12px}p{margin:10px 0}code{font:12px/1.5 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.eyebrow,.tag{font-size:13px;color:var(--blue);letter-spacing:.3px;font-weight:650}.intro{max-width:900px}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.stat{background:white;border:1px solid var(--line);border-radius:12px;padding:12px 17px;flex:1;min-width:180px}.stat strong{font-size:24px;display:block}.stat span{font-size:13px;color:var(--muted)}nav{display:flex;gap:12px;flex-wrap:wrap;margin:23px 0}a{color:var(--blue);text-underline-offset:3px}.explain{background:#eaf1f6;border-left:4px solid var(--blue);padding:15px 20px;border-radius:0 10px 10px 0;margin:24px 0}.case{background:white;border:1px solid var(--line);border-radius:16px;margin:34px 0;padding:26px;box-shadow:0 5px 22px #182e4310}.case-head{margin-bottom:20px}.player{background:#f6f8fa;border:1px solid var(--line);border-radius:12px;padding:18px;margin:18px 0 24px}.play-top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px}.clock{font-variant-numeric:tabular-nums}.stage{font-size:13px;background:#e5edf5;color:#355c7b;border-radius:20px;padding:3px 10px}.paired{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:720px;margin:0 auto}.paired figure{margin:0;text-align:center}.paired figcaption{font-size:13px;color:var(--muted);margin:4px 0 10px}.paired img{image-rendering:pixelated;image-rendering:crisp-edges;display:block;width:100%;aspect-ratio:1;background:#181d26;border-radius:3px}.controls{display:flex;gap:14px;align-items:center;margin-top:17px}.slider{flex:1;accent-color:var(--blue);min-width:0}button{font:inherit;font-size:13px;background:white;border:1px solid #c4d1dc;border-radius:7px;color:var(--ink);padding:7px 12px;cursor:pointer}button:hover{background:#e9f0f5}button.play{min-width:62px;background:var(--blue);color:white}.jumps{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px}.caption{font-size:13px;color:var(--muted);line-height:1.6}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:690px;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:11px 12px;text-align:left;vertical-align:middle}thead th{font-size:12px;background:#f0f4f7;color:#526271}tbody th{font-weight:600;white-space:nowrap}td small{display:block;color:var(--muted);font-size:12px}.num{font-family:ui-monospace,SFMono-Regular,monospace;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}details{border-top:1px solid var(--line);margin-top:18px;padding-top:13px}summary{cursor:pointer;color:var(--blue);font-size:14px}.evidence{width:100%;margin-top:16px;display:block}.downloads{display:flex;gap:22px;font-size:14px}footer{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:24px}@media(max-width:620px){main{padding:24px 13px 40px}h1{font-size:28px}h2{font-size:21px}.case{padding:17px 13px}.player{padding:12px}.paired{gap:12px}.paired figcaption{font-size:11px}.stage{font-size:11px;padding:2px 7px}.stats{gap:8px}.stat{min-width:130px;padding:10px 12px}.play-top{font-size:13px}.caption{font-size:12px}.explain{padding:13px 15px}}
@media(max-width:620px){table,thead,tbody{display:block;width:100%}table{min-width:0;font-size:12px}thead tr,tbody tr{display:grid;grid-template-columns:1.3fr 1fr .9fr .9fr}th,td{padding:9px 5px}thead th:nth-child(2){display:none}tbody th{grid-column:1;grid-row:1;white-space:normal}tbody td:nth-child(2){grid-column:1/-1;grid-row:2;background:#f8fafb;color:var(--muted)}tbody td:nth-child(3){grid-column:2;grid-row:1}tbody td:nth-child(4){grid-column:3;grid-row:1}tbody td:nth-child(5){grid-column:4;grid-row:1}td small{font-size:11px}}
</style></head><body><main><header><p class="eyebrow">CV PROJECT · 已有 DEVELOPMENT 数据 · 真实 RGB</p>
<h1>线索被看见了，为什么预测仍会错？</h1><p class="intro">下面把三个既有案例的真实画面与冻结预测器的数值放在一起。图像全部来自缓存，没有生成或解码“预测图片”；query 后的画面是<strong>真实未来，仅用于评估</strong>。</p>
<div class="stats"><div class="stat"><strong>27 / 72</strong><span>可比较父集合中，保留 cue 更差</span></div><div class="stat"><strong>21 / 27</strong><span>这些失败中，正确 cue 比匹配错误 cue 更差</span></div><div class="stat"><strong>48 / 72</strong><span>全量对照中，正确 cue 比错误 cue 更好</span></div></div>
<p class="caption">72 个案例来自 192 个已观察 episode；其余 120 个父集合含两份等价 cue，一次删除无法完全去掉线索。这三例按预先固定的数值规则挑选，不表示普遍表现，也不替代全量分布或独立新场景验证。</p>
<nav><a href="#content_harm">内容失败</a><a href="#age_contribution">年龄参与</a><a href="#correct_cue_success">成功对照</a></nav></header>
<aside class="explain"><strong>先看什么</strong><p>早期顶部的斜线方向编码线索。后来 cue 消失，两条配对轨迹在 recent window 内完全相同；query 之后才分到上/下轨。轨道由 cue 与 query 第一动作共同决定，<strong>斜线不是固定指向某条轨道的箭头</strong>。</p><p><strong>α 是特征方向，不是位置。</strong>令两个真实未来的中点为 m、半差为 d：<code>α=(ŷ−m)·d / ‖d‖²</code>。+1 是该案例真实未来沿此方向的位置，−1 是 sibling 未来，0 是中点；α&gt;0 更接近正确分叉的特征目标。其余正交误差仍计入 MSE，不能把 α 当成实际坐标或完整准确率。</p></aside>
''' + ''.join(sections) + '''<footer>本页自包含，可离线打开。所有图像来自已校验的 unique_rgb.npz / records.json；所有数值来自已完成的 12 组合实验，无新增模型调用。播放器中的时序是真实轨迹，表中的 OOD 混合输入只用于诊断。完整源文件、逐帧像素哈希和 provenance 见同目录 JSON 清单。</footer></main>
<script id="case-data" type="application/json">'''+data_json+'''</script><script>
const data=JSON.parse(document.getElementById('case-data').textContent);
for(const c of data.cases){const p=document.querySelector(`[data-case="${c.case}"]`),s=p.querySelector('.slider'),b=p.querySelector('.play');let timer=null;
const stop=()=>{if(timer){clearInterval(timer);timer=null}b.textContent='播放'};
const show=()=>{const t=Number(s.value);p.querySelector('.actual').src=data.frames[c.actual_rgb_indices[t]];p.querySelector('.sibling').src=data.frames[c.sibling_rgb_indices[t]];p.querySelector('.clock').textContent=`t=${t} / ${c.frames-1}`;p.querySelector('.stage').textContent=t<c.query?'已观察历史':t===c.query?'query · 当前输入':'真实未来 · 仅评估';p.dataset.time=String(t)};
s.addEventListener('input',()=>{stop();show()});p.querySelectorAll('[data-time]').forEach(j=>j.addEventListener('click',()=>{stop();s.value=j.dataset.time;show()}));
b.addEventListener('click',()=>{if(timer){stop();return}b.textContent='暂停';timer=setInterval(()=>{s.value=(Number(s.value)+1)%c.frames;show()},180)});show();}
</script></body></html>'''


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source_hash = sha(__file__)
    data = args.data.resolve(); manifest = read(data/'manifest.json')
    for name in ('records.json', 'unique_rgb.npz', 'unique_rgb_sha256.json'):
        verified(data, name, manifest)
    records = {r['id']: r for r in read(data/'records.json')}
    hashes = read(data/'unique_rgb_sha256.json')
    factor_manifest = read(args.factors/'manifest.json')
    if sha(args.factors/'case_cards.json') != factor_manifest['files']['case_cards.json']['sha256']:
        raise ValueError('Fixed factorial case cards changed')
    original_cards = read(args.factors/'case_cards.json')
    if set(original_cards) != set(CASES): raise ValueError('Require the three fixed factorial representatives')
    with np.load(data/'unique_rgb.npz', allow_pickle=False) as a: rgb = a['rgb'].copy()
    frames_dir, figures_dir = output/'frames', output/'figures'
    frames_dir.mkdir(exist_ok=True); figures_dir.mkdir(exist_ok=True)
    cards = {}; used_frames = {}; gif_proofs = {}
    for kind in CASES:
        case = original_cards[kind]; rec = records[case['episode_id']]; sibling = records[case['sibling_id']]
        q, cue = case['query'], case['cue_time']
        for t in range(q-3, q+1):
            if not np.array_equal(rgb[rec['rgb_indices'][t]], rgb[sibling['rgb_indices'][t]]):
                raise ValueError('Recent paired RGB mismatch')
        for t in case['target_times']:
            if np.array_equal(rgb[rec['rgb_indices'][t]], rgb[sibling['rgb_indices'][t]]):
                raise ValueError('Paired true future RGB does not differ')
        difference = np.any(rgb[rec['rgb_indices'][cue]] != rgb[sibling['rgb_indices'][cue]], axis=-1)
        yy, xx = np.nonzero(difference)
        if not len(yy) or yy.max() > 22: raise ValueError('Cue difference is not confined to the top historical diagram')
        with np.load(verified(data, rec['feature_path'], manifest), allow_pickle=False) as a: action = a['actions'][q].copy()
        with np.load(verified(data, rec['eval_path'], manifest), allow_pickle=False) as a:
            label = int(a['label']); future_classes = a['classes'][case['target_times']].copy()
        if not np.all(future_classes == (label ^ int(action[0] > 0))):
            raise ValueError('Cue/action branch description differs from actual evaluation metadata')
        for r in (rec, sibling):
            for t, index in enumerate(r['rgb_indices']):
                if index in used_frames: continue
                pixel_hash = hashlib.sha256(rgb[index].tobytes()).hexdigest()
                if pixel_hash != hashes[index]: raise ValueError('Raw RGB frame pixel hash mismatch')
                relative = f'frames/rgb_{index:04d}.png'; path = output/relative
                Image.fromarray(rgb[index]).save(path)
                if not np.array_equal(np.asarray(Image.open(path).convert('RGB')), rgb[index]):
                    raise ValueError('Native PNG changed cached pixels')
                used_frames[index] = {'png': relative, 'rgb_sha256': pixel_hash, 'shape': list(rgb[index].shape)}
        selected = select_rows(case)
        figure = f'figures/{kind}.png'; animation = f'figures/{kind}.gif'
        make_figure(output/figure, kind, case, rec, sibling, rgb, selected)
        gif_proofs[kind] = make_gif(output/animation, case, rec, sibling, rgb)
        cards[kind] = {k: v for k, v in case.items() if k != 'arms'} | {
            'case': kind, 'frames': rec['frames'], 'display_rows': selected,
            'actual_rgb_indices': rec['rgb_indices'], 'sibling_rgb_indices': sibling['rgb_indices'],
            'cue_difference_bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
            'first_query_action': float(action[0]), 'cue_label': label,
            'true_lane': int(future_classes[0]), 'figure_png': figure, 'animation_gif': animation}
    save(output/'cases.json', cards); save(output/'rgb_frames.json', {str(k): v for k, v in used_frames.items()})
    frame_uri = {str(index): uri(output/info['png'], 'image/png') for index, info in used_frames.items()}
    (output/'index.html').write_text(make_html(output, cards, frame_uri))
    audit = {'status': 'completed_real_rgb_casebook', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'new_model_calls': 0, 'new_encoder_calls': 0, 'simulator_calls': 0, 'synthetic_prediction_images': 0,
             'case_count': len(cards), 'native_png_frames': len(used_frames), 'native_png_pixels_exact': True,
             'gif_proofs': gif_proofs, 'source_sha256': source_hash,
             'fixed_factor_case_cards_sha256': sha(args.factors/'case_cards.json'),
             'raw_bank_sha256': sha(data/'unique_rgb.npz'), 'records_sha256': sha(data/'records.json'),
             'selection': 'three previously fixed factorial representatives; not typicality estimates',
             'population': {'fresh_episodes': 192, 'comparable_parents': 72, 'original_cue_harm': 27,
                            'matched_correct_cue_worse_inside_original_harm': 21, 'matched_correct_cue_better_all': 48}}
    if sha(__file__) != source_hash: raise AssertionError('Exporter source changed during run')
    save(output/'audit.json', audit)
    (output/'source').mkdir(exist_ok=True); shutil.copy2(__file__, output/'source'/Path(__file__).name)
    output_manifest = {'status': 'complete', 'files': {str(p.relative_to(output)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                      for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    save(output/'manifest.json', output_manifest)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'build/literature-baselines-20261022/fresh-data')
    p.add_argument('--factors', type=Path, default=ROOT/'results/literature_diagnosis/cue-factors-20261022')
    p.add_argument('--output', type=Path, default=ROOT/'results/literature_diagnosis/visual-casebook-20261022')
    run(p.parse_args())
