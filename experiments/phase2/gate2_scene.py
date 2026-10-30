"""Gate2: two independent observable bits and a terminal three-cue parent.

Only NumPy and Pillow are required. This module generates raw scenes; it does
not encode images, fit a reader, run retention policies, or select scenes by
model outcomes. `distant_style_duplicate` describes a fixed rendering change,
not a claim that any learned embedding actually places the copy farther away.

`episode` separates `rgb`/`actions` from evaluator-only states, bits and evidence.
The runner must feed retention only observations through `retention_stop=query`
and just-executed actions. Future frames are targets, never later admissions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from itertools import permutations,product
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

VERSION="gate2-terminal-v1"
IMAGE_SIZE=96
RECENT_LENGTH=4
MEMORY_CAPACITY=2
HORIZONS=(3,6)
AXES=("a","b")  # a is horizontal x; b is vertical screen y.
BIT_COMBINATIONS=tuple(product((0,1),repeat=2))
CONDITIONS=("benign_duplicate","distant_style_duplicate")
SPLIT_SEEDS={"train":20261028,"validation":20261029,"geometry":20261030,"development":20261031}
FAMILIES_PER_SPLIT={"train":120,"validation":24,"geometry":12,"development":24}
ROLE_PERMUTATIONS=tuple(permutations(("A","Aprime","B")))
PUBLIC_OBSERVATION_KEYS=("rgb","actions")
EVALUATOR_ONLY_KEYS=("states","bits","evidence","semantic_axes","roles","target_times")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def scene(seed,index,split,condition="benign_duplicate"):
    """Return JSON-serializable public scene parameters, independent of bits.

    A/Aprime are two views of the duplicated axis; B supplies the other axis.
    For every block of 12 family indices, both duplicated axes and all six role
    orders appear once. The two styles at one family index share every public
    factor; their sole visual difference is Aprime's presentation.
    """
    if split not in SPLIT_SEEDS or int(seed)!=SPLIT_SEEDS[split]:
        raise ValueError("Use the fixed seed for the named split")
    if not isinstance(index,(int,np.integer)) or isinstance(index,(bool,np.bool_)) or index<0:
        raise ValueError("index must be a nonnegative integer")
    if condition not in CONDITIONS:
        raise ValueError("Unknown fixed style condition")
    seed,index=int(seed),int(index)
    family_id=digest([VERSION,seed,index,split])[:24]
    scene_id=digest([family_id,condition])[:24]
    rng=np.random.default_rng(int(digest([VERSION,"public",seed,index,split])[:16],16))
    query=int(rng.integers(12,19))
    length=query+max(HORIZONS)+1
    actions=np.zeros((length-1,2),np.float32)
    actions[query:]=rng.uniform(.75,1.05,(max(HORIZONS),2)).astype(np.float32)
    duplicate_axis=AXES[index%2]
    order=ROLE_PERMUTATIONS[(index//2)%6]
    other_axis=AXES[1-AXES.index(duplicate_axis)]
    semantic_axes=[other_axis if role=="B" else duplicate_axis for role in order]
    return {"version":VERSION,"seed":seed,"index":index,"split":split,"condition":condition,
        "family_id":family_id,"scene_id":scene_id,"image_size":IMAGE_SIZE,
        "query":query,"retention_stop":query,"length":length,"horizons":list(HORIZONS),
        "recent_length":RECENT_LENGTH,"memory_capacity":MEMORY_CAPACITY,
        "cue_times":[query-6,query-5,query-4],"recent_times":list(range(query-3,query+1)),
        "target_times":[query+h for h in HORIZONS],"roles":list(order),"semantic_axes":semantic_axes,
        "duplicate_axis":duplicate_axis,"role_permutation_index":(index//2)%6,
        "origin":[float(rng.uniform(.47,.53)),float(rng.uniform(.61,.65))],
        "motion_scale":.03,"actions":actions.tolist(),
        "decor":[[int(v) for v in row] for row in rng.integers(75,170,size=(3,3))],
        "style_claim":"Rendering condition only; embedding geometry must be measured on the geometry split."}


def _arrow(draw,axis,bit,*,center,horizontal_length,vertical_length,color,width):
    cx,cy=center
    dx,dy=((horizontal_length,0) if axis=="a" else (0,vertical_length))
    sign=1 if bit else -1
    start=(cx-sign*dx/2,cy-sign*dy/2)
    end=(cx+sign*dx/2,cy+sign*dy/2)
    draw.line((start,end),fill=color,width=width)
    if axis=="a":
        head=[end,(end[0]-sign*7,end[1]-6),(end[0]-sign*7,end[1]+6)]
    else:
        head=[end,(end[0]-6,end[1]-sign*7),(end[0]+6,end[1]-sign*7)]
    draw.polygon(head,fill=color)


def _render(s,t,bits,state):
    image=Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(24,29,38))
    draw=ImageDraw.Draw(image)
    draw.rectangle((7,35,89,91),fill=(44,49,60),outline=(112,121,136),width=2)
    # Public, static nuisance colors are shared by every sibling and both styles.
    for j,color in enumerate(s["decor"]):
        draw.rectangle((13+28*j,92,23+28*j,95),fill=tuple(color))
    x,y=(float(state[0])*IMAGE_SIZE,float(state[1])*IMAGE_SIZE)
    draw.ellipse((x-4,y-4,x+4,y+4),fill=(237,175,70),outline=(255,235,174),width=1)
    if t in s["cue_times"]:
        j=s["cue_times"].index(t)
        role,axis=s["roles"][j],s["semantic_axes"][j]
        bit=bits[AXES.index(axis)]
        distant=s["condition"]=="distant_style_duplicate" and role=="Aprime"
        if distant:
            # Fixed large palette/layout change. Arrow orientation and axis are
            # preserved: this is another observable view of the SAME bit.
            draw.rectangle((1,1,94,33),fill=(213,179,70),outline=(230,97,181),width=3)
            for yy in range(5,30,8):
                draw.rectangle((5,yy,23,yy+3),fill=(111,52,150))
            _arrow(draw,axis,bit,center=(61,17),horizontal_length=38,vertical_length=23,
                   color=(31,39,57),width=5)
        else:
            draw.rectangle((15,3,81,31),fill=(47,67,83),outline=(144,164,181),width=2)
            _arrow(draw,axis,bit,center=(48,17),horizontal_length=29,vertical_length=19,
                   color=(240,234,202),width=4)
    return np.asarray(image,dtype=np.uint8).copy()


def episode(s,bits):
    """Generate one sibling. Only rgb/actions are observation/control channels."""
    if s.get("version")!=VERSION or s["condition"] not in CONDITIONS:
        raise ValueError("Require an unchanged Gate2 scene")
    bits=tuple(bits)
    if len(bits)!=2 or any(not isinstance(v,(int,np.integer,bool,np.bool_)) or v not in (0,1) for v in bits):
        raise ValueError("Require exactly two binary bits")
    bits=tuple(int(v) for v in bits)
    actions=np.asarray(s["actions"],np.float32).copy()
    if actions.shape!=(s["length"]-1,2) or not np.isfinite(actions).all():
        raise ValueError("Invalid public control sequence")
    query=s["query"]
    if np.any(actions[:query]!=0) or np.any(actions[query:]<=0):
        raise ValueError("Controls must activate motion at query, never earlier")
    states=np.repeat(np.asarray(s["origin"],np.float32)[None],s["length"],axis=0)
    signs=np.asarray([2*v-1 for v in bits],np.float32)
    for t in range(query+1,s["length"]):
        states[t]=states[t-1]+signs*np.float32(s["motion_scale"])*actions[t-1]
    if not np.all((states>.08)&(states<.90)):
        raise ValueError("Fixed motion left its declared visible arena")
    rgb=np.stack([_render(s,t,bits,states[t]) for t in range(s["length"])])
    episode_id=digest([s["scene_id"],list(bits)])[:24]
    return {"id":episode_id,"episode_id":episode_id,"scene_id":s["scene_id"],"family_id":s["family_id"],
        "query":query,"retention_stop":query,"rgb":rgb,"actions":actions,"states":states,
        "evidence":np.asarray(s["cue_times"],np.int64),"semantic_axes":tuple(s["semantic_axes"]),
        "roles":tuple(s["roles"]),"bits":np.asarray(bits,np.int64),"target_times":np.asarray(s["target_times"],np.int64),
        "observation_keys":PUBLIC_OBSERVATION_KEYS,"evaluator_only_keys":EVALUATOR_ONLY_KEYS}


def validate_quartet(s):
    """Raw structural checks only; never selects based on encoder/model loss."""
    episodes=[episode(s,bits) for bits in BIT_COMBINATIONS]
    first=episodes[0];q=s["query"];cue=set(s["cue_times"])
    for ep in episodes[1:]:
        if not np.array_equal(ep["actions"],first["actions"]) or not np.array_equal(ep["rgb"][q-3:q+1],first["rgb"][q-3:q+1]):
            raise AssertionError("Quartet actions or R4 recent frames differ")
        for t in range(q+1):
            if t not in cue and not np.array_equal(ep["rgb"][t],first["rgb"][t]):
                raise AssertionError("A hidden bit leaked outside its cue before query")
    for h in HORIZONS:
        if len({ep["rgb"][q+h].tobytes() for ep in episodes})!=4:
            raise AssertionError("Four distinct future RGB outcomes are required")
    by_bits={tuple(ep["bits"]):ep for ep in episodes}
    for bits in BIT_COMBINATIONS:
        for axis_i,axis in enumerate(AXES):
            flipped=list(bits);flipped[axis_i]=1-flipped[axis_i]
            a,b=by_bits[bits],by_bits[tuple(flipped)]
            for t,semantic_axis in zip(s["cue_times"],s["semantic_axes"]):
                equal=np.array_equal(a["rgb"][t],b["rgb"][t])
                if equal!=(semantic_axis!=axis):
                    raise AssertionError("Cue changed for the wrong independent bit")
            if not np.array_equal(a["states"][:,1-axis_i],b["states"][:,1-axis_i]):
                raise AssertionError("Bit affected the other motion axis")
    return episodes


def manifest(split,families=None):
    families=FAMILIES_PER_SPLIT[split] if families is None else int(families)
    if families<12 or families%12:
        raise ValueError("Use complete blocks of 12 axis/order cells")
    records=[]
    for index in range(families):
        for condition in CONDITIONS:
            s=scene(SPLIT_SEEDS[split],index,split,condition)
            records.append(s)
    return {"version":VERSION,"split":split,"seed":SPLIT_SEEDS[split],"families":families,
        "quartets":2*families,"episodes":8*families,"conditions":list(CONDITIONS),
        "families_are_independent_unit":"Two styles and four bits share a base family; cluster comparisons by family.",
        "records":records,"observation_keys":list(PUBLIC_OBSERVATION_KEYS),"evaluator_only_keys":list(EVALUATOR_ONLY_KEYS),
        "no_outcome_based_filtering":True,"embedding_distance_verified":False}


def contact_sheet(scenes,path):
    """Scientific contact sheet from exact rendered simulator frames."""
    if len(scenes)!=2 or {s["condition"] for s in scenes}!=set(CONDITIONS):
        raise ValueError("Show both fixed styles of one family")
    if len({s["family_id"] for s in scenes})!=1:
        raise ValueError("Contact sheet must use a matched style family")
    cell=IMAGE_SIZE;left=190;top=64;row_height=cell+27
    image=Image.new("RGB",(left+6*(cell+7)+8,top+8*row_height+28),(247,248,250));draw=ImageDraw.Draw(image)
    draw.text((10,9),"Gate2 raw RGB: same family, two independent bits, matched styles",fill=(20,30,45))
    draw.text((10,28),"a = horizontal; b = vertical. Bit 0: left/up. Bit 1: right/down.",fill=(20,30,45))
    s0=scenes[0]
    headers=["cue 1: "+s0["roles"][0],"cue 2: "+s0["roles"][1],"cue 3: "+s0["roles"][2],"shared query","true q+3","true q+6"]
    for j,text in enumerate(headers):draw.text((left+j*(cell+7),47),text,fill=(20,30,45))
    row=0
    for s in scenes:
        for bits in BIT_COMBINATIONS:
            ep=episode(s,bits);y=top+row*row_height
            draw.text((8,y+13),s["condition"],fill=(20,30,45))
            draw.text((8,y+31),"a="+str(bits[0])+", b="+str(bits[1])+"; duplicate "+s["duplicate_axis"],fill=(20,30,45))
            for j,t in enumerate(s["cue_times"]+[s["query"]]+s["target_times"]):
                x=left+j*(cell+7);image.paste(Image.fromarray(ep["rgb"][t]),(x,y));draw.text((x,y+cell+3),"t="+str(t),fill=(20,30,45))
            row+=1
    image.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=tuple(SPLIT_SEEDS),default="geometry")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("Use a new empty preflight directory")
    out.mkdir(parents=True,exist_ok=True)
    m=manifest(args.split);counts=Counter();rgb_bytes=0
    for s in m["records"]:
        eps=validate_quartet(s);rgb_bytes+=sum(e["rgb"].nbytes for e in eps)
        counts[(s["condition"],s["duplicate_axis"],s["role_permutation_index"])]+=1
    contact_sheet(m["records"][:2],out/"contact_sheet.png")
    (out/"manifest.json").write_text(json.dumps(m,indent=2,allow_nan=False)+"\n")
    report={"status":"passed_raw_structural_preflight","created_utc":datetime.now(timezone.utc).isoformat(),
        "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest(),
        "split":args.split,"families":m["families"],"quartets":m["quartets"],"episodes":m["episodes"],
        "axis_order_cells":{str(k):v for k,v in counts.items()},"generated_rgb_bytes_not_saved":rgb_bytes,
        "recent_identical_within_quartet":True,"controls_identical_within_quartet":True,
        "independent_bit_visual_channels_checked":True,"four_future_outcomes_checked":True,
        "encoding_or_model_calls":0,"geometry_in_embedding_space_verified":False,
        "limits":["Raw structure only; no predictor readability or learned geometry claim",
                  "Final confirmation split is intentionally absent"]}
    (out/"preflight.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="axis_order_cells"}),flush=True)


if __name__=="__main__":main()
