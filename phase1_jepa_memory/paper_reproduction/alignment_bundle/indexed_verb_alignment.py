"""Official indexed-TSM label axis; does not map any re-extracted raw RGB."""
from collections import Counter, defaultdict
import numpy as np


def selected_segments(raw_rows, selected_videos, catalog):
    """Filter recording/camera names before interpreting any numeric fields."""
    for row in raw_rows:
        name=row.get('video')
        if name not in selected_videos:
            continue
        start,end=int(row['start_frame']),int(row['end_frame'])
        if start<0 or end<start:
            raise ValueError('invalid selected inclusive annotation interval')
        numeric=tuple(int(row[k]) for k in ('action_id','verb_id','noun_id'))
        if numeric[0] not in catalog or catalog[numeric[0]]!=numeric[1:]:
            raise ValueError('selected numeric annotation unavailable or invalid')
        yield name,start,end,*numeric


def indexed_multihot(frame_indices, intervals):
    """gen_fine_labels.py maps inclusive CSV [s,e] to JPEG [s+1,e+1]."""
    frames=np.asarray(frame_indices,dtype=np.int64)
    if frames.ndim!=1 or (frames<1).any() or (np.diff(frames)<=0).any():
        raise ValueError('strictly increasing one-based indexed TSM keys required')
    events=defaultdict(Counter)
    for start,end,verb in intervals:
        if start<0 or end<start or not 0<=verb<24:raise ValueError('invalid selected verb interval')
        events[start][verb]+=1;events[end+1][verb]-=1
    ordered=sorted(events);position=0;counts=Counter();labels=np.zeros((len(frames),24),np.uint8)
    for j,frame in enumerate(frames):
        annotation_index=int(frame)-1
        while position<len(ordered) and ordered[position]<=annotation_index:
            counts.update(events[ordered[position]])
            if any(v<0 for v in counts.values()):raise ValueError('annotation sweep underflow')
            position+=1
        for verb,count in counts.items():
            if count>0:labels[j,verb]=1
    return labels
