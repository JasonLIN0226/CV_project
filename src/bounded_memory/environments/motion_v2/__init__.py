"""Motion v2: environment, generator, feature store, reader qualification metrics, and the
student stage (labels, students, online evaluation, metrics).

The modules keep their original file names because they locate each other by file name and the
tracked student checkpoints record the SHA256 of step34_students.py. All are byte-identical to
experiments/motion_v2/ at commit fe69074 except reader_v2.py, where one constant now points at the
predictor source vendored in this package. pipeline/ holds the original run scripts and their
frozen configs, also byte-identical; they do not run from here yet (pipeline/README.md). The
submit, export and transfer scripts of the original deployment were not kept.

    envs_v2.py, generate_v2.py   the environment and the dataset generator
    audit_v2.py                  dataset audit
    reader_v2.py                 feature store and predictor inputs for the four memory conditions
    step2_metrics.py             reader qualification metrics
    step34_data.py               training states and the label definitions (local / use-time, feature / position, shuffled)
    step34_students.py           set-conditioned and independent students
    step34_core.py               frozen assets, online replay, baselines
    step34_metrics.py            online validation metrics and paired comparisons
    evidence.py                  recomputes the published tables from the tracked loss arrays

Added after the refactor (new code, not extracted):

    frozen.py                    checks the frozen study assets against their recorded SHA256 and replays the
                                 online validation with the unchanged step-3/4 code
    episodes.py                  Motion episodes for the literature baselines; scoring kept frames on the readers
    fresh.py                     fresh scenes from the registered generator under a split name of its own, for
                                 training our scorer on episodes the frozen readers never saw
"""
