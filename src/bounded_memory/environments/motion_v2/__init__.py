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
"""
