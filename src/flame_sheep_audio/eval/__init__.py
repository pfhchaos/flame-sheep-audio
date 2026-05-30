"""Daemon-internal behavior evaluations for flame-sheep-audio.

Distinct from `flame_sheep/eval/`, which compares multiple beat detector
implementations against osu ground truth. This package evaluates the
flame-sheep-audio daemon's own behavior on synthetic stimuli — band
routing, tempo accuracy, density tracking, stability — by running the
full AudioProcessor pipeline and comparing measured metrics against a
checked-in baseline. Drift signals algorithm change.

Structure:
  common.py        — baseline load/save, drift comparison
  <category>.py    — one module per behavioral axis, each exposes a
                      run() function returning a metrics dict
  baselines/       — checked-in baseline JSONs, one per category

Refresh via tools/refresh_eval_baselines.py after intentional algorithm
changes. The regression test in tests/eval/test_audio_regressions.py
loads baselines, runs each eval, and asserts drift within per-metric
tolerance.
"""
