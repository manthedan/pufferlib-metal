"""Require fresh checkpoints and finite train metrics from the real trainer."""
import configparser
from pathlib import Path
import re
import sys

import numpy as np

root, precision = Path(sys.argv[1]), sys.argv[2]
logs = list((root / precision / "logs").rglob("*.ini"))
checkpoints = list((root / precision / "checkpoints").rglob("*.bin"))
assert len(logs) == len(checkpoints) == 1, (logs, checkpoints)
weights = np.fromfile(checkpoints[0], dtype="<f4")
assert weights.size > 400000 and np.isfinite(weights).all(), "invalid checkpoint"
ini = configparser.ConfigParser(strict=False)
ini.read(logs[0])
metrics = ini["metrics"]
for key in ("env/score", "env/episode_return", "env/episode_length", "importance", "agent_steps"):
    values = np.array([float(v) for v in metrics[key].split(",")])
    assert np.isfinite(values).all(), (key, values)
assert float(metrics["agent_steps"].split(",")[-1]) == 262144
summary = re.search(r"^metal_summary .*$", (root / f"train_{precision}.log").read_text(), re.M)
assert summary and float(re.search(r"\bsps=(\S+)", summary[0])[1]) > 0
print(f"robot_arm {precision} smoke PASS: finite checkpoint and completed episodes")
