"""Break a production physics term in an isolated copy; the same gate must fail."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path.cwd()
with tempfile.TemporaryDirectory(prefix="mutation.", dir=root / "build/robot_arm") as temp:
    tree = Path(temp)
    shutil.copytree(root / "ocean/robot_arm", tree / "ocean/robot_arm")
    header = tree / "ocean/robot_arm/robot_arm_cuda.cuh"
    source = header.read_text()
    before = "rhs[joint] = arm_motor - state->qd[joint] + gravity[joint];"
    assert source.count(before) == 1
    header.write_text(source.replace(before,
        "rhs[joint] = arm_motor - state->qd[joint]; // mutation: omit gravity"))
    env = dict(os.environ, ROBOT_ARM_SOURCE_ROOT=str(tree), ROBOT_ARM_TRAJECTORY_ONLY="1")
    run = subprocess.run(["bash", "tests/metal/run_test_robot_arm.sh"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = root / "build/robot_arm/mutation.log"
    log.write_text(run.stdout)
    assert run.returncode != 0 and "gate" in run.stdout and "AssertionError" in run.stdout, run.stdout
    assert "observation gate" in run.stdout or "physics gate" in run.stdout or "reward gate" in run.stdout, run.stdout
    print("PASS: removed joint gravity term; trajectory gate rejected mutant", log)
