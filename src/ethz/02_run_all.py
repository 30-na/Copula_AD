"""Step 2: run every ETHZ step in order, with the paper's settings (any OS).

Needs step 1 first (python src/ethz/01_download.py all, seis env).
Settings: IVR fitted on 0-30 s, 1% level (99th percentile), tested part 30 s -> S.

The steps use two conda environments (see environment.yml, environment_seis.yml):
  seis       models (EQTransformer, CRED, DPP; resamples to 100 Hz first), Baer-Kradolfer
  copula-ad  baselines (STA/LTA, sliding std, MP), IVR, evaluate
Each step saves its results in parts of 500 time series and skips finished parts, so a
stopped run continues where it stopped. Metrics are updated after each step; logs and
results go to results/ethz/. Runs at low priority with few threads, to keep the computer usable.

Run from the project folder (any environment):
  python src/ethz/02_run_all.py
Then, for the 5% level from the saved results (no refit):
  EVAL_Q=95 IVR_FA=0 IVR_FB=3000 IVR_Q=99 TEST_FROM=3000 TEST_TO=S python src/ethz/ethz_run.py evaluate all
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "src" / "ethz" / "ethz_run.py"
LOG = ROOT / "results" / "ethz"
SETTINGS = {"IVR_FA": "0", "IVR_FB": "3000", "IVR_Q": "99", "TEST_FROM": "3000", "TEST_TO": "S",
            "PYTHONPATH": "src", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "TORCH_THREADS": "2",
            "WORKERS": "3"}
STEPS = [("seis", "models"), ("copula-ad", "evaluate"),
         ("copula-ad", "baselines"), ("seis", "bk"), ("copula-ad", "evaluate"),
         ("copula-ad", "ivr"), ("copula-ad", "evaluate")]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M} {msg}"
    print(line, flush=True)
    with open(LOG / "log_chain.txt", "a") as f:
        f.write(line + "\n")


def python_of(env: str) -> list[str]:
    """The environment's own python if it can be found (conda's envs folder), else `conda run`."""
    import shutil
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    bases = [Path(conda).resolve().parents[1]] if conda else []
    bases += [Path.home() / d for d in ("miniconda3", "anaconda3", "AppData/Local/miniconda3",
                                         "AppData/Local/anaconda3")]
    for b in bases:
        for p in (b / "envs" / env / "python.exe", b / "envs" / env / "bin" / "python"):
            if p.exists():
                return [str(p)]
    return ["conda", "run", "-n", env, "--no-capture-output", "python"]


def run(env: str, step: str) -> None:
    cmd = python_of(env) + ["-u", str(RUN), step, "all"]
    low = {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name == "nt" else \
          {"preexec_fn": lambda: os.nice(10)}
    with open(LOG / f"log_{step}.txt", "a") as out:
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **SETTINGS}, stdout=out,
                           stderr=subprocess.STDOUT, shell=(os.name == "nt"), **low)
    if r.returncode:
        log(f"FAILED {step} (see results/ethz/log_{step}.txt)")
        sys.exit(1)
    log(f"{step} done")


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    log("START")
    for env, step in STEPS:
        run(env, step)
    log("ALL DONE")


if __name__ == "__main__":
    main()
