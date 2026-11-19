# IVR: anomaly detection with the innovation variance ratio

An ARMA model is fitted on the normal (anomaly-free) start of a time series. Each window of
the rest gets the ratio R_w = σ²_w / σ²_0 of its own innovation variance to the one of the
normal part; a window is flagged when R_w is above a threshold taken from a bootstrap of
windows of the normal part. The method is in [`src/ivr.py`](src/ivr.py) (each function
has a short description there).

Two benchmarks:

- **UCR anomaly archive** (250 time series, KDD21 in TSB-UAD): IVR against the 12 methods whose
  per-series results TSB-UAD published, with TSB-UAD's metric code.
- **ETHZ earthquakes** (9,670 time series, SeisBench): IVR against EQTransformer, CRED and DPP
  (ETHZ weights), STA/LTA, sliding std, matrix profile and the Baer-Kradolfer picker.

## Folders

```
src/
  ivr.py          the method (used by all scripts)
  ucr/            01_prepare  02_run_ivr  03_run_rolling_std  04_compare
  ethz/           01_download  02_run_all  (ethz_run.py: the steps that 02 runs)
  simulation/     01_generate ... 08_ucr_null_distributions
data/
  raw/            ucr/, ethz/, tsb_uad/   (as downloaded)
  processed/      ucr/, ethz/             (prepared / resampled to 100 Hz)
results/          ucr/, ethz/ (figures/), simulation/
baselines/        other people's code (see Setup)
documents/        cited papers
archive/          old scripts, kept for reference
```

## Setup

```bash
conda env create -f environment.yml        # "copula-ad": everything except the next line
conda env create -f environment_seis.yml   # "seis": ETHZ download, deep models, Baer-Kradolfer

# TSB-UAD: published per-series results, metric code, window-length rule (version used: 313f0fd)
git clone https://github.com/TheDatumOrg/TSB-UAD.git baselines/TSB-UAD
git -C baselines/TSB-UAD checkout 313f0fd
```

Run every command from the project folder.

## UCR (copula-ad environment)

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python src/ucr/01_prepare.py` | Downloads the UCR archive and the anomaly types (Rewicki et al.) to `data/raw/ucr/`, writes `data/processed/ucr/` | minutes |
| 2 | `python src/ucr/02_run_ivr.py` | IVR on all 250 time series → `results/ucr/ivr_scores/` (skips finished ones) | hours |
| 3 | `python src/ucr/03_run_rolling_std.py` | Rolling std baseline → `results/ucr/rolling_std_scores/` | minutes |
| 4 | `python src/ucr/04_compare.py` | Metrics (TSB-UAD code) + table with the published results → `results/ucr/summary.csv` | minutes |

Step 4 with TSB-UAD's own labels (`UCR_LABELS=tsb`, outputs `*_tsblabels.csv`) needs TSB-UAD's copy
of the data: download <https://www.thedatum.org/datasets/TSB-UAD-Public.zip> to
`data/raw/tsb_uad/` and unzip its `KDD21` folder there.

## ETHZ

| Step | Command | Environment | What it does |
|---|---|---|---|
| 1 | `python src/ethz/01_download.py all` | seis | ETHZ test time series with a P pick (9,670) → `data/raw/ethz/` |
| 2 | `python src/ethz/02_run_all.py` | any | All steps in order (models, baselines, Baer-Kradolfer, IVR, metrics) → `results/ethz/` |

Paper settings: IVR fitted on 0–30 s, 1% level, tested part 30 s → S arrival, window 50 points
(0.5 s), score on the window's end point. Step 2 took about 1.5 days on a laptop; it saves its
results in parts of 500 time series and continues where it stopped. The 5% level from the saved
results (no refit):

```bash
EVAL_Q=95 IVR_FA=0 IVR_FB=3000 IVR_Q=99 TEST_FROM=3000 TEST_TO=S python src/ethz/ethz_run.py evaluate all
```

## Simulation (copula-ad environment)

`python src/simulation/<NN>_<name>.py --help` for the options of each script; outputs go to
`results/simulation/`.
