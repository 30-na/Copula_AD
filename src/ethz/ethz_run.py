"""ETHZ test time series (data/raw/ethz): IVR vs EQTransformer, CRED, DPP (ETHZ weights),
STA/LTA, sliding std, matrix profile and the Baer-Kradolfer picker.

Data: raw (not filtered), resampled to 100 Hz (scipy resample_poly), 3 channels ZNE.
P wave is at 60 s in these time series.
  IVR (Z channel): fitted on 10-40 s (scaled by its median / MAD), ARMA(p,q) p,q <= 3 by
      AIC, 50-point windows sliding by 1, bootstrap threshold (95th percentile of R_w of
      2000 random windows of the fitting part, one-sided), score on the window's END point.
  Models: SeisBench ETHZ weights (from_pretrained("ethz")), annotate() on the whole time series
      (EQTransformer blinding (0, 0)); DPP has no annotate(): the benchmark's own way, 5 s
      windows. 0/1 = score > the weights' own detection threshold (EQTransformer 0.0300,
      CRED 0.0614 = classify() defaults for these weights; DPP 0.5).
Tested part: 40 s -> P + 5 s. Truth: 0 before P, 1 from P to P + 5 s.

Settings of the paper run (env vars): IVR_FA=0 IVR_FB=3000 IVR_Q=99 TEST_FROM=3000 TEST_TO=S
(fit 0-30 s, 1% level, tested part 30 s -> S). The defaults below are older settings.
All steps, in order: python src/ethz/02_run_all.py. One step:
  python src/ethz/ethz_run.py models all     # seis env (resamples first if needed)
  python src/ethz/ethz_run.py baselines all  # copula-ad env
  python src/ethz/ethz_run.py bk all         # seis env
  python src/ethz/ethz_run.py ivr all        # copula-ad env
  python src/ethz/ethz_run.py evaluate all   # copula-ad env (metrics + 5 example figures)
  EVAL_Q=95 python src/ethz/ethz_run.py evaluate all   # same, 5% level, from saved results
Output: data/processed/ethz/waves100.npz, results/ethz/ (parts/, summary_*.csv, figures/)
"""
import os
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

MODE, N = sys.argv[1], sys.argv[2]  # N: "all", or the N of a quick test download
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ (ivr.py)
_T = "" if N == "all" else f"_test{N}"
RAW, PROC = Path(f"data/raw/ethz{_T}"), Path(f"data/processed/ethz{_T}")
OUT = Path("results/ethz") if N == "all" else Path(f"results/ethz/test{N}")
FIGS = Path("results/ethz/figures")
SR, W, AFTER = 100, 50, 500
EA = int(os.environ.get("TEST_FROM", 4000))  # start of the tested part in points (default 40 s)
# TEST_TO=S: earthquake part = P -> S arrival (end of the P wave; P + 5 s if there is no S pick);
# default: P -> P + 5 s
TO_S = os.environ.get("TEST_TO", "") == "S"
FB = int(os.environ.get("IVR_FB", 4000))  # end of the fitting part in points (default 40 s)
# IVR_FA: start of the fitting part in points (default 1000 = 10 s); IVR_Q: bootstrap percentile (default 95)
FA = int(os.environ.get("IVR_FA", 1000))
Q = float(os.environ.get("IVR_Q", 95))
TAG = ("" if (FA, FB, Q) == (1000, 4000, 95) else f"_fit{FA // SR}-{FB // SR}_q{Q:g}") +     ("" if EA == 4000 else f"_test{EA // SR}") + ("_toS" if TO_S else "")
SHIFT = (W - 1) // 2
N_BOOT = 2000
WORKERS = int(os.environ.get("WORKERS", 3))  # parallel processes for IVR / baselines (low = gentle on the laptop)
CHUNK = 500  # time series per part: every step works on (and saves) one part at a time to keep RAM low
PARTS = OUT / "parts"


def part_file(key: str, c: int) -> Path:
    return PARTS / f"{key}_part{c:03d}.npz"


def n_series() -> int:
    return len(pd.read_csv(RAW / "metadata.csv", usecols=[0]))


def n_parts() -> int:
    return -(-n_series() // CHUNK)


def load_part(key: str, c: int):
    """-> (row indices, list of per-point score arrays, thresholds per time series) or None."""
    f = part_file(key, c)
    if not f.exists():
        return None
    d = np.load(f)
    ks = [int(k) for k in d["keys"]]
    return ks, [d[str(k)] for k in ks], np.broadcast_to(d["threshold"], (len(ks),)).astype(float)


def resample() -> None:
    from scipy.signal import resample_poly
    meta = pd.read_csv(RAW / "metadata.csv", low_memory=False)
    if (RAW / "waves.npz").exists():
        raw = np.load(RAW / "waves.npz")
    else:  # downloaded in parts (01_download.py)
        raw = {}
        for f in sorted(RAW.glob("waves_part*.npz")):
            raw.update(dict(np.load(f)))
    out = {}
    for k, sr in enumerate(meta["trace_sampling_rate_hz"]):
        x = raw[str(k)].astype(float)
        fr = Fraction(SR, int(sr)).limit_denominator(1000)
        out[str(k)] = (resample_poly(x, fr.numerator, fr.denominator, axis=-1) if sr != SR else x).astype(np.float32)
    np.savez_compressed(PROC / "waves100.npz", **out)


def models() -> None:
    import obspy
    import seisbench.models as sbm
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))  # keep the laptop responsive

    if not (PROC / "waves100.npz").exists():
        resample()
    w = np.load(PROC / "waves100.npz")  # lazy: one time series is read only when it is used
    n = len(w.files)
    t0 = obspy.UTCDateTime(2000, 1, 1)
    PARTS.mkdir(parents=True, exist_ok=True)
    for name, cls in (("EQTransformer", sbm.EQTransformer), ("CRED", sbm.CRED), ("DPP", sbm.DPPDetector)):
        if all(part_file(name, c).exists() for c in range(n_parts())):
            continue
        try:
            model = cls.from_pretrained("ethz")
        except TypeError:
            import seisbench.models.base as smb  # noqa: F401
            model = load_pretrained_ethz(cls)
        model.eval()
        thr = getattr(model, "default_args", {}).get("detection_threshold", 0.5)
        for c in range(n_parts()):
            if part_file(name, c).exists():
                continue
            ks = range(c * CHUNK, min((c + 1) * CHUNK, n))
            scores = _model_scores(name, model, [(k, w[str(k)]) for k in ks], t0, torch, obspy)
            np.savez_compressed(part_file(name, c), keys=np.array([str(k) for k in ks]), threshold=thr, **scores)
            print(f"{name} {ks[-1] + 1}/{n}", flush=True)
        print(name, "done, threshold", thr, flush=True)


def _model_scores(name, model, items, t0, torch, obspy) -> dict:
    scores = {}
    if True:
        if name == "DPP":
            with torch.no_grad():
                for k, x in items:
                    n = x.shape[-1] // 3000 * 3000
                    s = np.zeros(x.shape[-1], np.float32)
                    for h in range(0, n, 3000):
                        v = x[:, h:h + 3000] - x[:, h:h + 3000].mean(axis=-1, keepdims=True)
                        v = v / (np.abs(v).max(axis=-1, keepdims=True) + 1e-10)
                        v = torch.as_tensor(v[None]).reshape(1, 3, 6, 500).permute(0, 2, 1, 3).reshape(-1, 3, 500)
                        s[h:h + 3000] = np.repeat(1 - model(v)[:, -1].numpy(), 500)
                    scores[str(k)] = s
        else:
            kw = {"blinding": (0, 0)} if name == "EQTransformer" else {}
            for k, x in items:
                st = obspy.Stream([obspy.Trace(x[c], header={"network": "XX", "station": "S", "channel": f"HH{ch}",
                                                             "sampling_rate": SR, "starttime": t0})
                                   for c, ch in enumerate("ZNE")])
                tr = model.annotate(st, **kw).select(channel="*Detection")[0]
                tt = (tr.stats.starttime - t0) + np.arange(tr.stats.npts) / tr.stats.sampling_rate
                grid = np.arange(x.shape[-1]) / SR
                s = np.zeros(x.shape[-1], np.float32)
                inside = (grid >= tt[0]) & (grid <= tt[-1])
                s[inside] = np.interp(grid[inside], tt, tr.data)
                scores[str(k)] = s
    return scores


def load_pretrained_ethz(cls):
    import inspect
    import json
    import torch
    cache = next(Path.home().joinpath(".seisbench", "models").rglob(f"{cls.__name__.lower()}/ethz.json*"))
    cfg = json.loads(cache.read_text())
    ok = inspect.signature(cls.__init__).parameters
    model = cls(**{k: v for k, v in cfg.get("model_args", {}).items() if k in ok})
    model.load_state_dict(torch.load(cache.with_name(cache.name.replace(".json", ".pt")), map_location="cpu"))
    return model


def _ivr_one(z):
    from ivr import fit_reference_model, to_points, window_ratios
    med = np.median(z[FA:FB])
    mad = 1.4826 * np.median(np.abs(z[FA:FB] - med)) or 1.0
    s = (z - med) / mad
    for order in (3, 2, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit_reference_model(s[FA:FB], order, order)
        try:
            r = window_ratios(s, m, W)
            break
        except np.linalg.LinAlgError:
            continue
    starts = np.random.default_rng(0).integers(FA, FB - W + 1, N_BOOT)
    boot = r[starts].astype(np.float32)  # kept so any other threshold can be computed later
    thr = float(np.percentile(boot, Q))
    mid = to_points(r, W).astype(np.float32)
    end = np.empty_like(mid)
    end[SHIFT:], end[:SHIFT] = mid[:-SHIFT], mid[0]
    fit = {"order": list(m["order"]), "param_names": m["param_names"],
           "fixed_params": np.asarray(m["fixed_params"]).tolist(), "sigma2": m["reference_sigma2"],
           "median": float(med), "mad": float(mad)}
    return end, thr, str(m["order"]), boot, fit


def ivr() -> None:
    """Fits CHUNK time series at a time; each part is saved as soon as it is done
    (parts/IVR{TAG}_part<c>.npz) and finished parts are skipped on a restart. Saved per time
    series: R_w on every point (window end point), the threshold, the 2000 bootstrap R_w
    values, and the fitted ARMA."""
    import json
    from multiprocessing import Pool
    w = np.load(PROC / "waves100.npz")
    n = len(w.files)
    PARTS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(WORKERS) as pool:
        for c, a in enumerate(range(0, n, CHUNK)):
            part = part_file(f"IVR{TAG}", c)
            if part.exists():
                continue
            ks = range(a, min(a + CHUNK, n))
            out = pool.map(_ivr_one, [w[str(k)][0].astype(float) for k in ks])
            np.savez_compressed(part, keys=np.array([str(k) for k in ks]),
                                threshold=np.array([o[1] for o in out]), order=np.array([o[2] for o in out]),
                                boot=np.stack([o[3] for o in out]), fit=json.dumps([o[4] for o in out]),
                                **{str(k): o[0] for k, o in zip(ks, out)})
            print(f"IVR {min(a + CHUNK, n)}/{n}", flush=True)
    print("IVR done", flush=True)


STA, LTA = 50, 1000  # STA/LTA windows: 0.5 s and 10 s (both end at the current point)
BASELINES = ["STA/LTA", "sliding std", "MP"]


def _baselines_one(z):
    """Causal scores on the Z channel (each window ends at the scored point) and a threshold =
    the Q-th percentile of the method's own score inside the fitting part [FA, FB)."""
    import stumpy
    med = np.median(z[FA:FB])
    mad = 1.4826 * np.median(np.abs(z[FA:FB] - med)) or 1.0
    s = (z - med) / mad
    n = len(s)
    c2 = np.r_[0.0, np.cumsum(s ** 2)]
    out = {}
    # STA/LTA: mean power of the last 0.5 s / mean power of the last 10 s
    sta = np.full(n, np.nan); lta = np.full(n, np.nan)
    i = np.arange(n)
    sta[STA - 1:] = (c2[i[STA - 1:] + 1] - c2[i[STA - 1:] + 1 - STA]) / STA
    lta[LTA - 1:] = (c2[i[LTA - 1:] + 1] - c2[i[LTA - 1:] + 1 - LTA]) / LTA
    stalta = sta / np.maximum(lta, 1e-12)
    out["STA/LTA"] = (stalta, np.nanpercentile(stalta[max(FA, LTA - 1):FB], Q))
    # sliding std of the last W points
    c1 = np.r_[0.0, np.cumsum(s)]
    sd = np.full(n, np.nan)
    m1 = (c1[i[W - 1:] + 1] - c1[i[W - 1:] + 1 - W]) / W
    m2 = (c2[i[W - 1:] + 1] - c2[i[W - 1:] + 1 - W]) / W
    sd[W - 1:] = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    out["sliding std"] = (sd, np.nanpercentile(sd[FA + W - 1:FB], Q))
    # matrix profile (semi-supervised, as TSB-UAD): z-normalised distance of each W-point
    # subsequence to its nearest subsequence in the fitting part; put on the subsequence's END point.
    # Inside the fitting part: self-join of the fitting part (trivial matches excluded).
    fit = s[FA:FB]
    ab = stumpy.stump(s, W, T_B=fit, ignore_trivial=False)[:, 0].astype(float)
    self_mp = stumpy.stump(fit, W)[:, 0].astype(float)
    mp = np.full(n, np.nan)
    mp[W - 1:] = ab
    mp[FA + W - 1:FB] = self_mp
    out["MP"] = (mp, np.nanpercentile(self_mp, Q))
    return {k: (np.nan_to_num(v[0], nan=0.0).astype(np.float32), float(v[1])) for k, v in out.items()}


def baselines() -> None:
    """copula-ad env: STA/LTA, sliding std, MP. Saves per-point scores + thresholds per method."""
    from multiprocessing import Pool
    w = np.load(PROC / "waves100.npz")
    n = len(w.files)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    PARTS.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS) as pool:
        for c, a in enumerate(range(0, n, CHUNK)):
            if all(part_file(base_key(b), c).exists() for b in BASELINES):
                continue
            ks = range(a, min(a + CHUNK, n))
            res = pool.map(_baselines_one, [w[str(k)][0].astype(float) for k in ks])
            for name in BASELINES:
                np.savez_compressed(part_file(base_key(name), c), keys=np.array([str(k) for k in ks]),
                                    threshold=np.array([r[name][1] for r in res]),
                                    **{str(k): r[name][0] for k, r in zip(ks, res)})
            print(f"baselines {ks[-1] + 1}/{n}", flush=True)
    print("baselines done", flush=True)


def base_key(name: str) -> str:
    return name.replace("/", "").replace(" ", "_") + TAG


def bk() -> None:
    """seis env (ObsPy): Baer-Kradolfer P picker (ObsPy pk_baer, its documented example settings),
    run on the Z channel from the end of the fitting part on; only the past is used. One pick per
    time series (sample index in the 100 Hz time series, NaN if none)."""
    from obspy.signal.trigger import pk_baer
    w = np.load(PROC / "waves100.npz")
    picks = []
    for k in range(len(w.files)):
        z = w[str(k)][0].astype(np.float32)
        pk, _ = pk_baer(z[FB - 1000:], SR, 20, 60, 7.0, 12.0, 100, 100)  # 10 s before the test part as warm-up
        picks.append(FB - 1000 + pk if pk > 0 else np.nan)
    np.save(OUT / f"picks_BK{TAG}.npy", np.array(picks, float))
    print("BK done", flush=True)


EVAL_Q = float(os.environ["EVAL_Q"]) if os.environ.get("EVAL_Q") else None  # other level, no refit


def _thresholds_at(name, key, c, scores, q) -> np.ndarray:
    """Threshold at percentile q from saved values only: IVR from its 2000 bootstrap R_w,
    baselines from their own saved scores inside the fitting part (same ranges as when fitted)."""
    if name == "IVR":
        return np.percentile(np.load(part_file(key, c))["boot"], q, axis=1)
    lo = {"STA/LTA": max(FA, LTA - 1), "sliding std": FA + W - 1, "MP": FA + W - 1}[name]
    return np.array([np.percentile(s[lo:FB], q) for s in scores])


def evaluate() -> None:
    from scipy.stats import rankdata
    meta = pd.read_csv(RAW / "metadata.csv", low_memory=False)
    p = np.round(meta["first_p_sample"] / meta["trace_sampling_rate_hz"] * SR).astype(int).to_numpy()
    s_pick = np.round(meta["first_s_sample"] / meta["trace_sampling_rate_hz"] * SR).to_numpy()
    e = np.where(TO_S & ~np.isnan(s_pick), s_pick, p + AFTER).astype(int)  # end of the earthquake part
    rows = []
    keys = {"IVR": f"IVR{TAG}", "EQTransformer": "EQTransformer", "CRED": "CRED", "DPP": "DPP"}
    keys |= {b: base_key(b) for b in BASELINES}
    done = {}
    for name, key in keys.items():
        y_all, s_all, b_all, start, early, delay, used = [], [], [], [], [], [], []
        for c in range(n_parts()):
            part = load_part(key, c)
            if part is None:
                continue
            if EVAL_Q is not None and name not in ("EQTransformer", "CRED", "DPP"):
                part = (part[0], part[1], _thresholds_at(name, key, c, part[1], EVAL_Q))
            for k, sc, th in zip(*part):  # only the tested part of each time series is kept
                seg = slice(EA, e[k])
                y = np.arange(EA, e[k]) >= p[k]
                s = sc[seg]
                b = s > th
                y_all.append(y); s_all.append(s); b_all.append(b); used.append(k)
                on = np.flatnonzero(sc[p[k] - 50:e[k]] > th)
                start.append((on[0] - 50) / SR if len(on) else np.nan)
                after = np.flatnonzero(sc[p[k]:e[k]] > th)
                delay.append(after[0] / SR if len(after) else np.nan)
                early.append((sc[EA:p[k] - 50] > th).any())
        if not used:
            continue
        done[name] = key
        y, s, b = np.concatenate(y_all), np.concatenate(s_all), np.concatenate(b_all)
        r = rankdata(s)
        auc = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
        tp, fp, fn = (b & y).sum(), (b & ~y).sum(), (~b & y).sum()
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        start, delay = np.array(start), np.array(delay)
        rows.append({"method": name, "AUC_points": auc, "precision": prec, "recall": rec,
                     "F1": 2 * prec * rec / max(prec + rec, 1e-12),
                     "before_P_points_flagged_%": b[~y].mean() * 100,
                     "series_flag_earlier_than_P-0.5s_%": np.mean(early) * 100,
                     "eq_found_%": np.mean(~np.isnan(delay)) * 100,
                     "mean_delay_P_to_flag_s": np.nanmean(delay),
                     # latency: time from P to the first flag at or after P (flags before P = false positives)
                     "delay_median_s": np.nanmedian(delay), "delay_p90_s": np.nanpercentile(delay, 90),
                     "detected_within_0.2s_%": (delay <= 0.2).mean() * 100,
                     "detected_within_0.5s_%": (delay <= 0.5).mean() * 100,
                     "detected_within_1s_%": (delay <= 1).mean() * 100,
                     "start_within_0.1s_%": (np.abs(start) <= 0.1).mean() * 100,
                     "start_within_0.5s_%": (np.abs(start) <= 0.5).mean() * 100, "n": len(used),
                     "earthquake_seconds_median": np.median((e[used] - p[used]) / SR)})
    if (OUT / f"picks_BK{TAG}.npy").exists():  # Baer-Kradolfer: one pick per time series
        pk = np.load(OUT / f"picks_BK{TAG}.npy")
        d = (pk - p) / SR
        ok = (pk >= p - 50) & (pk < e)
        delay = np.where(ok & (pk >= p), d, np.nan)
        rows.append({"method": "Baer-Kradolfer", "eq_found_%": ok.mean() * 100,
                     "series_flag_earlier_than_P-0.5s_%": (pk < p - 50).mean() * 100,
                     "mean_delay_P_to_flag_s": np.nanmean(delay), "delay_median_s": np.nanmedian(delay),
                     "delay_p90_s": np.nanpercentile(delay, 90),
                     "detected_within_0.2s_%": (delay <= 0.2).mean() * 100,
                     "detected_within_0.5s_%": (delay <= 0.5).mean() * 100,
                     "detected_within_1s_%": (delay <= 1).mean() * 100,
                     "start_within_0.1s_%": (np.abs(d) <= 0.1).mean() * 100,
                     "start_within_0.5s_%": (np.abs(d) <= 0.5).mean() * 100, "n": len(p)})
    out = pd.DataFrame(rows)
    if EVAL_Q is not None:  # Baer-Kradolfer has no threshold level: same row as before
        out.to_csv(OUT / f"summary{TAG}_level{100 - EVAL_Q:g}pct.csv", index=False)
        print(out.round(3).T.to_string())
        return
    out.to_csv(OUT / f"summary{TAG}.csv", index=False)
    print(out.round(2).T.to_string())
    plots(meta, p, e, done)


def plots(meta, p, e, methods: dict) -> None:
    """methods: name -> part key. Same 5 time series every time (seed 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = np.load(PROC / "waves100.npz")
    for j, k in enumerate(np.random.default_rng(0).choice(len(p), 5, replace=False), 1):
        z = w[str(k)][0]
        t = np.arange(len(z)) / SR
        test = (np.arange(len(z)) >= EA) & (np.arange(len(z)) < e[k])
        fig, ax = plt.subplots(1 + len(methods), 1, figsize=(12, 2 * (1 + len(methods))), sharex=True)
        ax[0].plot(t, z, lw=0.4, color="k")
        ax[0].axvspan(FA / SR, FB / SR, color="tab:green", alpha=0.2, label=f"IVR fitting part ({FA // SR}-{FB // SR} s)")
        ax[0].axvspan(EA / SR, (e[k]) / SR, color="tab:gray", alpha=0.12, label=f"tested part ({EA // SR} s -> {'S (or P + 5 s)' if TO_S else 'P + 5 s'})")
        ax[0].axvspan(p[k] / SR, (e[k]) / SR, color="tab:red", alpha=0.2, label=f"truth: earthquake (P to {'S' if TO_S else 'P + 5 s'})")
        ax[0].set_title(f"ETHZ raw Z, resampled to 100 Hz (M {meta.source_magnitude[k]}, {meta.trace_name[k]})", fontsize=10)
        ax[0].legend(loc="upper right", fontsize=8)
        for a, name in zip(ax[1:], methods):
            part = load_part(methods[name], k // CHUNK)
            if part is None:
                continue
            i = part[0].index(k)
            v, thr = part[1][i], {k: part[2][i]}
            a.plot(t, v, lw=0.6, color="tab:blue")
            a.axhline(thr[k], color="tab:orange", ls="--", lw=1, label=f"threshold {thr[k]:.3g}")
            a.fill_between(t, 0, 1, where=(v > thr[k]) & test, transform=a.get_xaxis_transform(),
                           color="tab:orange", alpha=0.25, label="output = 1 (tested part)")
            a.axvline(p[k] / SR, color="tab:red", lw=0.8)
            if name in BASELINES:
                a.set_yscale("log")
                a.set_title(f"{name} (Z channel, window end point): score (log scale)", fontsize=10)
            elif name == "IVR":
                a.set_yscale("log")
                a.set_title(f"IVR, 50-point window, bootstrap {100 - Q:g}% threshold, score on window END point: R_w (log scale)", fontsize=10)
            else:
                a.set_ylim(-0.02, 1.02)
                a.set_title(f"{name} (ETHZ weights): earthquake score", fontsize=10)
            a.legend(loc="upper right", fontsize=8)
        ax[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(FIGS / f"ethz_raw{TAG}_example_{j}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    {"models": models, "ivr": ivr, "baselines": baselines, "bk": bk, "evaluate": evaluate}[MODE]()
