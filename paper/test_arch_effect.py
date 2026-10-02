"""Is the residual dependency we keep finding actually an ARCH/GARCH
effect (volatility clustering: innovations uncorrelated but their SQUARES
are autocorrelated) rather than leftover mean-dynamics misspecification?

If raw residuals are autocorrelated -> classic "need more AR/MA" story.
If squared residuals are MORE/similarly autocorrelated -> ARCH effect:
no ARMA order can ever fix this, since ARMA only models the conditional
mean, not the conditional variance.

Tests, on the same pooled out-of-sample residuals used throughout:
  - Ljung-Box on raw residuals (what we've been reporting)
  - Ljung-Box on SQUARED residuals (manual check for volatility clustering)
  - Engle's ARCH-LM test (statsmodels.stats.diagnostic.het_arch) -- the
    formal, standard test for this exact question
"""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from algorithm import fit_reference_model
from plot_order_heatmap import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER, fit_pq
from plot_best_order_acf import pooled_residuals_for


def report(residuals, label: str) -> None:
    raw_lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    raw_p = float(raw_lb["lb_pvalue"].iloc[0])

    squared = residuals**2
    sq_lb = acorr_ljungbox(squared, lags=[10], return_df=True)
    sq_p = float(sq_lb["lb_pvalue"].iloc[0])
    sq_stat = float(sq_lb["lb_stat"].iloc[0])

    arch_stat, arch_p, _, _ = het_arch(residuals, nlags=10)

    print(f"-- {label} (n={len(residuals)}) --")
    print(f"   Raw residual Ljung-Box:      p={raw_p:.4f}  {'(white noise)' if raw_p > 0.05 else '(AUTOCORRELATED)'}")
    print(f"   Squared residual Ljung-Box:  p={sq_p:.4f}  stat={sq_stat:.1f}  {'(no volatility clustering)' if sq_p > 0.05 else '(VOLATILITY CLUSTERING / ARCH-like)'}")
    print(f"   Engle's ARCH-LM test:        p={arch_p:.4f}  stat={arch_stat:.1f}  {'(no ARCH effect)' if arch_p > 0.05 else '(ARCH EFFECT DETECTED)'}")
    print()


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]

    aic_model = fit_reference_model(train_values, 3, 3)
    aic_residuals = pooled_residuals_for(aic_model, test_values, WINDOW_SIZE)
    report(aic_residuals, f"AIC-method order {aic_model['order']}")

    best_model = fit_pq(train_values, 4, 4)
    best_residuals = pooled_residuals_for(best_model, test_values, WINDOW_SIZE)
    report(best_residuals, "Best grid order (4,0,4)")


if __name__ == "__main__":
    main()
