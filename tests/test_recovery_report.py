import numpy as np
import pandas as pd

from src.recovery.report import build_report, error_correlations, model_recovery_summary


def _model_df(correct=(10, 10, 6)):
    rows = []
    for m, n_ok in zip(("M1", "M2", "M3"), correct):
        for i in range(10):
            win = m if i < n_ok else ("M2" if m != "M2" else "M1")
            b = {"M1": 100.0, "M2": 100.0, "M3": 100.0}
            b[win] -= 25.0
            rows.append({"true_model": m, "winning_model": win, "bic_M1": b["M1"], "bic_M2": b["M2"], "bic_M3": b["M3"]})
    return pd.DataFrame(rows)


def test_model_recovery_summary_applies_the_85_percent_criterion():
    s = model_recovery_summary(_model_df((10, 10, 9)))
    assert s["passes"] and s["recovery_rate"]["M3"] == 0.9 and np.isclose(s["overall"], 29 / 30)
    s2 = model_recovery_summary(_model_df((10, 10, 8)))
    assert not s2["passes"]
    assert np.allclose(s["confusion"].sum(axis=1), 1.0) and np.allclose(s["inversion"].sum(axis=0), 1.0)


def test_error_correlations_detect_a_built_in_tradeoff():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    df = pd.DataFrame({"n_trials": 1000, "true_z": 0.0, "true_v_b": 0.0, "fit_z": z, "fit_v_b": -z + 0.1 * rng.normal(size=200)})
    ec = error_correlations(df)
    assert ec.loc[0, "error_corr"] < -0.9


def test_build_report_from_csvs(tmp_path):
    _model_df().to_csv(tmp_path / "model_recovery_bics.csv", index=False)
    rng = np.random.default_rng(1)
    t = rng.uniform(size=60)
    pd.DataFrame({"sim_idx": range(60), "n_trials": [200] * 30 + [500] * 30, "true_z": t, "fit_z": t + 0.05 * rng.normal(size=60),
                  "true_v_b": t[::-1], "fit_v_b": t[::-1] + 0.05 * rng.normal(size=60)}).to_csv(tmp_path / "param_recovery_JOINT.csv", index=False)
    text = build_report(str(tmp_path))
    assert "NOT MET" in text and "z vs v_b" in text and "JOINT" in text


def test_recovery_by_effect_size_splits_into_terciles():
    from src.recovery.report import recovery_by_effect_size
    rows = []
    for i in range(30):
        ok = i >= 10                                   # weakest third (smallest b_max) is misclassified
        rows.append({"true_model": "M3", "winning_model": "M3" if ok else "M2", "true_b_max": 0.2 + i * 0.04,
                     "bic_M1": 120.0, "bic_M2": 100.0 if not ok else 110.0, "bic_M3": 105.0 if not ok else 100.0})
    out = recovery_by_effect_size(pd.DataFrame(rows))
    assert len(out) == 3 and out["recovered"].tolist() == [0.0, 1.0, 1.0]
