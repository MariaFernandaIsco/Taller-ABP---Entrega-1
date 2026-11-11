"""Validacion walk-forward (ventana expansiva): reentrena cada BLOCK semanas y pronostica 1 semana adelante."""
import json, numpy as np, pandas as pd, torch
from model import *
torch.set_num_threads(2)
SEEDS, FIRST_TEST, BLOCK = 5, 35, 5

def walk_forward(arr, cell):
    X, y = make_windows(arr)
    preds = np.zeros((len(X) - FIRST_TEST, y.shape[1]))
    for s in range(FIRST_TEST, len(X), BLOCK):
        e = min(s + BLOCK, len(X))
        n_val = max(6, int(0.2 * s))
        Xtr, ytr, Xva, yva = X[:s - n_val], y[:s - n_val], X[s - n_val:s], y[s - n_val:s]
        ms = [fit(Xtr, ytr, Xva, yva, cell=cell, seed=k) for k in range(SEEDS)]
        preds[s - FIRST_TEST:e - FIRST_TEST] = predict(ms, X[s:e])
    return preds

def main():
    ser = load_series(); arr = ser.values.astype(np.float32)
    X, y = make_windows(arr); yt, Xt = y[FIRST_TEST:], X[FIRST_TEST:]
    res, P = {}, {}
    for cell in ["gru", "lstm"]:
        P[cell] = walk_forward(arr, cell); res[cell.upper()] = metrics(yt, P[cell])
    # baselines (media historica = media expansiva solo con pasado)
    hist = np.stack([y[:FIRST_TEST + i].mean(0) for i in range(len(yt))])
    B = baselines(Xt, None); B["media_historica"] = hist
    for k, v in B.items(): P[k] = v; res[k] = metrics(yt, v)
    res = {k: {m: float(x) for m, x in v.items()} for k, v in res.items()}
    pd.DataFrame(res).T.round(3).to_csv(str(ROOT / "reports/metricas_walk_forward.csv"))
    json.dump(res, open(str(ROOT / "reports/metricas_walk_forward.json"), "w"), indent=2)
    np.savez(str(ROOT / "reports/predicciones_walk_forward.npz"), y=yt, **P)
    print(pd.DataFrame(res).T.round(3))
    # intervalo bootstrap del MAE_total: GRU - media_movil_4
    rng = np.random.default_rng(0); n = len(yt); d = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        d.append(np.abs(yt[i].sum(1) - P["gru"][i].sum(1)).mean() - np.abs(yt[i].sum(1) - P["media_movil_4"][i].sum(1)).mean())
    print("MAE_total GRU - MM4: media %.3f IC95 [%.3f, %.3f]" % (np.mean(d), *np.percentile(d, [2.5, 97.5])))
    json.dump({"dif_mae_total_gru_vs_mm4": [float(np.mean(d)), *map(float, np.percentile(d, [2.5, 97.5]))],
               "n_test": int(n)}, open(str(ROOT / "reports/bootstrap.json"), "w"))
if __name__ == "__main__": main()
