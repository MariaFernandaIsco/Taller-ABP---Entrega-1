"""Modelo, datos y entrenamiento: pronostico semanal de casos de VBG por zona (CRIC)."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

MAIN_ZONES = ["CXHAB WALA KIWE", "TIERRADENTRO", "ORIENTE", "SAT TAMA KIWE", "SUR"]
WINDOW = 8

def load_series(path=ROOT / "data/processed/casos_semana_zona.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    out = df[MAIN_ZONES].copy()
    out["OTRAS"] = df.drop(columns=MAIN_ZONES).sum(axis=1)
    return out

def make_windows(arr, window=WINDOW):
    X = np.stack([arr[i:i + window] for i in range(len(arr) - window)])
    y = arr[window:]
    return X.astype(np.float32), y.astype(np.float32)

class Forecaster(nn.Module):
    def __init__(self, n_series, hidden=16, cell="gru", dropout=0.2):
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(n_series, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_series)
    def forward(self, x):                          # x: conteos crudos (B, T, S)
        h, _ = self.rnn(torch.log1p(x))
        return nn.functional.softplus(self.head(self.drop(h[:, -1]))) + 1e-3   # tasa Poisson > 0

def fit(Xtr, ytr, Xva=None, yva=None, cell="gru", hidden=16, epochs=300, lr=5e-3,
        wd=1e-3, patience=40, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Forecaster(Xtr.shape[2], hidden, cell)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)
    Xt, yt = map(torch.tensor, (Xtr, ytr))
    best, best_state, bad = np.inf, None, 0
    m.history = {"train": [], "val": []}
    for ep in range(epochs):
        m.train(); opt.zero_grad(); loss = loss_fn(m(Xt), yt); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.history["train"].append(loss.item())
        if Xva is not None:
            m.eval()
            with torch.no_grad():
                v = loss_fn(m(torch.tensor(Xva)), torch.tensor(yva)).item()
            m.history["val"].append(v)
            if v < best - 1e-5: best, bad = v, 0; best_state = {k: t.clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state: m.load_state_dict(best_state)
    return m.eval()

def predict(models, X):
    with torch.no_grad():
        return np.mean([m(torch.tensor(X)).numpy() for m in models], axis=0)

def baselines(X, ytrain_mean):
    return {"ultima_semana": X[:, -1],
            "media_movil_4": X[:, -4:].mean(1),
            "media_movil_8": X.mean(1),
            "media_historica": np.tile(ytrain_mean, (len(X), 1))}

def metrics(y, p):
    mae = np.abs(y - p).mean()
    rmse = np.sqrt(((y - p) ** 2).mean())
    pp = np.clip(p, 1e-6, None)
    dev = 2 * np.mean(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / pp), 0) - (y - pp))
    return {"MAE": mae, "RMSE": rmse, "Poisson_dev": dev,
            "MAE_total": np.abs(y.sum(1) - p.sum(1)).mean()}
