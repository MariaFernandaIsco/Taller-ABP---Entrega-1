"""Entrena el modelo final (GRU, 5 semillas) con todos los datos y guarda los pesos en models/."""
import numpy as np, torch
from model import *
ser = load_series(); X, y = make_windows(ser.values.astype(np.float32)); n_val = 12
ms = [fit(X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:], seed=k) for k in range(5)]
torch.save({"state_dicts": [m.state_dict() for m in ms], "zonas": list(ser.columns), "window": WINDOW}, ROOT / "models/gru_vbg_cric.pt")
print("modelo guardado; pronostico semana siguiente:", dict(zip(ser.columns, predict(ms, ser.values[-WINDOW:][None].astype(np.float32))[0].round(1))))
