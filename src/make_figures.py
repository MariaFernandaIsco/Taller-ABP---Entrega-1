import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import load_series, MAIN_ZONES, make_windows
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"gru": "#1b6ca8", "base": "#c9732c", "real": "#222222"}
ser = load_series(); tot = ser.sum(axis=1)
# 1. serie total + por zona
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tot.index, tot.values, color=C["real"], lw=1.5); ax[0].set_title("Casos registrados por semana (total)")
ax[0].set_ylabel("casos")
import matplotlib.dates as mdates
ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
share = ser.sum().sort_values()
ax[1].barh([s.title() for s in share.index], share.values, color=C["gru"]); ax[1].set_title("Casos por zona (ene 2025 - may 2026)")
for i, v in enumerate(share.values): ax[1].text(v + 5, i, str(int(v)), va="center", fontsize=9)
plt.tight_layout(); plt.savefig("reports/figures/01_serie_y_zonas.png"); plt.close()
# 2. walk-forward
z = np.load("reports/predicciones_walk_forward.npz"); n = len(z["y"])
idx = ser.index[8 + 35: 8 + 35 + n]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(idx, z["y"].sum(1), color=C["real"], lw=2, label="Real")
ax.plot(idx, z["gru"].sum(1), color=C["gru"], lw=2, label="GRU")
ax.plot(idx, z["media_movil_4"].sum(1), color=C["base"], lw=1.5, ls="--", label="Media movil 4 sem.")
ax.set_title("Validacion walk-forward: casos totales por semana (pronostico a 1 semana)"); ax.set_ylabel("casos"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("reports/figures/02_walk_forward_total.png"); plt.close()
# 3. metricas
m = pd.read_csv("reports/metricas_walk_forward.csv", index_col=0)
order = ["GRU", "LSTM", "media_historica", "media_movil_8", "media_movil_4", "ultima_semana"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, col, t in zip(ax, ["MAE", "Poisson_dev"], ["MAE por zona-semana (menor = mejor)", "Devianza de Poisson (menor = mejor)"]):
    v = m.loc[order, col]; a.barh(order[::-1], v.values[::-1], color=[C["gru"] if o in ("GRU","LSTM") else C["base"] for o in order[::-1]])
    a.set_title(t)
    for i, x in enumerate(v.values[::-1]): a.text(x, i, f" {x:.2f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig("reports/figures/03_metricas.png"); plt.close()
# 4. MAE por zona
y = z["y"]; zs = [s.title() for s in ser.columns]
mz = pd.DataFrame({k: np.abs(y - z[k]).mean(0) for k in ["gru", "media_movil_4", "media_historica"]}, index=zs)
mz.to_csv("reports/mae_por_zona.csv"); print(mz.round(2))
fig, ax = plt.subplots(figsize=(10, 4)); mz.plot.bar(ax=ax, color=[C["gru"], C["base"], "#8a8a8a"]); ax.set_ylabel("MAE"); ax.set_title("MAE por zona"); plt.xticks(rotation=20); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("reports/figures/04_mae_por_zona.png"); plt.close()
