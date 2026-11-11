"""Agrega el Excel del Observatorio CRIC a conteos semana x zona.
El archivo crudo contiene datos personales (nombres, cedulas, telefonos): NO se versiona.
Uso: python src/prepare_data.py ruta/OBSERVATORIO_2026.xlsx
"""
import sys, re, unicodedata, warnings
import pandas as pd
warnings.filterwarnings("ignore")
SHEET = "OBSERVATORIO 2025 2026 AIC"
CUTOFF = pd.Timestamp("2026-06-30")   # corte del boletin (ene 2025 - jun 2026)
# Las semanas posteriores a esta fecha estan incompletas (rezago de registro: 0 casos la semana del 25-may-2026
# frente a ~16 casos/semana en promedio), se excluyen del modelado.
LAST_COMPLETE_WEEK = pd.Timestamp("2026-05-18")

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper().strip()
    return re.sub(r"\s+", " ", s)

ZONA_MAP = {"SATH TAMA KIWE": "SAT TAMA KIWE", "REASENTAMIENTOS": "REASENTAMIENTO"}

def main(path, out="data/processed/casos_semana_zona.csv"):
    raw = pd.read_excel(path, sheet_name=SHEET, header=None, skiprows=6, usecols=range(26))
    d = raw[[0, 4, 8, 16, 17]].copy()
    d.columns = ["id", "rango_edad", "zona", "tipo", "fecha"]      # solo columnas no identificables
    d = d[d.id.notna()]
    d["fecha"] = pd.to_datetime(d.fecha, errors="coerce")
    n0 = len(d)
    d = d.dropna(subset=["fecha"])
    n_fuera = (d.fecha > CUTOFF).sum()
    d = d[d.fecha <= CUTOFF]
    for c in ["zona", "tipo", "rango_edad"]:
        d[c] = d[c].map(norm).replace({"NAN": "SIN DATO", "": "SIN DATO"})
    d["zona"] = d.zona.replace(ZONA_MAP)
    d["tipo"] = d.tipo.str.replace("VIOLENCIA ", "", regex=False)
    d["semana"] = d.fecha.dt.to_period("W-SUN").dt.start_time
    print(f"filas leidas {n0}; descartadas por fecha posterior al corte: {n_fuera}; usadas: {len(d)}")
    zonas = d.zona.value_counts()
    print(zonas.to_string())
    semanas = pd.date_range(d.semana.min(), d.semana.max(), freq="W-MON")
    piv = (d.groupby(["semana", "zona"]).size().unstack(fill_value=0)
             .reindex(semanas, fill_value=0))
    piv = piv[piv.index <= LAST_COMPLETE_WEEK]
    piv.index.name = "semana"
    piv.to_csv(out)
    tip = d.groupby(["semana", "tipo"]).size().unstack(fill_value=0).reindex(semanas, fill_value=0)
    tip = tip[tip.index <= LAST_COMPLETE_WEEK]
    tip.index.name = "semana"; tip.to_csv("data/processed/casos_semana_tipo.csv")
    print(piv.shape, "->", out)

if __name__ == "__main__":
    main(sys.argv[1])
