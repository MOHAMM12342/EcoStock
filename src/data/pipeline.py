"""Pipeline de données réelles et découpage Train/Test (Braise).

Traite l'historique complet de 18 mois (944 451 tickets réels) extrait de
tickets_et_compositions_18_mois.zip :
- Extraction automatique du zip
- Jointure des tickets avec la table de composition (Hamb, Whopper, Master)
- Ré-échantillonnage temporel en créneaux de 30 minutes de service (10h00 - 23h30)
- Enrichissement contextuel (horaire, météo, calendrier, soirées étudiantes, fériés)
- Découpage chronologique strict Train (80% / 14 mois) et Test (20% / 4 mois)
- Cache compressé ultra-rapide (train_30min.csv.gz, test_30min.csv.gz)
"""

import io
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
ROOT_DIR = DATA_DIR.parent.parent
ZIP_PATH = DATA_DIR / "tickets_et_compositions_18_mois.zip"
EXTRACTED_DIR = DATA_DIR / "extracted"
TRAIN_DATA_PATH = DATA_DIR / "train_30min.csv.gz"
TEST_DATA_PATH = DATA_DIR / "test_30min.csv.gz"

FIXED_HOLIDAYS = {
    (1, 1),   # Jour de l'an
    (5, 1),   # Fête du Travail
    (5, 8),   # Victoire 1945
    (7, 14),  # Fête Nationale
    (8, 15),  # Assomption
    (11, 1),  # Toussaint
    (11, 11), # Armistice
    (12, 25), # Noël
}


def ensure_extracted() -> Tuple[Path, Path]:
    """S'assure que les fichiers tickets.csv et composition_produits.csv sont extraits."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    tickets_path = EXTRACTED_DIR / "tickets.csv"
    comp_path = EXTRACTED_DIR / "composition_produits.csv"

    if not tickets_path.exists() or not comp_path.exists():
        if not ZIP_PATH.exists():
            raise FileNotFoundError(f"Archive introuvable : {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACTED_DIR)

    return tickets_path, comp_path


def build_and_split_dataset(
    split_date: str = "2026-03-01",
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construit le dataset 30 min complet à partir des tickets réels et applique le split.

    Train : du 01/01/2025 au 28/02/2026 (424 jours, ~80%)
    Test  : du 01/03/2026 au 30/06/2026 (122 jours, ~20%)
    """
    if not force_recompute and TRAIN_DATA_PATH.exists() and TEST_DATA_PATH.exists():
        train_df = pd.read_csv(TRAIN_DATA_PATH, compression="gzip")
        test_df = pd.read_csv(TEST_DATA_PATH, compression="gzip")
        return train_df, test_df

    tickets_path, comp_path = ensure_extracted()

    # 1. Table de composition
    comp = pd.read_csv(comp_path)
    meat_map = {
        row["produit_id"]: (row["type_viande"], int(row["nombre_pieces_viande"]))
        for _, row in comp.iterrows()
    }

    def parse_items(json_str: str) -> Tuple[int, int, int]:
        hamb, whopper, master = 0, 0, 0
        items = json.loads(json_str)
        for it in items:
            pid = it.get("produit_id")
            qty = it.get("quantite", 1)
            if pid in meat_map:
                t, n = meat_map[pid]
                if t == "Hamb":
                    hamb += n * qty
                elif t == "Whopper":
                    whopper += n * qty
                elif t == "Master":
                    master += n * qty
        return hamb, whopper, master

    # 2. Lecture par chunks de tickets.csv
    all_times, all_h, all_w, all_m = [], [], [], []
    chunksize = 250000
    for chunk in pd.read_csv(tickets_path, chunksize=chunksize, usecols=["commande_a", "produits_json"]):
        parsed = [parse_items(j) for j in chunk["produits_json"]]
        arr = np.array(parsed)
        all_times.extend(chunk["commande_a"])
        all_h.extend(arr[:, 0])
        all_w.extend(arr[:, 1])
        all_m.extend(arr[:, 2])

    # 3. Normalisation temporelle et agrégation 30 min
    ts = pd.to_datetime(all_times, utc=True).tz_convert("Europe/Paris")
    df_tickets = pd.DataFrame({
        "timestamp": ts,
        "Hamb": all_h,
        "Whopper": all_w,
        "Master": all_m,
    })
    df_tickets = df_tickets.set_index("timestamp")
    df_30min = df_tickets.resample("30min").sum()

    # 4. Features d'entraînement
    rows = []
    for dt, r in df_30min.iterrows():
        if not (10 <= dt.hour <= 23):
            continue
        dow = int(dt.weekday())
        month = int(dt.month)
        is_we = 1 if dow in (5, 6) else 0
        is_hol = 1 if (dt.month, dt.day) in FIXED_HOLIDAYS else 0
        next_day = dt + pd.Timedelta(days=1)
        is_hol_eve = 1 if (next_day.month, next_day.day) in FIXED_HOLIDAYS else 0
        is_student = 1 if (dow in (3, 4) and dt.hour >= 19) else 0

        # Proxy météo réaliste (saisonnier)
        temp = 12.0 + 10.0 * np.sin(2 * np.pi * (month - 4) / 12) + (dt.hour - 10) * 0.5
        rain = 2.5 if (dt.day % 7 == 0) else 0.0

        for prod in ["Hamb", "Whopper", "Master"]:
            rows.append({
                "periode_30min": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": dt.strftime("%Y-%m-%d"),
                "produit": prod,
                "hour": int(dt.hour),
                "minute": int(dt.minute),
                "day_of_week": dow,
                "month": month,
                "is_weekend": is_we,
                "is_holiday": is_hol,
                "is_holiday_eve": is_hol_eve,
                "is_student_night": is_student,
                "temperature": round(float(temp), 1),
                "rain_mm": float(rain),
                "commandes": int(r[prod]),
            })

    df_all = pd.DataFrame(rows)

    # 5. Découpage Train / Test chronologique
    train_df = df_all[df_all["date"] < split_date].copy().reset_index(drop=True)
    test_df = df_all[df_all["date"] >= split_date].copy().reset_index(drop=True)

    # Sauvegarde en cache
    train_df.to_csv(TRAIN_DATA_PATH, compression="gzip", index=False)
    test_df.to_csv(TEST_DATA_PATH, compression="gzip", index=False)

    return train_df, test_df


def get_train_data() -> pd.DataFrame:
    """Retourne le jeu d'entraînement (14 mois de tickets réels)."""
    train_df, _ = build_and_split_dataset()
    return train_df


def get_test_data() -> pd.DataFrame:
    """Retourne le jeu de test / simulation quotidienne (4 mois hors échantillon)."""
    _, test_df = build_and_split_dataset()
    return test_df


def get_test_dates() -> List[str]:
    """Liste ordonnée des dates disponibles dans le split de test."""
    test_df = get_test_data()
    return sorted(test_df["date"].unique().tolist())


def get_daily_test_data(date_str: Optional[str] = None) -> pd.DataFrame:
    """Extrait les 28 créneaux de 30 min pour une journée de test donnée (par défaut la première)."""
    test_df = get_test_data()
    available_dates = get_test_dates()
    if not available_dates:
        raise ValueError("Aucune date de test disponible.")

    target_date = date_str if date_str in available_dates else available_dates[0]
    return test_df[test_df["date"] == target_date].copy().reset_index(drop=True)

