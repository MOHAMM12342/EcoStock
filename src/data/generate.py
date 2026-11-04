"""Génération de données synthétiques pour le projet Braise.

Produit un historique de ventes par tranche de 30 minutes (12 à 24 mois)
respectant la saisonnalité réaliste de la restauration rapide :
- Pointes de midi (12h-14h30) et du soir (20h30-22h30)
- Soirées étudiantes (jeudis et vendredis soirs)
- Jours fériés et veilles de fériés
- Facteurs météo (chaleur / pluie)
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import PRODUCTS, ProductConfig


# Jours fériés fixes en France
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


def is_french_holiday(dt: datetime) -> bool:
    """Vérifie si la date correspond à un jour férié."""
    return (dt.month, dt.day) in FIXED_HOLIDAYS


def is_french_holiday_eve(dt: datetime) -> bool:
    """Vérifie si la date correspond à une veille de jour férié."""
    next_day = dt + timedelta(days=1)
    return is_french_holiday(next_day)


def baseline_hourly_intensity(dt: datetime) -> float:
    """Courbe d'intensité de base en burgers / minute.

    Modélise :
    - Fond moyen en journée
    - Pic de midi centré à 12h45 (765 min)
    - Pic du soir centré à 21h20 (1280 min)
    - Période pré-soirée (19h-20h30)
    """
    minute_of_day = dt.hour * 60 + dt.minute

    # Heures de fermeture (01h00 - 10h30)
    if minute_of_day < 630 and minute_of_day >= 60:
        return 0.0

    # Fonction gaussienne
    def gauss(t, center, sigma):
        return np.exp(-((t - center) ** 2) / (2.0 * sigma * sigma))

    lunch_peak = 3.4 * gauss(minute_of_day, 765, 55)
    dinner_peak = 3.0 * gauss(minute_of_day, 1280, 50)
    pre_dinner = 0.6 if (1140 <= minute_of_day <= 1230) else 0.0

    return 0.9 + lunch_peak + dinner_peak + pre_dinner


def generate_30min_dataset(
    start_date: date = date(2025, 1, 1),
    days_count: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Génère un DataFrame d'historique de 30 min pour tous les produits."""
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    current_dt = datetime(start_date.year, start_date.month, start_date.day, 10, 30)
    end_dt = current_dt + timedelta(days=days_count)

    while current_dt < end_dt:
        # Service actif entre 10h30 et 23h30
        if 10 <= current_dt.hour <= 23:
            dow = current_dt.weekday()  # 0=Lundi, 6=Dimanche
            month = current_dt.month
            holiday = is_french_holiday(current_dt)
            holiday_eve = is_french_holiday_eve(current_dt)

            # Soirée étudiante : jeudi ou vendredi soir (à partir de 19h)
            student_night = (dow in (3, 4)) and (current_dt.hour >= 19)

            # Facteur jour de la semaine
            dow_factor = 1.0
            if dow == 4:  # Vendredi
                dow_factor = 1.25
            elif dow == 5:  # Samedi
                dow_factor = 1.40
            elif dow == 6:  # Dimanche
                dow_factor = 1.20
            elif dow == 3:  # Jeudi
                dow_factor = 1.10

            # Facteur férié / veille
            if holiday:
                dow_factor *= 1.35
            elif holiday_eve and current_dt.hour >= 18:
                dow_factor *= 1.30

            # Soirée étudiante
            if student_night:
                dow_factor *= 1.25

            # Météo simulée
            temp_c = 15.0 + 10.0 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 3)
            rain_mm = max(0.0, rng.exponential(0.5) - 0.3)
            rain_factor = 0.95 if rain_mm > 2.0 else 1.0

            # Intensité moyenne par minute sur le créneau
            mid_slot = current_dt + timedelta(minutes=15)
            lam_b = baseline_hourly_intensity(mid_slot) * dow_factor * rain_factor

            # Demande totale de burgers sur 30 minutes
            mean_burgers_30m = lam_b * 30.0

            for prod_name, pcfg in PRODUCTS.items():
                # Demande produit via Poisson
                prod_mean = mean_burgers_30m * pcfg.demand_share
                burgers_sold = int(rng.poisson(prod_mean))
                steaks_used = burgers_sold * pcfg.steaks_per_burger
                ca = burgers_sold * pcfg.price_euros

                rows.append({
                    "periode_30min": current_dt,
                    "produit": prod_name,
                    "hour": current_dt.hour,
                    "minute": current_dt.minute,
                    "day_of_week": dow,
                    "month": month,
                    "is_weekend": int(dow in (5, 6)),
                    "is_holiday": int(holiday),
                    "is_holiday_eve": int(holiday_eve),
                    "is_student_night": int(student_night),
                    "temperature": round(temp_c, 1),
                    "rain_mm": round(rain_mm, 2),
                    "commandes": burgers_sold,
                    "steaks_utilises": steaks_used,
                    "ca": round(ca, 2),
                })

        current_dt += timedelta(minutes=30)

    df = pd.DataFrame(rows)
    df.sort_values(by=["periode_30min", "produit"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

