"""Modèle de prévision de la demande (Braise).

Utilise LightGBM avec perte de Poisson pour modéliser les comptages de demande,
avec calcul des quantiles d'incertitude (q10, q50, q90) via la distribution de Poisson.
Fournit également une baseline (moyenne historique par créneau/produit/jour).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from src.config import PRODUCTS


FEATURE_COLUMNS = [
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "is_student_night",
    "temperature",
    "rain_mm",
    "baseline_mu",
]
CATEGORICAL_FEATURES = ["day_of_week", "hour", "minute", "month"]


class DemandForecaster:
    """Modèle de prévision par produit combinant LightGBM Poisson et Baseline."""

    def __init__(self):
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.baseline_lookup: Dict[Tuple[str, int, int, int], float] = {}
        self.overall_product_means: Dict[str, float] = {}

    def _get_baseline_for_row(self, row: pd.Series) -> float:
        key = (
            str(row["produit"]),
            int(row["day_of_week"]),
            int(row["hour"]),
            int(row["minute"]),
        )
        if key in self.baseline_lookup:
            return float(self.baseline_lookup[key])
        return float(self.overall_product_means.get(str(row["produit"]), 5.0))

    def fit(self, df: pd.DataFrame) -> "DemandForecaster":
        """Entraîne les modèles de Poisson (un par produit) et calcule la baseline."""
        # 1. Calcul de la baseline historique : moyenne par (produit, day_of_week, hour, minute)
        grouped = df.groupby(["produit", "day_of_week", "hour", "minute"])["commandes"].mean()
        self.baseline_lookup = grouped.to_dict()

        overall_means = df.groupby("produit")["commandes"].mean()
        self.overall_product_means = overall_means.to_dict()

        df_copy = df.copy()
        df_copy["baseline_mu"] = [self._get_baseline_for_row(row) for _, row in df_copy.iterrows()]

        # 2. Entraînement LightGBM Poisson par produit
        for prod_name in PRODUCTS:
            prod_df = df_copy[df_copy["produit"] == prod_name].copy()
            if prod_df.empty:
                continue

            X = prod_df[FEATURE_COLUMNS]
            y = prod_df["commandes"].values

            model = lgb.LGBMRegressor(
                objective="poisson",
                n_estimators=150,
                learning_rate=0.08,
                num_leaves=63,
                min_child_samples=5,
                verbosity=-1,
                random_state=42,
            )
            model.fit(X, y)
            self.models[prod_name] = model

        return self

    def predict_baseline(self, row: pd.Series) -> float:
        """Retourne la prédiction baseline (moyenne du créneau)."""
        return self._get_baseline_for_row(row)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prévoit la demande moyenne (mu) et les quantiles (q10, q50, q90).

        Contrat requis :
        DataFrame avec colonnes : [product, mu, q10, q50, q90, baseline_mu]
        """
        results: List[Dict] = []
        feat_copy = features.copy()
        if "baseline_mu" not in feat_copy.columns:
            feat_copy["baseline_mu"] = [self._get_baseline_for_row(r) for _, r in feat_copy.iterrows()]

        for _, row in feat_copy.iterrows():
            prod_name = str(row["produit"])
            model = self.models.get(prod_name)

            if model is not None:
                feat_df = pd.DataFrame([row[FEATURE_COLUMNS]])
                pred_mu = float(model.predict(feat_df)[0])
                pred_mu = max(0.01, pred_mu)
            else:
                pred_mu = self.predict_baseline(row)

            base_mu = self.predict_baseline(row)

            # Calcul des quantiles via la loi de Poisson de paramètre mu
            # Pour une demande discrète : stats.poisson.ppf
            q10 = float(stats.poisson.ppf(0.10, pred_mu))
            q50 = float(stats.poisson.ppf(0.50, pred_mu))
            q90 = float(stats.poisson.ppf(0.90, pred_mu))

            results.append({
                "product": prod_name,
                "mu": round(pred_mu, 2),
                "q10": round(q10, 2),
                "q50": round(q50, 2),
                "q90": round(q90, 2),
                "baseline_mu": round(base_mu, 2),
            })

        return pd.DataFrame(results)


# Instance globale pour le contrat standard forecast(features)
_GLOBAL_FORECASTER: Optional[DemandForecaster] = None


def set_global_forecaster(forecaster: DemandForecaster) -> None:
    global _GLOBAL_FORECASTER
    _GLOBAL_FORECASTER = forecaster


def forecast(features: pd.DataFrame) -> pd.DataFrame:
    """Contrat de module Braise : forecast(features) -> pd.DataFrame."""
    global _GLOBAL_FORECASTER
    if _GLOBAL_FORECASTER is None:
        # Initialise avec un forecaster par défaut
        _GLOBAL_FORECASTER = DemandForecaster()
    return _GLOBAL_FORECASTER.predict(features)
