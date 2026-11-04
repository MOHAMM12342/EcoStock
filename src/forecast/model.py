"""Modèle de prévision de la demande (Braise).

Utilise LightGBM avec perte de Poisson pour modéliser les comptages de demande,
avec calcul des quantiles d'incertitude (q10, q50, q90) via la distribution de Poisson.
Fournit également une baseline (moyenne historique par créneau/produit/jour).
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from src.config import PRODUCTS

DEFAULT_MODEL_PATH = Path("models/demand_forecaster.joblib")


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
        self.baseline_counts: Dict[Tuple[str, int, int, int], int] = {}
        self.overall_product_means: Dict[str, float] = {}
        self.overall_product_counts: Dict[str, int] = {}
        self.days_trained: int = 0
        self.last_trained_at: Optional[str] = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "baseline_counts"):
            self.baseline_counts = {}
        if not hasattr(self, "overall_product_counts"):
            self.overall_product_counts = {}
        if not hasattr(self, "days_trained"):
            self.days_trained = 0
        if not hasattr(self, "last_trained_at"):
            self.last_trained_at = None

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
        grouped = df.groupby(["produit", "day_of_week", "hour", "minute"])["commandes"]
        self.baseline_lookup = grouped.mean().to_dict()
        self.baseline_counts = grouped.count().to_dict()

        overall_grp = df.groupby("produit")["commandes"]
        self.overall_product_means = overall_grp.mean().to_dict()
        self.overall_product_counts = overall_grp.count().to_dict()
        self.last_trained_at = datetime.utcnow().isoformat()

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

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Sauvegarde les modèles et la baseline sur disque (sérialisation des poids)."""
        target = Path(path) if path is not None else DEFAULT_MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "DemandForecaster":
        """Recharge les modèles et poids préalablement entraînés depuis le disque."""
        target = Path(path) if path is not None else DEFAULT_MODEL_PATH
        if not target.exists():
            raise FileNotFoundError(f"Fichier de modèle introuvable : {target}")
        loaded = joblib.load(target)
        if not isinstance(loaded, DemandForecaster):
            raise TypeError(f"L'objet chargé depuis {target} n'est pas une instance de DemandForecaster.")
        return loaded

    def update_daily(
        self,
        daily_df: pd.DataFrame,
        n_additional_trees: int = 15,
        learning_rate: float = 0.05,
        save_after: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> "DemandForecaster":
        """Entraînement continu de fin de journée sur les nouvelles données journalières.
        
        1. Met à jour la baseline historique (moyenne pondérée par créneau).
        2. Ajoute de nouveaux arbres de boosting LightGBM via init_model sans réentraîner de zéro.
        3. Sauvegarde automatiquement les poids mis à jour.
        """
        if daily_df.empty:
            return self

        # 1. Mise à jour incrémentale de la baseline
        for (prod, dow, hr, mn), group in daily_df.groupby(["produit", "day_of_week", "hour", "minute"]):
            key = (str(prod), int(dow), int(hr), int(mn))
            new_sum = float(group["commandes"].sum())
            new_n = len(group)
            if key in self.baseline_lookup:
                old_mean = self.baseline_lookup[key]
                old_n = self.baseline_counts.get(key, 1)
                updated_n = old_n + new_n
                self.baseline_lookup[key] = (old_mean * old_n + new_sum) / updated_n
                self.baseline_counts[key] = updated_n
            else:
                self.baseline_lookup[key] = new_sum / new_n
                self.baseline_counts[key] = new_n

        for prod, group in daily_df.groupby("produit"):
            p_name = str(prod)
            new_sum = float(group["commandes"].sum())
            new_n = len(group)
            if p_name in self.overall_product_means:
                old_mean = self.overall_product_means[p_name]
                old_n = self.overall_product_counts.get(p_name, 1)
                up_n = old_n + new_n
                self.overall_product_means[p_name] = (old_mean * old_n + new_sum) / up_n
                self.overall_product_counts[p_name] = up_n
            else:
                self.overall_product_means[p_name] = new_sum / new_n
                self.overall_product_counts[p_name] = new_n

        # 2. Enrichissement des features avec la nouvelle baseline
        df_copy = daily_df.copy()
        df_copy["baseline_mu"] = [self._get_baseline_for_row(row) for _, row in df_copy.iterrows()]

        # 3. Boosting continu (init_model) pour chaque produit
        for prod_name in PRODUCTS:
            p_df = df_copy[df_copy["produit"] == prod_name].copy()
            if p_df.empty:
                continue

            X_daily = p_df[FEATURE_COLUMNS]
            y_daily = p_df["commandes"].values

            if prod_name in self.models and hasattr(self.models[prod_name], "booster_"):
                # Entraînement continu sur le booster existant
                old_model = self.models[prod_name]
                ds_daily = lgb.Dataset(X_daily, label=y_daily, free_raw_data=False)
                new_booster = lgb.train(
                    {
                        "objective": "poisson",
                        "learning_rate": learning_rate,
                        "min_child_samples": 2,
                        "verbosity": -1,
                        "random_state": 42,
                    },
                    ds_daily,
                    num_boost_round=n_additional_trees,
                    init_model=old_model.booster_,
                    keep_training_booster=True,
                )
                old_model._Booster = new_booster
                old_model.n_estimators = new_booster.num_trees()
            else:
                # Premier entraînement pour ce produit s'il n'existait pas encore
                model = lgb.LGBMRegressor(
                    objective="poisson",
                    n_estimators=n_additional_trees,
                    learning_rate=learning_rate,
                    min_child_samples=2,
                    verbosity=-1,
                    random_state=42,
                )
                model.fit(X_daily, y_daily)
                self.models[prod_name] = model

        self.days_trained += 1
        self.last_trained_at = datetime.utcnow().isoformat()

        if save_after:
            self.save(save_path)

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
