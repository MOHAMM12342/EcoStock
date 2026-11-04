"""Métriques d'évaluation pour le projet Braise.

Calcule :
- Indicateurs opérationnels : Taux de gaspillage, Taux de rupture, Burgers en attente
- Indicateurs de prévision : WAPE, Biais, Pinball Loss, Couverture d'intervalles
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|actual - predicted|) / sum(actual)
    """
    total_actual = np.sum(actual)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / total_actual)


def compute_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Biais relatif moyen.

    Biais = sum(predicted - actual) / sum(actual)
    """
    total_actual = np.sum(actual)
    if total_actual == 0:
        return 0.0
    return float(np.sum(predicted - actual) / total_actual)


def compute_pinball_loss(actual: np.ndarray, predicted_quantile: np.ndarray, q: float) -> float:
    """Pinball loss pour un quantile q donné (ex: 0.10, 0.50, 0.90).

    L_q(y, y_hat) = mean(max(q * (y - y_hat), (q - 1) * (y - y_hat)))
    """
    diff = actual - predicted_quantile
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def compute_interval_coverage(actual: np.ndarray, lower_q: np.ndarray, upper_q: np.ndarray) -> float:
    """Couverture empirique de l'intervalle [lower_q, upper_q] (ex: [q10, q90])."""
    if len(actual) == 0:
        return 0.0
    inside = (actual >= lower_q) & (actual <= upper_q)
    return float(np.mean(inside))


def compute_operational_kpis(
    steaks_cooked: int,
    steaks_wasted: int,
    steaks_used: int,
    unmet_burgers: float,
    total_revenue: float,
) -> Dict[str, float]:
    """Calcule les KPI de performance en cuisine."""
    waste_rate = (steaks_wasted / steaks_cooked) if steaks_cooked > 0 else 0.0
    service_rate = (steaks_used / (steaks_used + unmet_burgers)) if (steaks_used + unmet_burgers) > 0 else 1.0

    return {
        "steaks_cuits": float(steaks_cooked),
        "steaks_jetes": float(steaks_wasted),
        "steaks_utilises": float(steaks_used),
        "taux_gaspillage_pct": round(waste_rate * 100, 2),
        "burgers_en_attente": round(unmet_burgers, 1),
        "taux_service_pct": round(service_rate * 100, 2),
        "chiffre_affaires_euros": round(total_revenue, 2),
    }

