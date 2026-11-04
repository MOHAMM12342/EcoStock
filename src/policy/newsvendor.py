"""Politique de décision Newsvendor et règles de repli (Braise).

Contrat :
def decide(state: KitchenState, fc: pd.DataFrame, t: datetime) -> list[Action]: ...
"""

from datetime import datetime
import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    DEFAULT_CO,
    DEFAULT_CU,
    DEFAULT_TAU,
    FALLBACK_COOK_QTY,
    FALLBACK_TRIGGER_STOCK,
    LOOKAHEAD_MARGIN_MIN,
    MAX_HOLD_MINUTES,
    MIN_COOK_BATCH,
    PRODUCTS,
    TREND_K_MAX,
    TREND_K_MIN,
    ProductConfig,
)
from src.simulator.kitchen import Action, KitchenProductState, KitchenState


def compute_intra_day_trend(pstate: KitchenProductState, is_pos_connected: bool) -> float:
    """Calcule le facteur multiplicatif de tendance k sur les 15 dernières minutes.

    k = clip( (sum(réel) + 2) / (sum(prévu) + 2), k_min, k_max )
    Si la caisse est déconnectée, k = 1.0.
    """
    if not is_pos_connected:
        return 1.0

    actual_sum = sum(pstate.recent_actual_steaks) + 2.0
    forecast_sum = sum(pstate.recent_forecast_steaks) + 2.0

    ratio = actual_sum / forecast_sum
    return float(np.clip(ratio, TREND_K_MIN, TREND_K_MAX))


def calculate_newsvendor_target(
    mu_steaks: float,
    tau: float = DEFAULT_TAU,
) -> int:
    """Calcule le stock cible selon le quantile Newsvendor F^-1(tau) d'une loi de Poisson."""
    if mu_steaks <= 0:
        return 0
    # ppf donne le quantile pour la loi de Poisson de moyenne mu_steaks
    target = int(math.ceil(stats.poisson.ppf(tau, mu_steaks)))
    return max(0, target)


def compute_expected_waste(q: int, mu_steaks: float, max_k: int = 50) -> float:
    """Calcule l'espérance de perte E[max(q - D, 0)] pour une demande de Poisson."""
    if q <= 0 or mu_steaks <= 0:
        return 0.0
    k_vals = np.arange(0, q)
    probs = stats.poisson.pmf(k_vals, mu_steaks)
    return float(np.sum((q - k_vals) * probs))


def decide_newsvendor(
    state: KitchenState,
    forecast_df: pd.DataFrame,
    current_time: datetime,
    tau: float = DEFAULT_TAU,
) -> List[Action]:
    """Politique principale de l'agent Braise.

    1. Si caisse coupée : applique la règle fixe de sécurité (lancer 6 steaks si stock < 6).
    2. Sinon : calcule pour chaque produit le besoin Newsvendor sur la fenêtre de cuisson + 8 min,
       corrigé par la tendance intra-journée.
    """
    actions: List[Action] = []

    # Map des prévisions par produit (commandes par tranche de 30 min)
    fc_dict = {}
    if not forecast_df.empty:
        for _, row in forecast_df.iterrows():
            fc_dict[str(row["product"])] = float(row.get("mu", 5.0))

    candidates: List[tuple[int, Action]] = []

    for prod_name, pstate in state.products.items():
        # Respect du snooze (si la recommandation a été ignorée par le cuisinier)
        if pstate.snooze_until and current_time < pstate.snooze_until:
            continue

        pcfg = PRODUCTS[prod_name]
        lookahead_min = pcfg.cook_time_minutes + LOOKAHEAD_MARGIN_MIN

        # Stock disponible (lots ayant >= 3 min de vie + lots en cours de cuisson)
        usable_in_bin = sum(
            b.quantity
            for b in pstate.bin_batches
            if (MAX_HOLD_MINUTES - b.age_minutes) >= 3
        )
        total_supply = usable_in_bin + pstate.cooking_steaks_total

        # MODE DÉGRADÉ : caisse déconnectée
        if not state.is_pos_connected:
            if total_supply < FALLBACK_TRIGGER_STOCK:
                act = Action(
                    product=prod_name,
                    quantity=FALLBACK_COOK_QTY,
                    reason="Règle fixe de sécurité (caisse déconnectée, stock < 6)",
                    urgency_minutes=2,
                )
                candidates.append((FALLBACK_COOK_QTY, act))
            continue

        # MODE NORMAL : Newsvendor avec tendance
        base_30m_burgers = fc_dict.get(prod_name, 5.0)
        # Demande en burgers par minute
        burgers_per_min = base_30m_burgers / 30.0

        # Tendance récente
        k_trend = compute_intra_day_trend(pstate, is_pos_connected=True)

        # Demande attendue en steaks sur la fenêtre utile
        mu_steaks = burgers_per_min * lookahead_min * pcfg.steaks_per_burger * k_trend

        # Calcul de la cible via quantile Newsvendor
        target_steaks = calculate_newsvendor_target(mu_steaks, tau=tau)
        qty_needed = target_steaks - total_supply

        if qty_needed >= MIN_COOK_BATCH:
            pct_trend = (k_trend - 1.0) * 100
            trend_str = f"{'+' if pct_trend >= 0 else ''}{round(pct_trend)}%"
            reason = (
                f"Demande attendue : {round(mu_steaks)} steaks d'ici {lookahead_min} min, "
                f"{total_supply} prêts/en cuisson. Tendance 15 min : {trend_str}."
            )
            act = Action(
                product=prod_name,
                quantity=qty_needed,
                reason=reason,
                urgency_minutes=2,
            )
            candidates.append((qty_needed, act))

    if candidates:
        # Trier par quantité nécessaire la plus critique
        candidates.sort(key=lambda c: c[0], reverse=True)
        actions.append(candidates[0][1])

    return actions


def decide_fixed_policy(
    state: KitchenState,
    forecast_df: pd.DataFrame,
    current_time: datetime,
    target_buffer: int = 8,
) -> List[Action]:
    """Politique de référence fixe (baseline sans IA).

    Tente simplement de maintenir en permanence un buffer fixe de steaks prêts.
    """
    actions: List[Action] = []
    candidates: List[tuple[int, Action]] = []

    for prod_name, pstate in state.products.items():
        total_supply = pstate.ready_steaks_total + pstate.cooking_steaks_total
        deficit = target_buffer - total_supply
        if deficit >= MIN_COOK_BATCH:
            act = Action(
                product=prod_name,
                quantity=deficit,
                reason=f"Politique fixe : maintien d'un tampon de {target_buffer} steaks",
                urgency_minutes=2,
            )
            candidates.append((deficit, act))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        actions.append(candidates[0][1])

    return actions


def decide(state: KitchenState, fc: pd.DataFrame, t: datetime) -> List[Action]:
    """Contrat de module Braise : decide(state, fc, t) -> list[Action]."""
    return decide_newsvendor(state, fc, t)

