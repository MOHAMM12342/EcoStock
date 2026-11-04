"""Orchestrateur de simulation d'une journée complète (Braise).

Contrat :
def simulate_day(day: date, policy: Callable, seed: int) -> SimLogs: ...
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import PRODUCTS
from src.data.generate import (
    baseline_hourly_intensity,
    is_french_holiday,
    is_french_holiday_eve,
)
from src.evaluation.metrics import compute_operational_kpis
from src.forecast.model import DemandForecaster, forecast
from src.simulator.kitchen import (
    Action,
    KitchenState,
    SteakBatch,
    create_initial_kitchen_state,
    execute_action,
    step_minute,
)


@dataclass
class SimLogs:
    day: date
    policy_name: str
    journal_30min: List[Dict]
    kpis: Dict[str, float]
    state: KitchenState


def simulate_day(
    day: date,
    policy: Callable[[KitchenState, pd.DataFrame, datetime], List[Action]],
    seed: int = 42,
    forecaster: Optional[DemandForecaster] = None,
    surge_start_hour: Optional[float] = None,
    surge_multiplier: float = 2.2,
    pos_connected: bool = True,
    policy_name: str = "Agent Braise (Newsvendor)",
) -> SimLogs:
    """Simule une journée de service complète (10h30 - 23h30) à la minute."""
    rng = np.random.default_rng(seed)

    start_dt = datetime(day.year, day.month, day.day, 10, 30)
    end_dt = datetime(day.year, day.month, day.day, 23, 30)

    state = create_initial_kitchen_state(start_dt, pos_connected=pos_connected)

    # Initialisation d'un petit stock de départ dans le bac (ex: 6 steaks par produit)
    for pstate in state.products.values():
        pstate.bin_batches.append(
            SteakBatch(batch_id=state.next_batch_id(), quantity=6, age_minutes=3)
        )

    # Contexte calendaire
    dow = day.weekday()
    holiday = is_french_holiday(start_dt)
    holiday_eve = is_french_holiday_eve(start_dt)
    student_night = (dow in (3, 4))

    dow_factor = 1.0
    if dow == 4:
        dow_factor = 1.25
    elif dow == 5:
        dow_factor = 1.40
    elif dow == 6:
        dow_factor = 1.20
    elif dow == 3:
        dow_factor = 1.10

    if holiday:
        dow_factor *= 1.35
    elif holiday_eve:
        dow_factor *= 1.25

    journal_30min: List[Dict] = []
    current_dt = start_dt

    # Accumulateurs pour la tranche de 30 min en cours
    slot_start_dt = current_dt
    slot_revenue = 0.0
    slot_used_steaks = 0
    slot_wasted_steaks = 0
    slot_predicted_steaks = 0.0
    slot_actual_steaks = 0
    slot_fc_mu = 0.0
    slot_q10 = 0.0
    slot_q50 = 0.0
    slot_q90 = 0.0

    cached_slot_key = None
    cached_fc_df = None

    while current_dt < end_dt:
        # Vérification du coup de feu imprévu (surge)
        is_surge = False
        if surge_start_hour is not None:
            cur_hour_float = current_dt.hour + current_dt.minute / 60.0
            if surge_start_hour <= cur_hour_float < (surge_start_hour + 0.35):  # 20 min de pic
                is_surge = True

        surge_eff = surge_multiplier if is_surge else 1.0

        # Demande minute de base
        base_intensity = baseline_hourly_intensity(current_dt) * dow_factor
        if student_night and current_dt.hour >= 19:
            base_intensity *= 1.20

        # Prévision par le modèle pour ce créneau (mise en cache par créneau de 30 min)
        current_slot_key = (current_dt.hour, (current_dt.minute // 30) * 30)
        if current_slot_key != cached_slot_key or cached_fc_df is None:
            feature_row = pd.DataFrame([{
                "produit": "Whopper",
                "hour": current_dt.hour,
                "minute": current_slot_key[1],
                "day_of_week": dow,
                "month": day.month,
                "is_weekend": int(dow in (5, 6)),
                "is_holiday": int(holiday),
                "is_holiday_eve": int(holiday_eve),
                "is_student_night": int(student_night and current_dt.hour >= 19),
                "temperature": 18.0,
                "rain_mm": 0.0,
            }])

            fc_rows = []
            for p in PRODUCTS:
                f_row = feature_row.copy()
                f_row["produit"] = p
                fc_rows.append(f_row)
            all_features = pd.concat(fc_rows, ignore_index=True)

            if forecaster is not None:
                cached_fc_df = forecaster.predict(all_features)
            else:
                cached_fc_df = forecast(all_features)
            cached_slot_key = current_slot_key

        fc_df = cached_fc_df

        # 1. Décision de l'agent / politique
        actions = policy(state, fc_df, current_dt)
        for act in actions:
            execute_action(state, act)

        # 2. Tirage aléatoire des commandes réelles pour chaque produit
        orders: Dict[str, int] = {}
        for p_name, pcfg in PRODUCTS.items():
            mean_p = (base_intensity * surge_eff) * pcfg.demand_share
            orders[p_name] = int(rng.poisson(mean_p))

        # Dictionnaire des prévisions en steaks par minute
        fc_dict: Dict[str, float] = {}
        for _, r in fc_df.iterrows():
            prod = str(r["product"])
            cfg = PRODUCTS[prod]
            fc_dict[prod] = (float(r["mu"]) / 30.0) * cfg.steaks_per_burger

        # 3. Avancement de la simulation d'une minute
        tick_res = step_minute(state, orders, forecast_steaks=fc_dict)

        # Cumul tranche 30 min
        slot_revenue += tick_res.revenue_generated
        consumed_now = sum(tick_res.steaks_consumed.values())
        wasted_now = sum(tick_res.steaks_wasted.values())
        slot_used_steaks += consumed_now
        slot_wasted_steaks += wasted_now
        slot_actual_steaks += consumed_now
        slot_predicted_steaks += sum(fc_dict.values())

        if fc_df is not None and not fc_df.empty:
            slot_fc_mu = float(fc_df["mu"].sum())
            slot_q10 = float(fc_df["q10"].sum())
            slot_q50 = float(fc_df["q50"].sum())
            slot_q90 = float(fc_df["q90"].sum())

        # Clôture de la tranche de 30 minutes
        if current_dt.minute in (29, 59):
            # Calcul de la perte potentielle instantanée
            loss_potential = 0.0
            for p_name, pstate in state.products.items():
                pcfg = PRODUCTS[p_name]
                expected_10m = (fc_dict.get(p_name, 1.0)) * 10.0
                ready = pstate.ready_steaks_total
                loss_potential += max(0.0, ready - expected_10m)

            journal_30min.append({
                "periode_30min": slot_start_dt,
                "ca": round(slot_revenue, 2),
                "prevision_mu": round(slot_predicted_steaks, 1),
                "q10": round(slot_q10, 1),
                "q50": round(slot_q50, 1),
                "q90": round(slot_q90, 1),
                "reel": int(slot_actual_steaks),
                "erreur": round(slot_actual_steaks - slot_predicted_steaks, 1),
                "perte_potentielle": round(loss_potential, 0),
                "perte_reelle": int(slot_wasted_steaks),
            })

            # Réinitialisation pour la tranche suivante
            slot_start_dt = current_dt + timedelta(minutes=1)
            slot_revenue = 0.0
            slot_used_steaks = 0
            slot_wasted_steaks = 0
            slot_predicted_steaks = 0.0
            slot_actual_steaks = 0

        current_dt += timedelta(minutes=1)

    kpis = compute_operational_kpis(
        steaks_cooked=state.total_cooked_steaks,
        steaks_wasted=state.total_wasted_steaks,
        steaks_used=state.total_used_steaks,
        unmet_burgers=state.total_unmet_burgers,
        total_revenue=state.total_revenue_euros,
    )

    return SimLogs(
        day=day,
        policy_name=policy_name,
        journal_30min=journal_30min,
        kpis=kpis,
        state=state,
    )
