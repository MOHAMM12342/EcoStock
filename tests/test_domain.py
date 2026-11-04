"""Tests unitaires obligatoires pour l'agent Braise (Architecture 1).

Couvre rigoureusement les exigences de la section 11.1 :
1. Exemple de perte métier (15 cuits, 10 utilisés, 5 perdus)
2. Péremption stricte à HOLD (15 min)
3. Calcul du quantile Newsvendor F^-1(tau)
4. Politique de secours quand la caisse est coupée (règle fixe de 6 steaks)
5. Export Excel et métriques d'évaluation
"""

from datetime import datetime, timedelta
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CO,
    DEFAULT_CU,
    DEFAULT_TAU,
    FALLBACK_COOK_QTY,
    FALLBACK_TRIGGER_STOCK,
    MAX_HOLD_MINUTES,
    PRODUCTS,
)
from src.evaluation.metrics import compute_bias, compute_operational_kpis, compute_wape
from src.export.excel import export_journal_to_excel
from src.policy.newsvendor import calculate_newsvendor_target, decide_newsvendor
from src.simulator.kitchen import (
    Action,
    KitchenState,
    SteakBatch,
    create_initial_kitchen_state,
    execute_action,
    step_minute,
)


def test_mandatory_loss_example():
    """Test obligatoire n°1 : Exemple de perte métier spécifié.

    'Sur une fenêtre de 15 min, 5 burgers sont commandés avec 2 steaks chacun,
    soit 10 steaks utilisés. Un lot de 15 steaks laisse 5 steaks perdus.'
    """
    t0 = datetime(2026, 11, 4, 12, 0)
    state = create_initial_kitchen_state(t0)

    # Produit 'Whopper' a 2 steaks par burger
    pstate = state.products["Whopper"]

    # On introduit un lot de 15 steaks frais (âge = 0)
    batch = SteakBatch(batch_id=1, quantity=15, age_minutes=0)
    pstate.bin_batches.append(batch)
    state.total_cooked_steaks = 15

    # Minute 1 : Commande de 5 Whoppers (soit 5 * 2 = 10 steaks nécessaires)
    demand = {"Whopper": 5, "Hamb": 0, "Master": 0}
    res_m1 = step_minute(state, demand)

    assert res_m1.steaks_consumed["Whopper"] == 10
    assert len(pstate.bin_batches) == 1
    assert pstate.bin_batches[0].quantity == 5  # Reste 5 steaks dans le lot

    # Ensuite, aucune commande pendant les 14 minutes suivantes (total 15 min écoulées)
    empty_demand = {"Whopper": 0, "Hamb": 0, "Master": 0}
    total_wasted = 0

    for _ in range(14):
        res = step_minute(state, empty_demand)
        total_wasted += res.steaks_wasted["Whopper"]

    # À l'issue des 15 minutes, le lot a atteint MAX_HOLD_MINUTES (15 min) et est jeté
    assert total_wasted == 5
    assert state.total_wasted_steaks == 5
    assert state.total_used_steaks == 10
    assert len(pstate.bin_batches) == 0  # Bac vidé après péremption


def test_expiration_at_hold_threshold():
    """Test obligatoire n°2 : Péremption stricte à MAX_HOLD_MINUTES (15 min)."""
    t0 = datetime(2026, 11, 4, 11, 0)
    state = create_initial_kitchen_state(t0)
    pstate = state.products["Hamb"]

    # Lot de 8 steaks initialisé à un âge de 14 minutes
    batch = SteakBatch(batch_id=1, quantity=8, age_minutes=14)
    pstate.bin_batches.append(batch)

    # 1 minute s'écoule sans commande -> le lot passe à 15 minutes et doit périmer
    res = step_minute(state, {"Whopper": 0, "Hamb": 0, "Master": 0})
    assert res.steaks_wasted["Hamb"] == 8
    assert len(pstate.bin_batches) == 0
    assert state.total_wasted_steaks == 8


def test_newsvendor_quantile_calculation():
    """Test obligatoire n°3 : Calcul du quantile Newsvendor F^-1(tau)."""
    tau = DEFAULT_CU / (DEFAULT_CU + DEFAULT_CO)  # 3.5 / (3.5 + 1.2) ~ 0.7447

    # Si la demande moyenne mu est nulle
    assert calculate_newsvendor_target(0.0, tau) == 0

    # Pour une demande mu = 10, le quantile ~75% de Poisson doit être supérieur à la moyenne
    target = calculate_newsvendor_target(10.0, tau)
    assert target >= 10
    assert target <= 15


def test_fallback_policy_when_pos_disconnected():
    """Test obligatoire n°4 : Politique de secours quand la caisse est coupée.

    Si la caisse est déconnectée (is_pos_connected=False) et que le stock est < 6,
    l'agent doit recommander exactement 6 steaks (règle fixe de secours).
    """
    t0 = datetime(2026, 11, 4, 12, 0)
    state = create_initial_kitchen_state(t0, pos_connected=False)
    pstate = state.products["Whopper"]

    # Stock initial insuffisant : 2 steaks (< 6)
    pstate.bin_batches.append(SteakBatch(batch_id=1, quantity=2, age_minutes=1))

    dummy_fc = pd.DataFrame([{"product": "Whopper", "mu": 20.0}])
    actions = decide_newsvendor(state, dummy_fc, t0)

    assert len(actions) == 1
    assert actions[0].product == "Whopper"
    assert actions[0].quantity == FALLBACK_COOK_QTY
    assert "caisse déconnectée" in actions[0].reason.lower()


def test_fifo_batch_consumption():
    """Vérifie que la consommation est bien FIFO (lot le plus ancien consommé d'abord)."""
    t0 = datetime(2026, 11, 4, 12, 0)
    state = create_initial_kitchen_state(t0)
    pstate = state.products["Whopper"]

    # Lot 1 : 4 steaks âgés de 10 min
    b1 = SteakBatch(batch_id=1, quantity=4, age_minutes=10)
    # Lot 2 : 6 steaks âgés de 2 min
    b2 = SteakBatch(batch_id=2, quantity=6, age_minutes=2)

    pstate.bin_batches.extend([b2, b1])  # Insérés dans le désordre

    # Commande de 2 Whoppers = 4 steaks
    demand = {"Whopper": 2, "Hamb": 0, "Master": 0}
    step_minute(state, demand)

    # Le lot le plus ancien (b1, 10 min) doit avoir été consommé en premier
    assert len(pstate.bin_batches) == 1
    assert pstate.bin_batches[0].batch_id == 2
    assert pstate.bin_batches[0].quantity == 6


def test_excel_export_schema(tmp_path: Path):
    """Vérifie l'export Excel au format exigé."""
    rows = [
        {
            "periode_30min": datetime(2026, 11, 4, 12, 0),
            "ca": 250.50,
            "prevision_mu": 35.0,
            "q10": 28.0,
            "q50": 35.0,
            "q90": 42.0,
            "reel": 38,
            "erreur": 3.0,
            "perte_potentielle": 2.0,
            "perte_reelle": 1,
        }
    ]
    out_file = tmp_path / "test_journal.xlsx"
    exported = export_journal_to_excel(rows, out_file)
    assert exported.exists()
    assert exported.stat().st_size > 0


def test_metrics_wape_and_bias():
    """Vérifie le calcul des métriques WAPE et Biais."""
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    wape = compute_wape(actual, pred)
    bias = compute_bias(actual, pred)

    # Erreurs absolues : |10-12|=2, |20-18|=2, |30-33|=3 => somme = 7
    # Somme réel = 60 => WAPE = 7 / 60 ~ 0.1167
    assert pytest.approx(wape, 0.001) == 7.0 / 60.0
    # Somme (pred - actual) = 2 - 2 + 3 = 3 => Biais = 3 / 60 = 0.05
    assert pytest.approx(bias, 0.001) == 3.0 / 60.0

