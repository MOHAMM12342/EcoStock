"""Tests unitaires pour la gestion d'inventaire en temps réel et les règles produits (Braise).

Valide :
1. Suivi d'inventaire en temps réel (chambre froide vs préparation)
2. Règle stricte sanitaire pour la viande (non réintégrable au congélateur)
3. Règle pour les glaces (réintégrables au congélateur en fin de journée si excès)
4. Péremption automatique des matières premières après la durée max hors congélateur
5. Ajustement manuel de l'inventaire en temps réel par le manager
6. Calcul du modèle pour les pertes potentielles en fonction du temps passé hors froid
"""

from datetime import datetime
import pytest

from src.config import MAX_MEAT_OUT_OF_FREEZER_MINUTES
from src.simulator.inventory import InventoryManager


def test_checkout_from_freezer():
    """Vérifie la sortie de stock du congélateur vers la zone de préparation."""
    inv = InventoryManager()
    init_freezer = inv.items["STEAK_WHOPPER"].freezer_stock

    batch = inv.checkout_from_freezer("STEAK_WHOPPER", quantity=40)
    assert batch.quantity == 40
    assert batch.time_out_of_freezer_min == 0
    assert inv.items["STEAK_WHOPPER"].freezer_stock == init_freezer - 40
    assert any(b.batch_id == batch.batch_id for b in inv.items["STEAK_WHOPPER"].prep_batches)


def test_ice_cream_can_be_returned_to_freezer():
    """Vérifie que les glaces (RETURNABLE_COLD) peuvent être réintégrées au congélateur en cas d'excès."""
    inv = InventoryManager()
    glace_state = inv.items["MIX_GLACE"]
    init_freezer = glace_state.freezer_stock
    init_prep = glace_state.prep_stock_total

    # Fin de journée : réintégration de 2 bacs de glace excédentaires
    success, msg = inv.return_to_freezer("MIX_GLACE", quantity=2)
    assert success is True
    assert "réintégrés au congélateur avec succès" in msg.lower()
    assert glace_state.freezer_stock == init_freezer + 2
    assert glace_state.prep_stock_total == init_prep - 2
    assert glace_state.returned_to_freezer_count == 2


def test_meat_cannot_be_returned_to_freezer_sanitary_rule():
    """Règle sanitaire stricte : la viande sortie du congélateur ne PEUT PAS être réintégrée."""
    inv = InventoryManager()
    meat_state = inv.items["STEAK_WHOPPER"]
    init_freezer = meat_state.freezer_stock

    success, msg = inv.return_to_freezer("STEAK_WHOPPER", quantity=5)
    assert success is False
    assert "sécurité sanitaire" in msg.lower() or "interdite" in msg.lower()
    assert meat_state.freezer_stock == init_freezer  # Aucun changement


def test_meat_expires_after_max_time_out_of_freezer():
    """Vérifie que la viande sortie du congélateur périme si non utilisée après max_out_of_freezer_minutes."""
    inv = InventoryManager()
    # Vide les lots par défaut pour isoler le test
    inv.items["STEAK_MASTER"].prep_batches.clear()

    # Sortie d'un lot de 20 steaks Master
    batch = inv.checkout_from_freezer("STEAK_MASTER", quantity=20)
    batch.time_out_of_freezer_min = MAX_MEAT_OUT_OF_FREEZER_MINUTES - 1

    # À t + 1 min, le lot atteint le seuil maximal (ex: 60 min)
    wasted = inv.step_minute()
    assert wasted["STEAK_MASTER"] == 20
    assert inv.items["STEAK_MASTER"].wasted_count == 20
    assert len(inv.items["STEAK_MASTER"].prep_batches) == 0


def test_manager_inventory_realtime_adjustment():
    """Le manager ajuste l'inventaire physique en temps réel (comptage de stock)."""
    inv = InventoryManager()
    inv.adjust_stock("STEAK_HAMB", freezer_stock=350, prep_stock=15)

    item = inv.items["STEAK_HAMB"]
    assert item.freezer_stock == 350
    assert item.prep_stock_total == 15
    assert len(item.prep_batches) == 1
    assert item.prep_batches[0].time_out_of_freezer_min == 0


def test_potential_loss_calculation():
    """Le modèle calcule la perte potentielle d'après le temps passé hors congélation et la prévision."""
    inv = InventoryManager()
    inv.items["STEAK_WHOPPER"].prep_batches.clear()

    # Sortie de 30 steaks Whopper, déjà sortis depuis 50 min (il reste 10 min de vie avant 60 min)
    batch = inv.checkout_from_freezer("STEAK_WHOPPER", quantity=30)
    batch.time_out_of_freezer_min = 50

    # Demande prévue très faible : 0.5 steak par minute => 5 steaks attendus sur les 10 min restantes
    forecast_rates = {"STEAK_WHOPPER": 0.5}
    losses = inv.calculate_potential_losses(forecast_rates)

    whopper_loss = losses["STEAK_WHOPPER"]
    # Perte attendue : 30 - 5 = 25 steaks
    assert whopper_loss["potential_loss_qty"] == 25
    assert whopper_loss["potential_loss_euros"] > 0
    assert whopper_loss["earliest_expiry_min"] == 10
    assert whopper_loss["status"] in ("RISQUE PERTE", "ALERTE EXPIRATION")

