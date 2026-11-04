"""Gestionnaire d'inventaire matières premières et produits en temps réel (Braise).

Permet au manager et au shift leader :
1. De suivre l'inventaire complet en temps réel au cours de la journée (Chambre froide / Préparation).
2. De sortir des matières du congélateur selon les besoins.
3. D'appliquer des règles différenciées :
   - Viandes (MEAT_STRICT) : durée maximale stricte hors congélation, non réintégrables.
   - Glaces (RETURNABLE_COLD) : conservées au froid, réintégrables au congélateur en cas d'excès en fin de journée.
4. D'ajuster manuellement les stocks réels (inventaire tournant / comptage manager).
5. De calculer les pertes potentielles en fonction du temps passé hors congélation et de la demande prévue.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math

from src.config import (
    INVENTORY_ITEMS,
    InventoryItemConfig,
    ItemCategory,
)


@dataclass
class InventoryBatch:
    batch_id: int
    sku: str
    quantity: int
    time_out_of_freezer_min: int = 0
    checked_out_at: Optional[datetime] = None


@dataclass
class InventoryItemState:
    sku: str
    freezer_stock: int
    prep_batches: List[InventoryBatch] = field(default_factory=list)
    returned_to_freezer_count: int = 0
    wasted_count: int = 0
    total_checked_out: int = 0

    @property
    def prep_stock_total(self) -> int:
        """Total des unités sorties du congélateur actuellement en zone de préparation."""
        return sum(b.quantity for b in self.prep_batches)

    @property
    def total_stock(self) -> int:
        """Total du stock en magasin (congélateur + préparation)."""
        return self.freezer_stock + self.prep_stock_total


class InventoryManager:
    """Gestionnaire central de l'inventaire en temps réel."""

    def __init__(self):
        self.items: Dict[str, InventoryItemState] = {}
        self._next_batch_id: int = 1
        self._init_defaults()

    def _init_defaults(self):
        """Initialise l'inventaire à partir de la configuration."""
        for sku, cfg in INVENTORY_ITEMS.items():
            state = InventoryItemState(
                sku=sku,
                freezer_stock=cfg.default_freezer_stock,
            )
            if cfg.default_prep_stock > 0:
                batch = InventoryBatch(
                    batch_id=self._get_next_batch_id(),
                    sku=sku,
                    quantity=cfg.default_prep_stock,
                    time_out_of_freezer_min=5,  # légèrement entamé en début de service
                )
                state.prep_batches.append(batch)
                state.total_checked_out += cfg.default_prep_stock
            self.items[sku] = state

    def _get_next_batch_id(self) -> int:
        bid = self._next_batch_id
        self._next_batch_id += 1
        return bid

    def checkout_from_freezer(
        self,
        sku: str,
        quantity: int,
        current_time: Optional[datetime] = None,
    ) -> InventoryBatch:
        """Sortie de matières premières du congélateur vers la zone de préparation."""
        if sku not in self.items:
            raise ValueError(f"SKU inconnu : {sku}")
        if quantity <= 0:
            raise ValueError("La quantité sortie doit être strictement positive.")

        item_state = self.items[sku]
        # Décrémentation du stock congélateur (avec tolérance négative bornée à 0 si besoin de réappro)
        actual_take = min(item_state.freezer_stock, quantity)
        item_state.freezer_stock = max(0, item_state.freezer_stock - quantity)

        batch = InventoryBatch(
            batch_id=self._get_next_batch_id(),
            sku=sku,
            quantity=quantity,
            time_out_of_freezer_min=0,
            checked_out_at=current_time,
        )
        item_state.prep_batches.append(batch)
        item_state.total_checked_out += quantity
        return batch

    def return_to_freezer(
        self,
        sku: str,
        quantity: int,
    ) -> Tuple[bool, str]:
        """Réintégration d'un excès au congélateur en fin de journée / de service.

        Règles :
        - Glaces (RETURNABLE_COLD) : AUTORISÉE
        - Viandes (MEAT_STRICT) : INTERDITE pour des raisons strictes de sécurité sanitaire (chaîne du froid).
        """
        if sku not in self.items:
            return False, f"SKU inconnu : {sku}"
        cfg = INVENTORY_ITEMS[sku]
        if not cfg.can_return_to_freezer:
            return False, (
                f"Action interdite pour '{cfg.name}' : pour des raisons de sécurité sanitaire, "
                "la viande sortie du froid négatif ne peut en aucun cas être réintégrée au congélateur."
            )

        item_state = self.items[sku]
        available_prep = item_state.prep_stock_total
        if available_prep <= 0:
            return False, f"Aucun stock en préparation disponible pour '{cfg.name}'."

        qty_to_return = min(available_prep, quantity)

        # Décrémenter depuis les lots en préparation
        remaining_to_deduct = qty_to_return
        for b in list(item_state.prep_batches):
            if remaining_to_deduct <= 0:
                break
            deduct = min(b.quantity, remaining_to_deduct)
            b.quantity -= deduct
            remaining_to_deduct -= deduct
            if b.quantity <= 0:
                item_state.prep_batches.remove(b)

        item_state.freezer_stock += qty_to_return
        item_state.returned_to_freezer_count += qty_to_return
        return True, f"{qty_to_return} {cfg.unit} de '{cfg.name}' réintégrés au congélateur avec succès."

    def adjust_stock(
        self,
        sku: str,
        freezer_stock: Optional[int] = None,
        prep_stock: Optional[int] = None,
    ) -> None:
        """Ajustement direct de l'inventaire physique par le manager."""
        if sku not in self.items:
            raise ValueError(f"SKU inconnu : {sku}")
        item_state = self.items[sku]

        if freezer_stock is not None:
            item_state.freezer_stock = max(0, freezer_stock)

        if prep_stock is not None:
            # Réajuste le stock en préparation sous forme d'un lot frais
            item_state.prep_batches.clear()
            if prep_stock > 0:
                item_state.prep_batches.append(
                    InventoryBatch(
                        batch_id=self._get_next_batch_id(),
                        sku=sku,
                        quantity=prep_stock,
                        time_out_of_freezer_min=0,
                    )
                )

    def consume_for_cooking(self, sku: str, quantity: int) -> int:
        """Consomme des steaks/produits en préparation pour lancer une cuisson (FIFO)."""
        if sku not in self.items:
            return 0
        item_state = self.items[sku]

        # Consommation FIFO : les lots sortis depuis le plus longtemps sont utilisés d'abord
        item_state.prep_batches.sort(key=lambda b: b.time_out_of_freezer_min, reverse=True)

        needed = quantity
        consumed = 0
        surviving: List[InventoryBatch] = []

        for b in item_state.prep_batches:
            if needed > 0:
                take = min(b.quantity, needed)
                b.quantity -= take
                needed -= take
                consumed += take
            if b.quantity > 0:
                surviving.append(b)

        item_state.prep_batches = surviving

        # Si le stock en préparation était insuffisant, sortie automatique du congélateur
        if needed > 0:
            extra_take = min(item_state.freezer_stock, needed)
            item_state.freezer_stock = max(0, item_state.freezer_stock - needed)
            consumed += needed

        return consumed

    def step_minute(self) -> Dict[str, int]:
        """Vieillit tous les lots en préparation et élimine les lots expirés hors congélateur."""
        wasted_by_sku: Dict[str, int] = {sku: 0 for sku in self.items}

        for sku, state in self.items.items():
            cfg = INVENTORY_ITEMS[sku]
            max_out = cfg.max_out_of_freezer_minutes

            surviving_batches: List[InventoryBatch] = []
            for batch in state.prep_batches:
                batch.time_out_of_freezer_min += 1

                # Vérification de péremption si l'article a une durée max hors congélateur
                if max_out is not None and batch.time_out_of_freezer_min >= max_out:
                    wasted_by_sku[sku] += batch.quantity
                    state.wasted_count += batch.quantity
                else:
                    surviving_batches.append(batch)

            state.prep_batches = surviving_batches

        return wasted_by_sku

    def calculate_potential_losses(
        self,
        forecast_demand_per_min: Dict[str, float],
    ) -> Dict[str, Dict]:
        """Calcule les pertes potentielles en fonction du temps passé hors congélation.

        Pour chaque lot sorti du congélateur :
        - Temps restant = max_out_of_freezer_minutes - time_out_of_freezer_min
        - Demande prévue sur cette fenêtre = temps restant * forecast_demand_per_min
        - Perte potentielle = max(0, quantite - demande prévue)
        """
        results: Dict[str, Dict] = {}

        for sku, state in self.items.items():
            cfg = INVENTORY_ITEMS[sku]
            max_out = cfg.max_out_of_freezer_minutes

            if max_out is None or not state.prep_batches:
                results[sku] = {
                    "sku": sku,
                    "name": cfg.name,
                    "potential_loss_qty": 0,
                    "potential_loss_euros": 0.0,
                    "earliest_expiry_min": None,
                    "status": "OK",
                }
                continue

            # Demande attendue par minute pour ce produit
            demand_rate = forecast_demand_per_min.get(sku, 0.5)

            tot_loss = 0.0
            earliest_expiry = 9999

            for b in state.prep_batches:
                time_left = max(0, max_out - b.time_out_of_freezer_min)
                earliest_expiry = min(earliest_expiry, time_left)

                expected_consumption = demand_rate * time_left
                if b.quantity > expected_consumption:
                    tot_loss += (b.quantity - expected_consumption)

            tot_loss_int = int(math.ceil(tot_loss))
            loss_euros = round(tot_loss_int * cfg.unit_cost_euros, 2)

            status = "OK"
            if tot_loss_int > 0:
                status = "RISQUE PERTE"
            elif earliest_expiry <= 10:
                status = "ALERTE EXPIRATION"

            results[sku] = {
                "sku": sku,
                "name": cfg.name,
                "potential_loss_qty": tot_loss_int,
                "potential_loss_euros": loss_euros,
                "earliest_expiry_min": earliest_expiry if earliest_expiry < 9999 else None,
                "status": status,
            }

        return results

    def get_summary_report(self) -> List[Dict]:
        """Génère le rapport d'inventaire complet pour le tableau de bord manager."""
        report: List[Dict] = []
        for sku, state in self.items.items():
            cfg = INVENTORY_ITEMS[sku]
            batches_info = [
                {
                    "batch_id": b.batch_id,
                    "quantity": b.quantity,
                    "time_out_min": b.time_out_of_freezer_min,
                    "time_left_min": (
                        max(0, cfg.max_out_of_freezer_minutes - b.time_out_of_freezer_min)
                        if cfg.max_out_of_freezer_minutes
                        else None
                    ),
                }
                for b in state.prep_batches
            ]
            report.append({
                "sku": sku,
                "name": cfg.name,
                "category": cfg.category.value,
                "unit": cfg.unit,
                "freezer_stock": state.freezer_stock,
                "prep_stock": state.prep_stock_total,
                "total_stock": state.total_stock,
                "wasted_count": state.wasted_count,
                "returned_to_freezer_count": state.returned_to_freezer_count,
                "can_return_to_freezer": cfg.can_return_to_freezer,
                "max_out_minutes": cfg.max_out_of_freezer_minutes,
                "prep_batches": batches_info,
            })
        return report

