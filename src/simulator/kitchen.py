"""Simulateur de cuisine et bac de maintien (Braise).

Gère l'état à la minute d'un restaurant :
- Bacs de maintien par produit avec règle FIFO stricte
- Péremption automatique des lots atteignant MAX_HOLD_MINUTES (15 min)
- Lots en cuisson sur le grill
- Consommation par les commandes POS
- Journalisation des steaks cuits, utilisés, jetés et burgers en attente
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.config import MAX_HOLD_MINUTES, PRODUCTS, ProductConfig
from src.simulator.inventory import InventoryManager


@dataclass
class SteakBatch:
    batch_id: int
    quantity: int
    age_minutes: int = 0
    cook_time_left: int = 0


@dataclass
class KitchenProductState:
    product_name: str
    bin_batches: List[SteakBatch] = field(default_factory=list)
    cooking_batches: List[SteakBatch] = field(default_factory=list)
    recent_actual_steaks: List[int] = field(default_factory=lambda: [0] * 15)
    recent_forecast_steaks: List[float] = field(default_factory=lambda: [0.0] * 15)
    snooze_until: Optional[datetime] = None

    @property
    def ready_steaks_total(self) -> int:
        """Nombre total de steaks actuellement prêts dans le bac."""
        return sum(b.quantity for b in self.bin_batches)

    @property
    def cooking_steaks_total(self) -> int:
        """Nombre total de steaks actuellement en cours de cuisson."""
        return sum(b.quantity for b in self.cooking_batches)

    @property
    def usable_steaks_count(self) -> int:
        """Steaks ayant au moins 2 minutes de durée de vie restante."""
        return sum(
            b.quantity
            for b in self.bin_batches
            if (MAX_HOLD_MINUTES - b.age_minutes) >= 2
        )


@dataclass
class KitchenState:
    current_time: datetime
    products: Dict[str, KitchenProductState] = field(default_factory=dict)
    inventory: InventoryManager = field(default_factory=InventoryManager)
    is_pos_connected: bool = True
    total_unmet_burgers: float = 0.0
    total_cooked_steaks: int = 0
    total_wasted_steaks: int = 0
    total_used_steaks: int = 0
    total_revenue_euros: float = 0.0
    _next_batch_id: int = 1

    def next_batch_id(self) -> int:
        bid = self._next_batch_id
        self._next_batch_id += 1
        return bid


@dataclass
class Action:
    product: str
    quantity: int
    reason: str
    urgency_minutes: int = 2


@dataclass
class MinuteTickResult:
    timestamp: datetime
    orders_placed: Dict[str, int]
    steaks_consumed: Dict[str, int]
    steaks_wasted: Dict[str, int]
    steaks_finished_cooking: Dict[str, int]
    unmet_burgers: Dict[str, float]
    revenue_generated: float
    raw_materials_wasted: Dict[str, int] = field(default_factory=dict)


def create_initial_kitchen_state(start_time: datetime, pos_connected: bool = True) -> KitchenState:
    """Crée l'état initial de la cuisine avec les produits configurés."""
    state = KitchenState(current_time=start_time, is_pos_connected=pos_connected)
    for name in PRODUCTS:
        state.products[name] = KitchenProductState(product_name=name)
    return state


def step_minute(
    state: KitchenState,
    burger_demand: Dict[str, int],
    forecast_steaks: Optional[Dict[str, float]] = None,
) -> MinuteTickResult:
    """Fait avancer la simulation d'une minute entière.

    Ordre des opérations :
    1. Vieillissement des lots dans le bac (+1 minute).
    2. Avancement des cuissons (-1 minute) et transfert dans le bac pour les lots terminés.
    3. Traitement des commandes reçues (consommation FIFO).
    4. Péremption et élimination des lots ayant atteint MAX_HOLD_MINUTES.
    5. Mise à jour de l'historique glissant (15 dernières minutes).
    """
    current_dt = state.current_time
    tick_consumed: Dict[str, int] = {p: 0 for p in state.products}
    tick_wasted: Dict[str, int] = {p: 0 for p in state.products}
    tick_finished: Dict[str, int] = {p: 0 for p in state.products}
    tick_unmet: Dict[str, float] = {p: 0.0 for p in state.products}
    tick_revenue = 0.0

    # 0. Vieillissement de l'inventaire en préparation et péremption hors congélateur
    raw_wasted = state.inventory.step_minute()

    for prod_name, pstate in state.products.items():
        pcfg = PRODUCTS[prod_name]

        # 1. Vieillir les lots existants dans le bac
        for batch in pstate.bin_batches:
            batch.age_minutes += 1

        # 2. Avancer les cuissons
        still_cooking: List[SteakBatch] = []
        for batch in pstate.cooking_batches:
            batch.cook_time_left -= 1
            if batch.cook_time_left <= 0:
                # Lot cuit, transféré dans le bac avec un âge de 0 min
                batch.age_minutes = 0
                pstate.bin_batches.append(batch)
                tick_finished[prod_name] += batch.quantity
            else:
                still_cooking.append(batch)
        pstate.cooking_batches = still_cooking

        # 3. Consommation des steaks par les commandes (FIFO)
        demand_burgers = burger_demand.get(prod_name, 0)
        steaks_needed = demand_burgers * pcfg.steaks_per_burger
        remaining_needed = steaks_needed

        # Les lots les plus anciens sont en tête ou triés par âge descendant
        pstate.bin_batches.sort(key=lambda b: b.age_minutes, reverse=True)

        surviving_batches: List[SteakBatch] = []
        for batch in pstate.bin_batches:
            if remaining_needed > 0:
                take = min(batch.quantity, remaining_needed)
                batch.quantity -= take
                remaining_needed -= take
                tick_consumed[prod_name] += take

            if batch.quantity > 0:
                surviving_batches.append(batch)

        pstate.bin_batches = surviving_batches

        # Calcul des ruptures éventuelles
        if remaining_needed > 0:
            unmet_b = remaining_needed / pcfg.steaks_per_burger
            tick_unmet[prod_name] = unmet_b
            state.total_unmet_burgers += unmet_b

        served_burgers = (steaks_needed - remaining_needed) / pcfg.steaks_per_burger
        rev = served_burgers * pcfg.price_euros
        tick_revenue += rev

        # 4. Péremption : tout lot dont l'âge atteint MAX_HOLD_MINUTES est jeté
        good_batches: List[SteakBatch] = []
        for batch in pstate.bin_batches:
            if batch.age_minutes >= MAX_HOLD_MINUTES:
                tick_wasted[prod_name] += batch.quantity
                state.total_wasted_steaks += batch.quantity
            else:
                good_batches.append(batch)
        pstate.bin_batches = good_batches

        # 5. Mise à jour de l'historique récent (15 minutes)
        act_val = tick_consumed[prod_name]
        fc_val = forecast_steaks.get(prod_name, float(act_val)) if forecast_steaks else float(act_val)

        pstate.recent_actual_steaks.append(act_val)
        pstate.recent_actual_steaks.pop(0)
        pstate.recent_forecast_steaks.append(fc_val)
        pstate.recent_forecast_steaks.pop(0)

        state.total_used_steaks += tick_consumed[prod_name]

    state.total_revenue_euros += tick_revenue
    state.current_time += timedelta(minutes=1)

    return MinuteTickResult(
        timestamp=current_dt,
        orders_placed=burger_demand,
        steaks_consumed=tick_consumed,
        steaks_wasted=tick_wasted,
        steaks_finished_cooking=tick_finished,
        unmet_burgers=tick_unmet,
        revenue_generated=tick_revenue,
        raw_materials_wasted=raw_wasted,
    )


def execute_action(state: KitchenState, action: Action) -> None:
    """Exécute une recommandation de cuisson en créant un lot sur le grill et en consommant l'inventaire."""
    pstate = state.products[action.product]
    pcfg = PRODUCTS[action.product]

    # Consommation des matières premières en préparation
    state.inventory.consume_for_cooking(pcfg.meat_sku, action.quantity)

    batch = SteakBatch(
        batch_id=state.next_batch_id(),
        quantity=action.quantity,
        age_minutes=0,
        cook_time_left=pcfg.cook_time_minutes,
    )
    pstate.cooking_batches.append(batch)
    state.total_cooked_steaks += action.quantity

