"""Configuration centralisée pour l'agent Braise (Architecture 1).

Source unique de vérité pour tous les paramètres métier :
- Durée de maintien maximale (HOLD)
- Produits finis, recettes et temps de cuisson
- Inventaire matières premières et types de conservation :
  * Viandes : périssable strict, durée limitée hors congélateur, non réintégrable
  * Glaces : conservées au froid de service, réintégrables au congélateur en excès
  * Autres accompagnements
- Tarification et coûts Newsvendor (Cu, Co)
- Seuils de repli (caisse coupée)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ItemCategory(str, Enum):
    MEAT_STRICT = "meat_strict"        # Viande : durée max hors congélation, perte stricte, non réintégrable
    RETURNABLE_COLD = "returnable_cold"  # Glaces : gardées au froid, réintégrables au congélateur en fin de shift
    AMBIENT_DRY = "ambient_dry"        # Pains, emballages : stables à température ambiante
    SIDES_FROZEN = "sides_frozen"      # Frites, onion rings : cuits directement depuis congélateur


@dataclass(frozen=True)
class InventoryItemConfig:
    sku: str
    name: str
    category: ItemCategory
    unit: str
    unit_cost_euros: float
    # Durée maximale en minutes passée hors congélateur avant péremption
    # Pour la viande sortie du congélateur : fenêtre d'utilisation stricte
    max_out_of_freezer_minutes: Optional[int]
    # Possibilité de réintégrer l'excès au congélateur en fin de journée
    can_return_to_freezer: bool
    # Stock initial par défaut en démo
    default_freezer_stock: int
    default_prep_stock: int


@dataclass(frozen=True)
class ProductConfig:
    name: str
    steaks_per_burger: int
    cook_time_minutes: int
    demand_share: float
    price_euros: float
    meat_sku: str  # Lien avec la matière première dans l'inventaire


# Durée de maintien maximale réglementaire en bac chaud (non négociable)
MAX_HOLD_MINUTES: int = 15

# Durée maximale autorisée pour la viande crue sortie du congélateur avant cuisson (en minutes)
# Par exemple 60 minutes de décongélation / attente en zone de préparation
MAX_MEAT_OUT_OF_FREEZER_MINUTES: int = 60

# Coûts unitaires pour la formule Newsvendor critique tau = Cu / (Cu + Co)
DEFAULT_CU: float = 3.5  # Perte de marge / insatisfaction client
DEFAULT_CO: float = 1.2  # Coût matière d'un steak jeté
DEFAULT_TAU: float = DEFAULT_CU / (DEFAULT_CU + DEFAULT_CO)  # ~0.745

# Produits burgers finis
PRODUCTS: Dict[str, ProductConfig] = {
    "Whopper": ProductConfig(
        name="Whopper",
        steaks_per_burger=2,
        cook_time_minutes=6,
        demand_share=0.34,
        price_euros=8.50,
        meat_sku="STEAK_WHOPPER",
    ),
    "Hamb": ProductConfig(
        name="Hamb",
        steaks_per_burger=1,
        cook_time_minutes=4,
        demand_share=0.42,
        price_euros=5.00,
        meat_sku="STEAK_HAMB",
    ),
    "Master": ProductConfig(
        name="Master",
        steaks_per_burger=3,
        cook_time_minutes=8,
        demand_share=0.24,
        price_euros=10.50,
        meat_sku="STEAK_MASTER",
    ),
}

# Inventaire complet des matières premières et produits suivis par le manager
INVENTORY_ITEMS: Dict[str, InventoryItemConfig] = {
    "STEAK_WHOPPER": InventoryItemConfig(
        sku="STEAK_WHOPPER",
        name="Steaks Whopper",
        category=ItemCategory.MEAT_STRICT,
        unit="steaks",
        unit_cost_euros=0.85,
        max_out_of_freezer_minutes=MAX_MEAT_OUT_OF_FREEZER_MINUTES,
        can_return_to_freezer=False,  # Interdiction stricte de recongeler de la viande sortie
        default_freezer_stock=400,
        default_prep_stock=24,
    ),
    "STEAK_HAMB": InventoryItemConfig(
        sku="STEAK_HAMB",
        name="Steaks Hamb",
        category=ItemCategory.MEAT_STRICT,
        unit="steaks",
        unit_cost_euros=0.55,
        max_out_of_freezer_minutes=MAX_MEAT_OUT_OF_FREEZER_MINUTES,
        can_return_to_freezer=False,
        default_freezer_stock=500,
        default_prep_stock=30,
    ),
    "STEAK_MASTER": InventoryItemConfig(
        sku="STEAK_MASTER",
        name="Steaks Master",
        category=ItemCategory.MEAT_STRICT,
        unit="steaks",
        unit_cost_euros=1.10,
        max_out_of_freezer_minutes=MAX_MEAT_OUT_OF_FREEZER_MINUTES,
        can_return_to_freezer=False,
        default_freezer_stock=250,
        default_prep_stock=18,
    ),
    "MIX_GLACE": InventoryItemConfig(
        sku="MIX_GLACE",
        name="Bacs Glace Vanille",
        category=ItemCategory.RETURNABLE_COLD,
        unit="bacs",
        unit_cost_euros=3.20,
        max_out_of_freezer_minutes=None,  # Maintenue au froid de service (machine)
        can_return_to_freezer=True,       # Réintégrable au congélateur en fin de journée si excès
        default_freezer_stock=50,
        default_prep_stock=6,
    ),
    "FRITES_SURGELEES": InventoryItemConfig(
        sku="FRITES_SURGELEES",
        name="Sachets Frites 2.5kg",
        category=ItemCategory.SIDES_FROZEN,
        unit="sachets",
        unit_cost_euros=1.80,
        max_out_of_freezer_minutes=120,
        can_return_to_freezer=False,
        default_freezer_stock=80,
        default_prep_stock=8,
    ),
    "PAINS_BUNS": InventoryItemConfig(
        sku="PAINS_BUNS",
        name="Pains Burgers (Buns)",
        category=ItemCategory.AMBIENT_DRY,
        unit="pains",
        unit_cost_euros=0.25,
        max_out_of_freezer_minutes=None,
        can_return_to_freezer=False,
        default_freezer_stock=0,
        default_prep_stock=350,
    ),
}

# Règle fixe de secours (quand la caisse est coupée / hors-ligne)
FALLBACK_TRIGGER_STOCK: int = 6
FALLBACK_COOK_QTY: int = 6

# Paramètres de décision Newsvendor & seuils
MIN_COOK_BATCH: int = 3       # Quantité minimale recommandée par lot
LOOKAHEAD_MARGIN_MIN: int = 8  # Fenêtre = durée_cuisson + 8 minutes
TREND_WINDOW_MINUTES: int = 15 # Fenêtre d'observation de la tendance récente
TREND_K_MIN: float = 0.8
TREND_K_MAX: float = 1.8
