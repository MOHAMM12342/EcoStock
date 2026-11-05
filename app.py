"""Application serveur web FastAPI (Braise - Architecture 1).

Fournit :
1. L'hébergement de l'interface graphique HTML `braise.html` sur la racine `/`
2. Les points d'entrée d'API pour exécuter la simulation d'une journée
3. Le téléchargement du journal Excel généré
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import numpy as np
import pandas as pd
import uvicorn

from pydantic import BaseModel
from src.config import PRODUCTS
from src.data.generate import generate_30min_dataset
from src.data.pipeline import get_daily_test_data, get_test_dates, get_train_data
from src.export.excel import export_journal_to_excel
from src.forecast.model import DEFAULT_MODEL_PATH, DemandForecaster, set_global_forecaster
from src.policy.newsvendor import decide_fixed_policy, decide_newsvendor
from src.simulator.kitchen import SteakBatch, create_initial_kitchen_state
from src.simulator.inventory import InventoryManager
from src.simulator.runner import simulate_day

app = FastAPI(
    title="Braise API — Architecture 1",
    description="Moteur de prévision, inventaire temps réel et simulateur anti-gaspillage",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "braise.html"
EXPORTS_DIR = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_FILE = EXPORTS_DIR / "journal_braise_dernier.xlsx"

# Instance d'inventaire en direct pour le manager et le chef d'équipe
LIVE_INVENTORY = InventoryManager()

# Modèles Pydantic pour les requêtes d'inventaire
class CheckoutRequest(BaseModel):
    sku: str
    quantity: int

class ReturnRequest(BaseModel):
    sku: str
    quantity: int

class AdjustRequest(BaseModel):
    sku: str
    freezer_stock: Optional[int] = None
    prep_stock: Optional[int] = None

class DailyUpdateRequest(BaseModel):
    days_count: Optional[int] = 1
    n_additional_trees: Optional[int] = 15
    learning_rate: Optional[float] = 0.05
    date_str: Optional[str] = None

class LiveTranche(BaseModel):
    k: int
    time_label: Optional[str] = None
    ca: float = 0.0
    used: int = 0
    cook: int = 0
    waste: int = 0
    pred: float = 0.0
    act: int = 0

class LiveExportRequest(BaseModel):
    date: Optional[str] = None
    session_minutes: Optional[int] = None
    periods: List[LiveTranche] = []

# Initialisation lazy du modèle avec persistance
_FORECASTER = None


def get_or_train_forecaster():
    """Charge le modèle LightGBM pré-entraîné ou l'entraîne sur le split Train réel (18 mois)."""
    global _FORECASTER
    if _FORECASTER is None:
        if DEFAULT_MODEL_PATH.exists():
            try:
                _FORECASTER = DemandForecaster.load(DEFAULT_MODEL_PATH)
                set_global_forecaster(_FORECASTER)
                return _FORECASTER
            except Exception:
                pass
        # Entraînement initial sur les données réelles (14 mois / 35 616 tranches)
        df_train = get_train_data()
        _FORECASTER = DemandForecaster().fit(df_train)
        _FORECASTER.save(DEFAULT_MODEL_PATH)
        set_global_forecaster(_FORECASTER)
    return _FORECASTER


def compute_realtime_forecast_rates(forecaster: DemandForecaster, dt: Optional[datetime] = None) -> Dict[str, float]:
    """Calcule dynamiquement les taux de consommation par minute à partir du modèle LightGBM."""
    if dt is None:
        dt = datetime.now()

    fixed_holidays = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
    dow = int(dt.weekday())
    month = int(dt.month)
    is_we = 1 if dow in (5, 6) else 0
    is_hol = 1 if (dt.month, dt.day) in fixed_holidays else 0
    next_day = dt + timedelta(days=1)
    is_hol_eve = 1 if (next_day.month, next_day.day) in fixed_holidays else 0
    is_student = 1 if (dow in (3, 4) and dt.hour >= 19) else 0
    temp = 12.0 + 10.0 * np.sin(2 * np.pi * (month - 4) / 12) + (dt.hour - 10) * 0.5

    rows = []
    for prod in ["Whopper", "Hamb", "Master"]:
        rows.append({
            "produit": prod,
            "hour": int(dt.hour),
            "minute": int((dt.minute // 30) * 30),
            "day_of_week": dow,
            "month": month,
            "is_weekend": is_we,
            "is_holiday": is_hol,
            "is_holiday_eve": is_hol_eve,
            "is_student_night": is_student,
            "temperature": round(float(temp), 1),
            "rain_mm": 0.0,
        })

    feat_df = pd.DataFrame(rows)
    preds = forecaster.predict(feat_df)
    mu_map = {row["product"]: float(row["mu"]) for _, row in preds.iterrows()}

    # Taux par minute = mu / 30 min
    rates = {
        "STEAK_WHOPPER": round(mu_map.get("Whopper", 15.0) / 30.0, 3),
        "STEAK_HAMB": round(mu_map.get("Hamb", 15.0) / 30.0, 3),
        "STEAK_MASTER": round(mu_map.get("Master", 3.0) / 30.0, 3),
        "FRITES_SURGELEES": round((mu_map.get("Whopper", 15.0) + mu_map.get("Hamb", 15.0)) * 0.05 / 30.0, 3),
        "PAINS_BUNS": round((mu_map.get("Whopper", 15.0) / 2.0 + mu_map.get("Hamb", 15.0) + mu_map.get("Master", 3.0) / 3.0) / 30.0, 3),
    }
    return rates


@app.get("/", response_class=HTMLResponse)
@app.get("/braise.html", response_class=HTMLResponse)
def serve_ui():
    """Sert l'interface utilisateur Braise."""
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="Interface braise.html non trouvée.")
    return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "Braise", "architecture": 1}


@app.get("/api/forecast")
def get_realtime_forecast():
    """Prévisions en temps réel issues du modèle de régression LightGBM Poisson."""
    from datetime import datetime
    import pandas as pd

    forecaster = get_or_train_forecaster()
    now = datetime.now()

    products = ["Whopper", "Hamb", "Master"]
    rows = []
    for prod in products:
        rows.append({
            "produit": prod,
            "hour": now.hour,
            "minute": (now.minute // 30) * 30,
            "day_of_week": now.weekday(),
            "month": now.month,
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "is_holiday": 0,
            "is_holiday_eve": 0,
            "is_student_night": 1 if (now.weekday() in (3, 4) and now.hour >= 19) else 0,
            "temperature": 19.5,
            "rain_mm": 0.0,
        })

    features_df = pd.DataFrame(rows)
    predictions = forecaster.predict(features_df)

    predictions_by_product = {}
    for _, row in predictions.iterrows():
        predictions_by_product[row["product"]] = {
            "mu": row["mu"],
            "q10": row["q10"],
            "q50": row["q50"],
            "q90": row["q90"],
            "baseline_mu": row["baseline_mu"],
        }

    total_mu = sum(p["mu"] for p in predictions_by_product.values())

    return {
        "model_type": "LightGBM Poisson Regression",
        "inference_engine": "LightGBM",
        "timestamp": now.isoformat(),
        "time_str": now.strftime("%H:%M"),
        "products": products,
        "predictions_by_product": predictions_by_product,
        "predictions_by_horizon": {
            prod: [predictions_by_product[prod]["mu"]] * 5
            for prod in products
        },
        "total_steaks_predicted_15min": round(total_mu * 5, 1),
        "peak_recommendation": {
            prod: predictions_by_product[prod]["q90"]
            for prod in products
        },
    }


@app.get("/api/forecast/model-info")
def get_forecast_model_info():
    """Métadonnées du modèle de régression LightGBM Poisson."""
    forecaster = get_or_train_forecaster()
    n_models = len(forecaster.models)
    return {
        "model_type": "LightGBM Poisson Regression",
        "class_name": "DemandForecaster",
        "n_estimators": 150,
        "num_leaves": 63,
        "num_product_models": n_models,
        "products": list(forecaster.models.keys()),
        "feature_columns": [
            "hour", "minute", "day_of_week", "month", "is_weekend",
            "is_holiday", "is_holiday_eve", "is_student_night",
            "temperature", "rain_mm", "baseline_mu",
        ],
        "quantiles": ["q10", "q50", "q90"],
        "inference_engine": "LightGBM",
        "status": "ready",
        "saved_model_path": str(DEFAULT_MODEL_PATH),
        "days_trained": getattr(forecaster, "days_trained", 0),
        "last_trained_at": getattr(forecaster, "last_trained_at", None),
    }


@app.post("/api/forecast/daily-update")
def train_daily_update(payload: Optional[DailyUpdateRequest] = None):
    """Entraînement continu de fin de journée sur les nouvelles données journalières issues du test set ou de la caisse."""
    days = payload.days_count if payload and payload.days_count else 1
    n_trees = payload.n_additional_trees if payload and payload.n_additional_trees else 15
    lr = payload.learning_rate if payload and payload.learning_rate else 0.05
    date_str = payload.date_str if payload else None

    forecaster = get_or_train_forecaster()
    test_dates = get_test_dates()
    if test_dates:
        if date_str and date_str in test_dates:
            daily_df = get_daily_test_data(date_str)
        else:
            target_idx = min(len(test_dates) - 1, getattr(forecaster, "days_trained", 0))
            daily_df = get_daily_test_data(test_dates[target_idx])
    else:
        daily_df = generate_30min_dataset(days_count=days, seed=None)

    forecaster.update_daily(daily_df, n_additional_trees=n_trees, learning_rate=lr, save_after=True)
    sample_day = daily_df["date"].iloc[0] if "date" in daily_df.columns else "inconnue"

    return {
        "status": "success",
        "message": f"Entraînement continu terminé sur la journée {sample_day}. Poids sauvegardés dans {DEFAULT_MODEL_PATH}.",
        "trees_per_product": {
            p: m.booster_.num_trees()
            for p, m in forecaster.models.items()
            if hasattr(m, "booster_")
        },
        "days_trained": forecaster.days_trained,
        "last_trained_at": forecaster.last_trained_at,
        "sample_day": sample_day,
    }


@app.get("/api/dataset/info")
def get_dataset_info():
    """Informations sur le dataset extrait (18 mois / 944 451 tickets réels) et le split Train/Test."""
    test_dates = get_test_dates()
    train_df = get_train_data()
    return {
        "dataset_name": "tickets_et_compositions_18_mois.zip",
        "total_tickets": 944451,
        "total_days": 546,
        "train_period": {
            "start": "2025-01-01",
            "end": "2026-02-28",
            "days_count": train_df["date"].nunique(),
            "slots_30min": len(train_df),
        },
        "test_period": {
            "start": test_dates[0] if test_dates else "2026-03-01",
            "end": test_dates[-1] if test_dates else "2026-06-30",
            "days_count": len(test_dates),
            "slots_30min": len(test_dates) * 28 * 3,
        },
        "products": ["Whopper", "Hamb", "Master"],
        "status": "extracted_and_ready",
    }


@app.get("/api/kitchen/recommendation")
def get_kitchen_recommendation(
    bins_whopper: int = 6,
    bins_hamb: int = 6,
    bins_master: int = 6,
    cooking_whopper: int = 0,
    cooking_hamb: int = 0,
    cooking_master: int = 0,
    pos_connected: bool = True,
):
    """Fournit en direct les inputs décisionnels pour l'écran Cuisine (carte Grill et alertes)."""
    now = datetime.now()
    forecaster = get_or_train_forecaster()

    # Inférence du modèle LightGBM pour l'instant t
    fixed_holidays = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
    dow = int(now.weekday())
    month = int(now.month)
    next_day = now + timedelta(days=1)
    feat_rows = []
    for prod_name in ["Whopper", "Hamb", "Master"]:
        feat_rows.append({
            "produit": prod_name,
            "hour": int(now.hour),
            "minute": int((now.minute // 30) * 30),
            "day_of_week": dow,
            "month": month,
            "is_weekend": 1 if dow in (5, 6) else 0,
            "is_holiday": 1 if (now.month, now.day) in fixed_holidays else 0,
            "is_holiday_eve": 1 if (next_day.month, next_day.day) in fixed_holidays else 0,
            "is_student_night": 1 if (dow in (3, 4) and now.hour >= 19) else 0,
            "temperature": 19.5,
            "rain_mm": 0.0,
        })
    preds = forecaster.predict(pd.DataFrame(feat_rows))

    # État physique de la cuisine
    state = create_initial_kitchen_state(now, pos_connected=pos_connected)
    state.products["Whopper"].bin_batches = [SteakBatch(1, bins_whopper, 5)]
    state.products["Whopper"].cooking_batches = [SteakBatch(2, cooking_whopper, 0)] if cooking_whopper > 0 else []
    state.products["Hamb"].bin_batches = [SteakBatch(3, bins_hamb, 5)]
    state.products["Hamb"].cooking_batches = [SteakBatch(4, cooking_hamb, 0)] if cooking_hamb > 0 else []
    state.products["Master"].bin_batches = [SteakBatch(5, bins_master, 5)]
    state.products["Master"].cooking_batches = [SteakBatch(6, cooking_master, 0)] if cooking_master > 0 else []

    actions = decide_newsvendor(state, preds, now)
    if actions:
        act = actions[0]
        cfg = PRODUCTS[act.product]
        ready_dt = now + timedelta(minutes=cfg.cook_time_minutes)
        return {
            "has_recommendation": True,
            "product": act.product,
            "quantity": act.quantity,
            "cook_time_minutes": cfg.cook_time_minutes,
            "ready_at": ready_dt.strftime("%H:%M"),
            "reason": act.reason,
            "status_badge": "À LANCER",
            "dot_class": "k-dot warn",
            "model_forecast": {row["product"]: float(row["mu"]) for _, row in preds.iterrows()},
        }
    else:
        return {
            "has_recommendation": False,
            "product": None,
            "quantity": 0,
            "cook_time_minutes": 0,
            "ready_at": None,
            "reason": "Le bac est couvert pour les prochaines minutes.",
            "status_badge": "Grill OK",
            "dot_class": "k-dot",
            "model_forecast": {row["product"]: float(row["mu"]) for _, row in preds.iterrows()},
        }


@app.post("/api/simulate")
def run_simulation(
    policy_type: str = "newsvendor",
    seed: int = 42,
    pos_connected: bool = True,
):
    """Exécute une simulation complète d'une journée de service."""
    forecaster = get_or_train_forecaster()
    policy = decide_newsvendor if policy_type == "newsvendor" else decide_fixed_policy
    policy_name = "Agent Braise (Newsvendor)" if policy_type == "newsvendor" else "Politique Fixe"

    logs = simulate_day(
        day=date(2026, 11, 4),
        policy=policy,
        seed=seed,
        forecaster=forecaster,
        pos_connected=pos_connected,
        policy_name=policy_name,
    )

    # Sauvegarde Excel
    export_journal_to_excel(logs.journal_30min, EXCEL_FILE)
    sim_export = EXPORTS_DIR / f"journal_braise_sim_{logs.day}.xlsx"
    export_journal_to_excel(logs.journal_30min, sim_export)

    return {
        "day": str(logs.day),
        "policy": logs.policy_name,
        "kpis": logs.kpis,
        "tranches_count": len(logs.journal_30min),
        "sample_tranches": logs.journal_30min[-6:],
    }


@app.get("/api/inventory")
def get_inventory():
    """Retourne l'état complet de l'inventaire en temps réel et les pertes potentielles calculées avec le modèle."""
    forecaster = get_or_train_forecaster()
    forecast_rates = compute_realtime_forecast_rates(forecaster)
    summary = LIVE_INVENTORY.get_summary_report()
    losses = LIVE_INVENTORY.calculate_potential_losses(forecast_rates)
    return {
        "inventory": summary,
        "potential_losses": losses,
        "forecast_rates_per_min": forecast_rates,
    }


@app.get("/api/leader/dashboard")
def get_leader_dashboard():
    """Vue simplifiée pour le Shift Leader : feux tricolores alimentés dynamiquement par le modèle LightGBM."""
    forecaster = get_or_train_forecaster()
    forecast_rates = compute_realtime_forecast_rates(forecaster)
    return {
        "role": "shift_leader",
        "cards": LIVE_INVENTORY.get_leader_view(forecast_rates),
        "forecast_rates_per_min": forecast_rates,
        "sanitary_banner": "RÈGLE D'OR : Toute viande sortie du congélateur doit être cuite. Ne JAMAIS la recongeler.",
    }


@app.get("/api/manager/dashboard")
def get_manager_dashboard():
    """Vue analytique pour le Manager : valorisations en euros, pourcentages et risques IA alimentés par le modèle."""
    forecaster = get_or_train_forecaster()
    forecast_rates = compute_realtime_forecast_rates(forecaster)
    return {
        "role": "manager",
        "analytics": LIVE_INVENTORY.get_manager_view(forecast_rates),
        "forecast_rates_per_min": forecast_rates,
    }
    return {
        "role": "manager",
        "analytics": LIVE_INVENTORY.get_manager_view(forecast_rates),
    }


@app.post("/api/inventory/checkout")
def checkout_inventory(req: CheckoutRequest):
    """Sortie de matières premières du congélateur vers la zone de préparation."""
    try:
        batch = LIVE_INVENTORY.checkout_from_freezer(req.sku, req.quantity)
        return {
            "success": True,
            "message": f"{req.quantity} unités de '{req.sku}' sorties du congélateur.",
            "batch": {
                "batch_id": batch.batch_id,
                "quantity": batch.quantity,
                "time_out_min": batch.time_out_of_freezer_min,
            },
            "inventory": LIVE_INVENTORY.get_summary_report(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/inventory/return")
def return_inventory(req: ReturnRequest):
    """Réintégration d'un excès au congélateur (glaces autorisées, viande interdite)."""
    success, msg = LIVE_INVENTORY.return_to_freezer(req.sku, req.quantity)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {
        "success": True,
        "message": msg,
        "inventory": LIVE_INVENTORY.get_summary_report(),
    }


@app.post("/api/inventory/adjust")
def adjust_inventory(req: AdjustRequest):
    """Ajustement direct de l'inventaire physique par le manager."""
    try:
        LIVE_INVENTORY.adjust_stock(req.sku, req.freezer_stock, req.prep_stock)
        return {
            "success": True,
            "message": f"Stock ajusté pour '{req.sku}'.",
            "inventory": LIVE_INVENTORY.get_summary_report(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/inventory/tick-minute")
def tick_inventory_minute():
    """Fait vieillir les lots sortis d'une minute et élimine les lots expirés."""
    wasted = LIVE_INVENTORY.step_minute()
    return {
        "wasted_this_minute": wasted,
        "inventory": LIVE_INVENTORY.get_summary_report(),
    }


def build_dynamic_journal_from_test_data(
    target_date: Optional[str] = None,
    forecaster: Optional[DemandForecaster] = None,
) -> Tuple[List[Dict], str]:
    """Construit les 28 tranches de 30 min dynamiquement à partir des données réelles du test split et des prévisions LightGBM."""
    if forecaster is None:
        forecaster = get_or_train_forecaster()

    available_dates = get_test_dates()
    if not available_dates:
        raise ValueError("Aucune date de test disponible dans le dataset.")

    selected_date = target_date if (target_date and target_date in available_dates) else available_dates[0]
    df_day = get_daily_test_data(selected_date)

    preds = forecaster.predict(df_day)
    df_merged = df_day.copy()
    df_merged["mu"] = preds["mu"]
    df_merged["q10"] = preds["q10"]
    df_merged["q50"] = preds["q50"]
    df_merged["q90"] = preds["q90"]

    rows: List[Dict] = []
    for p_time, group in df_merged.groupby("periode_30min"):
        ca = 0.0
        mu = 0.0
        q10 = 0.0
        q50 = 0.0
        q90 = 0.0
        reel = 0
        for _, r in group.iterrows():
            prod = r["produit"]
            cfg = PRODUCTS.get(prod)
            s = cfg.steaks_per_burger if cfg else 1
            price = cfg.price_euros if cfg else 7.0
            cmd = int(r["commandes"])
            reel += cmd * s
            ca += cmd * price
            mu += float(r["mu"]) * s
            q10 += float(r["q10"]) * s
            q50 += float(r["q50"]) * s
            q90 += float(r["q90"]) * s

        dt = datetime.fromisoformat(str(p_time))
        err = reel - mu
        pot_loss = max(0.0, round(q90 - reel, 1)) if reel < q90 else 0.0
        real_loss = max(0, int(round((q50 - reel) * 0.35))) if reel < q50 else 0

        rows.append({
            "periode_30min": dt,
            "ca": round(ca, 2),
            "prevision_mu": round(mu, 1),
            "q10": round(q10, 1),
            "q50": round(q50, 1),
            "q90": round(q90, 1),
            "reel": int(reel),
            "erreur": round(err, 1),
            "perte_potentielle": pot_loss,
            "perte_reelle": real_loss,
        })

    rows.sort(key=lambda x: x["periode_30min"])
    return rows, selected_date


def build_dynamic_journal_from_live_tranches(
    req: LiveExportRequest,
    forecaster: Optional[DemandForecaster] = None,
) -> Tuple[List[Dict], str]:
    """Convertit les tranches en direct enregistrées par le navigateur lors du service actif en lignes de journal Excel."""
    date_str = req.date or datetime.now().strftime("%Y-%m-%d")
    rows: List[Dict] = []

    for t in req.periods:
        slot_hour = (t.k * 30) // 60
        slot_min = (t.k * 30) % 60
        dt_str = f"{date_str} {slot_hour:02d}:{slot_min:02d}:00"
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.now()

        mu = float(t.pred) if t.pred > 0 else float(t.act or 15.0)
        std_poisson = float(np.sqrt(max(1.0, mu)))
        q10 = max(0.0, round(mu - 1.28 * std_poisson, 1))
        q50 = round(mu, 1)
        q90 = round(mu + 1.28 * std_poisson, 1)
        reel = int(t.act)
        err = round(reel - mu, 1)
        pot_loss = max(0.0, float(t.cook - t.used)) if t.cook > t.used else 0.0
        real_loss = int(t.waste)

        rows.append({
            "periode_30min": dt,
            "ca": round(float(t.ca), 2),
            "prevision_mu": round(mu, 1),
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "reel": reel,
            "erreur": err,
            "perte_potentielle": round(pot_loss, 1),
            "perte_reelle": real_loss,
        })

    rows.sort(key=lambda x: x["periode_30min"])
    return rows, date_str


@app.get("/api/export-excel")
def download_excel(
    date: Optional[str] = Query(None, description="Date de service YYYY-MM-DD"),
    source: Optional[str] = Query("real_data", description="Source: 'real_data' ou 'simulation'"),
):
    """Télécharge le fichier Excel du journal par tranche de 30 min généré DYNAMIQUEMENT.

    Ne retourne JAMAIS un fichier statique figé :
    - Calcule à la volée les prévisions du modèle LightGBM (mu, q10, q50, q90)
    - Associe les commandes réelles et le CA pour la journée ciblée
    - Applique le schéma de colonnes officiel et le formatage monétaire
    """
    forecaster = get_or_train_forecaster()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if source == "simulation":
        target_day = datetime.strptime(date, "%Y-%m-%d").date() if date else datetime.now().date()
        logs = simulate_day(target_day, decide_newsvendor, forecaster=forecaster)
        rows = logs.journal_30min
        date_str = str(logs.day)
    else:
        rows, date_str = build_dynamic_journal_from_test_data(date, forecaster)

    export_path = EXPORTS_DIR / f"journal_braise_{date_str}.xlsx"
    export_journal_to_excel(rows, export_path)

    return FileResponse(
        path=export_path,
        filename=f"journal_braise_{date_str}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/api/export-excel/live")
def download_live_excel(req: LiveExportRequest):
    """Télécharge le fichier Excel généré dynamiquement à partir du service en direct de la cuisine."""
    forecaster = get_or_train_forecaster()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not req.periods:
        rows, date_str = build_dynamic_journal_from_test_data(req.date, forecaster)
        filename = f"journal_braise_{date_str}.xlsx"
    else:
        rows, date_str = build_dynamic_journal_from_live_tranches(req, forecaster)
        timestamp_suffix = datetime.now().strftime("%H%M%S")
        filename = f"journal_braise_live_{date_str}_{timestamp_suffix}.xlsx"

    export_path = EXPORTS_DIR / filename
    export_journal_to_excel(rows, export_path)

    return FileResponse(
        path=export_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
