"""Application serveur web FastAPI (Braise - Architecture 1).

Fournit :
1. L'hébergement de l'interface graphique HTML `braise.html` sur la racine `/`
2. Les points d'entrée d'API pour exécuter la simulation d'une journée
3. Le téléchargement du journal Excel généré
"""

from datetime import date
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from pydantic import BaseModel
from src.data.generate import generate_30min_dataset
from src.export.excel import export_journal_to_excel
from src.forecast.model import DemandForecaster, set_global_forecaster
from src.policy.newsvendor import decide_fixed_policy, decide_newsvendor
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
EXCEL_FILE = BASE_DIR / "journal_braise_demo.xlsx"

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

# Initialisation lazy du modèle
_FORECASTER = None


def get_or_train_forecaster():
    global _FORECASTER
    if _FORECASTER is None:
        df = generate_30min_dataset(days_count=90, seed=42)
        _FORECASTER = DemandForecaster().fit(df)
        set_global_forecaster(_FORECASTER)
    return _FORECASTER


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

    return {
        "day": str(logs.day),
        "policy": logs.policy_name,
        "kpis": logs.kpis,
        "tranches_count": len(logs.journal_30min),
        "sample_tranches": logs.journal_30min[-6:],
    }


@app.get("/api/inventory")
def get_inventory():
    """Retourne l'état complet de l'inventaire en temps réel et les pertes potentielles."""
    summary = LIVE_INVENTORY.get_summary_report()
    forecast_rates = {
        "STEAK_WHOPPER": 0.65,
        "STEAK_HAMB": 0.80,
        "STEAK_MASTER": 0.45,
        "FRITES_SURGELEES": 0.50,
    }
    losses = LIVE_INVENTORY.calculate_potential_losses(forecast_rates)
    return {
        "inventory": summary,
        "potential_losses": losses,
    }


@app.get("/api/leader/dashboard")
def get_leader_dashboard():
    """Vue simplifiée pour le Shift Leader : feux tricolores, unités concrètes, sans pourcentages."""
    forecast_rates = {
        "STEAK_WHOPPER": 0.65,
        "STEAK_HAMB": 0.80,
        "STEAK_MASTER": 0.45,
        "FRITES_SURGELEES": 0.50,
    }
    return {
        "role": "shift_leader",
        "cards": LIVE_INVENTORY.get_leader_view(forecast_rates),
        "sanitary_banner": "RÈGLE D'OR : Toute viande sortie du congélateur doit être cuite. Ne JAMAIS la recongeler.",
    }


@app.get("/api/manager/dashboard")
def get_manager_dashboard():
    """Vue analytique pour le Manager : valorisations en euros, pourcentages de perte et marges."""
    forecast_rates = {
        "STEAK_WHOPPER": 0.65,
        "STEAK_HAMB": 0.80,
        "STEAK_MASTER": 0.45,
        "FRITES_SURGELEES": 0.50,
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


@app.get("/api/export-excel")
def download_excel():
    """Télécharge le fichier Excel du journal par tranche de 30 min."""
    if not EXCEL_FILE.exists():
        # Générer à la volée si inexistant
        run_simulation(policy_type="newsvendor", seed=42)
    return FileResponse(
        path=EXCEL_FILE,
        filename="journal_braise_30min.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
