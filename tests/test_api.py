"""Tests d'intégration de l'API Braise (Architecture 1)."""

import pytest
from starlette.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_inventory_get(client):
    res = client.get("/api/inventory")
    assert res.status_code == 200
    data = res.json()
    assert "inventory" in data
    assert "potential_losses" in data
    assert any(item["sku"] == "STEAK_WHOPPER" for item in data["inventory"])


def test_api_inventory_checkout(client):
    payload = {"sku": "STEAK_WHOPPER", "quantity": 10}
    res = client.post("/api/inventory/checkout", json=payload)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_api_inventory_return_rules(client):
    # Glaces : réintégration autorisée
    payload_glace = {"sku": "MIX_GLACE", "quantity": 2}
    res_glace = client.post("/api/inventory/return", json=payload_glace)
    assert res_glace.status_code == 200
    assert res_glace.json()["success"] is True

    # Viande : réintégration interdite (erreur 400 attendue)
    payload_viande = {"sku": "STEAK_WHOPPER", "quantity": 5}
    res_viande = client.post("/api/inventory/return", json=payload_viande)
    assert res_viande.status_code == 400
    assert "sécurité sanitaire" in res_viande.json()["detail"].lower()


def test_api_inventory_adjust(client):
    payload = {"sku": "STEAK_MASTER", "freezer_stock": 180, "prep_stock": 25}
    res = client.post("/api/inventory/adjust", json=payload)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_api_leader_dashboard(client):
    res = client.get("/api/leader/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "shift_leader"
    assert "cards" in data
    # Doit contenir des feux tricolores simples (GREEN, YELLOW, RED) et des messages sans formule
    for card in data["cards"]:
        assert card["traffic_light"] in ("GREEN", "YELLOW", "RED")
        assert "quick_action_qty" in card


def test_api_manager_dashboard(client):
    res = client.get("/api/manager/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "manager"
    analytics = data["analytics"]
    assert "total_stock_valuation_euros" in analytics
    assert "global_waste_rate_pct" in analytics
    assert "target_max_waste_rate_pct" in analytics


def test_api_forecast(client):
    res = client.get("/api/forecast")
    assert res.status_code == 200
    data = res.json()
    assert "predictions_by_horizon" in data
    assert "Whopper" in data["predictions_by_horizon"]
    assert "Hamb" in data["predictions_by_horizon"]
    assert "Master" in data["predictions_by_horizon"]
    assert len(data["predictions_by_horizon"]["Whopper"]) == 5
    assert data["model_type"] == "LightGBM Poisson Regression"
    assert "predictions_by_product" in data


def test_api_forecast_model_info(client):
    res = client.get("/api/forecast/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_type"] == "LightGBM Poisson Regression"
    assert data["class_name"] == "DemandForecaster"
    assert data["num_product_models"] == 3
    assert data["inference_engine"] == "LightGBM"


def test_api_forecast_daily_update(client):
    res = client.post("/api/forecast/daily-update", json={"days_count": 1, "n_additional_trees": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "trees_per_product" in data
    assert data["trees_per_product"]["Whopper"] >= 150


def test_api_dataset_info(client):
    res = client.get("/api/dataset/info")
    assert res.status_code == 200
    data = res.json()
    assert data["total_tickets"] == 944451
    assert data["status"] == "extracted_and_ready"
    assert data["train_period"]["days_count"] >= 400


def test_api_kitchen_recommendation(client):
    res = client.get("/api/kitchen/recommendation?bins_whopper=2&bins_hamb=2&bins_master=1")
    assert res.status_code == 200
    data = res.json()
    assert "has_recommendation" in data
    assert "status_badge" in data
    assert "model_forecast" in data
    assert "Whopper" in data["model_forecast"]


def test_api_leader_dashboard_dynamic_forecast_rates(client):
    res = client.get("/api/leader/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "forecast_rates_per_min" in data
    assert data["forecast_rates_per_min"]["STEAK_WHOPPER"] > 0


def test_api_export_excel_dynamic_get(client):
    """Vérifie que l'export Excel est généré dynamiquement sur les données réelles et prévisions du modèle."""
    import io
    import openpyxl

    res = client.get("/api/export-excel")
    assert res.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in res.headers["content-type"]
    assert "journal_braise_2026-03-01.xlsx" in res.headers["content-disposition"]

    # Chargement et validation du classeur Excel
    wb = openpyxl.load_workbook(io.BytesIO(res.content))
    assert "Journal Braise 30min" in wb.sheetnames
    ws = wb["Journal Braise 30min"]

    # 1 ligne d'en-tête + 28 tranches de 30 min (10h00 à 23h30)
    assert ws.max_row == 29
    assert ws.max_column == 10

    # Vérification des colonnes
    headers = [cell.value for cell in ws[1]]
    assert "Période 30 min" in headers[0]
    assert "Prévision (µ)" in headers[2]
    assert "Réel (steaks)" in headers[6]

    # Vérification des valeurs dynamiques de la 1ère tranche
    assert "2026-03-01 10:00" in str(ws.cell(row=2, column=1).value)
    assert ws.cell(row=2, column=2).value > 0  # CA
    assert ws.cell(row=2, column=3).value > 0  # prevision_mu
    assert ws.cell(row=2, column=7).value > 0  # reel


def test_api_export_excel_custom_date(client):
    """Vérifie que l'export Excel s'adapte dynamiquement à une date demandée."""
    import io
    import openpyxl

    res = client.get("/api/export-excel?date=2026-03-05")
    assert res.status_code == 200
    assert "journal_braise_2026-03-05.xlsx" in res.headers["content-disposition"]

    wb = openpyxl.load_workbook(io.BytesIO(res.content))
    ws = wb.active
    assert ws.max_row == 29
    assert "2026-03-05 10:00" in str(ws.cell(row=2, column=1).value)


def test_api_export_excel_live_post(client):
    """Vérifie l'export Excel en direct des tranches actives de la cuisine."""
    import io
    import openpyxl

    payload = {
        "date": "2026-11-05",
        "session_minutes": 750,
        "periods": [
            {
                "k": 24,
                "time_label": "12:00–12:30",
                "ca": 382.5,
                "used": 44,
                "cook": 50,
                "waste": 2,
                "pred": 46.2,
                "act": 45,
            },
            {
                "k": 25,
                "time_label": "12:30–13:00",
                "ca": 520.0,
                "used": 60,
                "cook": 62,
                "waste": 0,
                "pred": 63.8,
                "act": 61,
            },
        ],
    }

    res = client.post("/api/export-excel/live", json=payload)
    assert res.status_code == 200
    assert "journal_braise_live_2026-11-05" in res.headers["content-disposition"]

    wb = openpyxl.load_workbook(io.BytesIO(res.content))
    ws = wb.active
    assert ws.max_row == 3  # 1 en-tête + 2 tranches envoyées
    assert "2026-11-05 12:00" in str(ws.cell(row=2, column=1).value)
    assert ws.cell(row=2, column=2).value == 382.5
    assert ws.cell(row=2, column=3).value == 46.2
    assert ws.cell(row=2, column=7).value == 45
    assert ws.cell(row=2, column=10).value == 2  # perte réelle





