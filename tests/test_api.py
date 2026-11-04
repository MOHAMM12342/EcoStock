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


