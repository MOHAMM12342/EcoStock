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

