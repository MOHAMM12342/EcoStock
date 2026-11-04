"""Tests unitaires pour l'intégration du modèle PyTorch TorchScript (forecast.pt)."""

import numpy as np
import pytest

from src.forecast.pt_model import PyTorchForecastTrees, get_pt_model


def test_pt_model_initialization():
    """Vérifie que le modèle se charge et extrait les poids correctement."""
    model = get_pt_model()
    assert model.num_trees == 315
    assert model.tree_depth == 6
    assert model.output_dim == 15
    assert model.features.shape == (315, 6)
    assert model.borders.shape == (315, 6)
    assert model.leaves.shape == (315, 64, 15)
    assert model.bias.shape == (15,)


def test_pt_model_forward_inference():
    """Vérifie la passe avant sur différentes tailles de batch."""
    model = get_pt_model()

    # Batch de 1
    x1 = np.zeros((1, 61), dtype=np.float32)
    pred1 = model.forward(x1)
    assert pred1.shape == (1, 15)
    assert np.all(pred1 >= 0.0)

    # Batch de 4
    x4 = np.random.randn(4, 61).astype(np.float32)
    pred4 = model.forward(x4)
    assert pred4.shape == (4, 15)
    assert np.all(pred4 >= 0.0)

    # Erreur si dimension incorrecte
    with pytest.raises(RuntimeError):
        model.forward(np.zeros((1, 10), dtype=np.float32))


def test_pt_model_deployed_cummax():
    """Vérifie que la méthode deployed applique bien le cummax sur l'horizon temporel."""
    model = get_pt_model()
    x = np.zeros((1, 61), dtype=np.float32)
    deployed = model.deployed(x)
    assert deployed.shape == (1, 15)

    grid = deployed.reshape(5, 3)
    # Vérifie la non-décroissance sur l'horizon pour chaque produit
    for prod_idx in range(3):
        col = grid[:, prod_idx]
        for h in range(4):
            assert col[h + 1] >= col[h], f"Le cumul temporel au pas {h+1} doit être >= pas {h}"


def test_pt_model_build_features_and_service_prediction():
    """Vérifie la génération des 61 features et la structure de réponse opérationnelle."""
    model = get_pt_model()
    from datetime import datetime

    now = datetime(2026, 11, 4, 12, 30)
    service_res = model.predict_for_service(
        dt=now,
        recent_sales={"Whopper": 2.0, "Hamb": 2.5, "Master": 1.0},
    )

    assert "predictions_by_horizon" in service_res
    preds = service_res["predictions_by_horizon"]
    assert "Whopper" in preds and "Hamb" in preds and "Master" in preds
    assert len(preds["Whopper"]) == 5
    assert service_res["total_steaks_predicted_15min"] > 0
    assert service_res["inference_engine"] in ("Pure NumPy", "PyTorch")

