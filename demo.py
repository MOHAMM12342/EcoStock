"""Démonstration complète de l'Architecture 1 (Braise).

Exécute la chaîne complète en un seul processus :
1. Génération de 12 mois de données synthétiques réalistes
2. Entraînement du modèle LightGBM Poisson et calcul de la baseline
3. Comparaison statistique WAPE / Biais / Pinball Loss (LightGBM vs Baseline)
4. Simulation comparative d'une journée de service sur les MÊMES graines aléatoires :
   - Agent Braise (Newsvendor + correction intra-journée)
   - VS Politique fixe classique (Buffer constant)
5. Export du journal 30 min vers Excel (journal_braise_demo.xlsx)
"""

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.generate import generate_30min_dataset
from src.evaluation.metrics import (
    compute_bias,
    compute_interval_coverage,
    compute_pinball_loss,
    compute_wape,
)
from src.export.excel import export_journal_to_excel
from src.forecast.model import DemandForecaster, set_global_forecaster
from src.policy.newsvendor import decide_fixed_policy, decide_newsvendor
from src.simulator.runner import simulate_day


def run_full_demo():
    print("=" * 70)
    print("🔥 BRAISE — ARCHITECTURE 1 : DÉMONSTRATION COMPLÈTE EN PYTHON")
    print("=" * 70)

    # 1. Génération des données
    print("\n[1/5] Génération de l'historique synthétique (12 mois)...")
    dataset = generate_30min_dataset(start_date=date(2025, 1, 1), days_count=365, seed=42)
    print(f"  -> {len(dataset):,} lignes générées sur 3 produits (Whopper, Hamb, Master).")

    # Split Train (10 mois) / Test (2 mois)
    split_date = pd.Timestamp("2025-11-01")
    train_df = dataset[dataset["periode_30min"] < split_date].copy()
    test_df = dataset[dataset["periode_30min"] >= split_date].copy()
    print(f"  -> Train : {len(train_df):,} lignes | Test : {len(test_df):,} lignes.")

    # 2. Entraînement des modèles
    print("\n[2/5] Entraînement des modèles LightGBM (perte Poisson) & Baseline...")
    forecaster = DemandForecaster().fit(train_df)
    set_global_forecaster(forecaster)

    # 3. Évaluation sur l'échantillon de test (WAPE, Biais, Pinball)
    print("\n[3/5] Évaluation prédictive sur l'échantillon test non vu :")
    preds_df = forecaster.predict(test_df)

    actuals = test_df["commandes"].values
    lgbm_mu = preds_df["mu"].values
    base_mu = preds_df["baseline_mu"].values
    q10 = preds_df["q10"].values
    q90 = preds_df["q90"].values

    wape_lgbm = compute_wape(actuals, lgbm_mu)
    wape_base = compute_wape(actuals, base_mu)
    bias_lgbm = compute_bias(actuals, lgbm_mu)
    bias_base = compute_bias(actuals, base_mu)
    pinball_50 = compute_pinball_loss(actuals, preds_df["q50"].values, 0.50)
    coverage = compute_interval_coverage(actuals, q10, q90)

    print(f"  -> WAPE LightGBM  : {wape_lgbm * 100:.2f} %  (vs Baseline: {wape_base * 100:.2f} %)")
    print(f"  -> Biais LightGBM : {bias_lgbm * 100:+.2f} % (vs Baseline: {bias_base * 100:+.2f} %)")
    print(f"  -> Pinball Loss (q50) : {pinball_50:.3f}")
    print(f"  -> Couverture [q10, q90] : {coverage * 100:.1f} % (Cible théorique : ~80 %)")

    assert wape_lgbm < wape_base, "LightGBM doit battre la baseline historique en WAPE !"
    print("  ✅ Critère validé : LightGBM surpasse la baseline historique.")

    # 4. Simulation comparative sur une journée de service complète
    print("\n[4/5] Simulation d'une journée de service (10h30 - 23h30) :")
    sim_day = date(2026, 11, 4)
    seed = 1042

    print("  -> Simulation 1 : Politique fixe (Buffer fixe de 8 steaks par produit)...")
    logs_fixed = simulate_day(
        day=sim_day,
        policy=decide_fixed_policy,
        seed=seed,
        forecaster=forecaster,
        policy_name="Politique Fixe (Sans IA)",
    )

    print("  -> Simulation 2 : Agent Braise (Newsvendor + correction dynamique)...")
    logs_agent = simulate_day(
        day=sim_day,
        policy=decide_newsvendor,
        seed=seed,
        forecaster=forecaster,
        policy_name="Agent Braise (Newsvendor)",
    )

    print("\n" + "-" * 70)
    print(f"{'Indicateur':<30} | {'Politique Fixe':<16} | {'Agent Braise':<16} | {'Gain / Écart'}")
    print("-" * 70)

    kpi_f = logs_fixed.kpis
    kpi_a = logs_agent.kpis

    diff_waste_pct = kpi_a["taux_gaspillage_pct"] - kpi_f["taux_gaspillage_pct"]
    diff_unmet = kpi_a["burgers_en_attente"] - kpi_f["burgers_en_attente"]
    diff_wasted_steaks = int(kpi_a["steaks_jetes"] - kpi_f["steaks_jetes"])

    print(f"{'Steaks cuits':<30} | {int(kpi_f['steaks_cuits']):<16} | {int(kpi_a['steaks_cuits']):<16} | {int(kpi_a['steaks_cuits'] - kpi_f['steaks_cuits']):+d}")
    print(f"{'Steaks jetés (perte)':<30} | {int(kpi_f['steaks_jetes']):<16} | {int(kpi_a['steaks_jetes']):<16} | {diff_wasted_steaks:+d}")
    print(f"{'Taux de gaspillage':<30} | {kpi_f['taux_gaspillage_pct']:>5.2f} %         | {kpi_a['taux_gaspillage_pct']:>5.2f} %         | {diff_waste_pct:+.2f} pts")
    print(f"{'Burgers en attente (rupture)':<30} | {kpi_f['burgers_en_attente']:>5.1f}           | {kpi_a['burgers_en_attente']:>5.1f}           | {diff_unmet:+.1f}")
    print(f"{'Taux de service':<30} | {kpi_f['taux_service_pct']:>5.2f} %         | {kpi_a['taux_service_pct']:>5.2f} %         | {kpi_a['taux_service_pct'] - kpi_f['taux_service_pct']:+.2f} pts")
    print(f"{'Chiffre d’affaires':<30} | {kpi_f['chiffre_affaires_euros']:>7.2f} €       | {kpi_a['chiffre_affaires_euros']:>7.2f} €       | {kpi_a['chiffre_affaires_euros'] - kpi_f['chiffre_affaires_euros']:+7.2f} €")
    print("-" * 70)

    # 5. Export Excel du journal par tranche de 30 min
    out_excel_path = Path("journal_braise_demo.xlsx")
    print(f"\n[5/5] Export du journal par tranche de 30 min vers '{out_excel_path}'...")
    export_journal_to_excel(logs_agent.journal_30min, out_excel_path)
    print(f"  -> Fichier Excel généré ({len(logs_agent.journal_30min)} tranches de 30 min).")

    print("\n" + "=" * 70)
    print("✨ DÉMO ARCHITECTURE 1 RÉUSSIE AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    run_full_demo()

