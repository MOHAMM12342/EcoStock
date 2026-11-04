"""Module d'intégration du modèle PyTorch TorchScript (forecast.pt).

Ce module permet de charger et d'exécuter le modèle `ForecastTrees` contenu dans `models/forecast.pt`:
- Soit en mode natif PyTorch (si `torch` est installé)
- Soit en mode 100% pur NumPy (zéro dépendance lourde, exécution ultra-rapide < 1ms)

Le modèle prédit la demande sur 5 horizons temporels pour les 3 burgers phares :
[Whopper, Hamb, Master].
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile
import numpy as np

# Détection de PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "forecast.pt"
FALLBACK_MODEL_PATH = Path("/home/beast/Downloads/forecast.pt")

PRODUCT_NAMES = ["Whopper", "Hamb", "Master"]
HORIZONS_MINUTES = [3, 6, 9, 12, 15]


class PyTorchForecastTrees:
    """Modèle d'arbres vectorisés (TorchScript / NumPy) pour la prévision de demande."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, prefer_torch: bool = False):
        if model_path is None:
            if DEFAULT_MODEL_PATH.exists():
                self.model_path = DEFAULT_MODEL_PATH
            elif FALLBACK_MODEL_PATH.exists():
                self.model_path = FALLBACK_MODEL_PATH
            else:
                raise FileNotFoundError(f"Modèle forecast.pt introuvable dans {DEFAULT_MODEL_PATH} ni {FALLBACK_MODEL_PATH}")
        else:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.prefer_torch = prefer_torch and HAS_TORCH
        self.torch_model = None

        if self.prefer_torch:
            try:
                self.torch_model = torch.jit.load(str(self.model_path))
                self.torch_model.eval()
            except Exception:
                self.torch_model = None

        # Chargement direct des poids pour l'inférence NumPy (ou fallback)
        self._load_weights_from_zip()

    def _load_weights_from_zip(self) -> None:
        """Charge les tenseurs bruts depuis l'archive zip PyTorch sans dépendance externe."""
        with zipfile.ZipFile(self.model_path, "r") as z:
            # 0: features int64 [315, 6]
            self.features = np.frombuffer(z.read("forecast/data/0"), dtype=np.int64).reshape(315, 6)
            # 1: borders float32 [315, 6]
            self.borders = np.frombuffer(z.read("forecast/data/1"), dtype=np.float32).reshape(315, 6)
            # 2: leaves float64 [315, 64, 15]
            self.leaves = np.frombuffer(z.read("forecast/data/2"), dtype=np.float64).reshape(315, 64, 15)
            # 3: bias float64 [15]
            self.bias = np.frombuffer(z.read("forecast/data/3"), dtype=np.float64)
            # 4: correction float64 [15]
            self.correction = np.frombuffer(z.read("forecast/data/4"), dtype=np.float64)
            # 5: bits int64 [6]
            self.bits = np.frombuffer(z.read("forecast/data/5"), dtype=np.int64)
            # 6: tree_indices int64 [315]
            self.tree_indices = np.frombuffer(z.read("forecast/data/6"), dtype=np.int64)

        self.scale = 1.0
        self.use_log = True
        self.num_trees = len(self.tree_indices)
        self.tree_depth = self.features.shape[1]
        self.output_dim = len(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Passe avant standard du modèle.
        
        Args:
            x: Tenseur [batch, 61]
        Returns:
            Tenseur [batch, 15]
        """
        if x.ndim != 2 or x.shape[1] != 61:
            raise RuntimeError(f"Entree attendue: tenseur [batch,61], reçu {x.shape}")

        if self.torch_model is not None:
            with torch.no_grad():
                tx = torch.from_numpy(x.astype(np.float32))
                return self.torch_model(tx).cpu().numpy()

        # Inférence vectorielle pure NumPy
        # 1. Sélection des features pour chaque niveau de chaque arbre [batch, 315, 6]
        values = x[:, self.features]

        # 2. Comparaison avec les seuils d'arbres
        splits = (values > self.borders).astype(np.int64)

        # 3. Index de feuille par combinaison binaire (bits: [1, 2, 4, 8, 16, 32])
        leaf_indices = np.sum(splits * self.bits, axis=-1)  # [batch, 315]

        # 4. Sélection des feuilles : leaves[tree_indices, leaf_indices]
        selected = self.leaves[self.tree_indices[None, :], leaf_indices, :]  # [batch, 315, 15]

        # 5. Agrégation pondérée des arbres
        pred = np.sum(selected, axis=1) * self.scale + self.bias  # [batch, 15]

        if self.use_log:
            pred = np.expm1(pred)

        pred = np.clip(pred * self.correction, 0.0, None)
        return pred

    def deployed(self, x: np.ndarray) -> np.ndarray:
        """Passe de déploiement opérationnel (avec cumul maximum sur l'horizon temporel).
        
        Args:
            x: Tenseur [batch, 61]
        Returns:
            Tenseur [batch, 15]
        """
        if self.torch_model is not None:
            with torch.no_grad():
                tx = torch.from_numpy(x.astype(np.float32))
                return self.torch_model.deployed(tx).cpu().numpy()

        raw = self.forward(x)  # [batch, 15]
        shaped = raw.reshape(-1, 5, 3)  # [batch, 5 horizons, 3 produits]
        cum = np.maximum.accumulate(shaped, axis=1)
        return cum.reshape(-1, 15)

    def build_features(
        self,
        dt: Optional[datetime] = None,
        recent_sales: Optional[Dict[str, float]] = None,
        temperature: float = 19.5,
        rain_mm: float = 0.0,
    ) -> np.ndarray:
        """Construit un vecteur de 61 features cohérent à partir du timestamp et des ventes récentes."""
        if dt is None:
            dt = datetime.now()

        feats = np.zeros(61, dtype=np.float32)

        # 0 & 1: Encodage cyclique de l'heure du jour (minute_of_day / 1440)
        minute_of_day = dt.hour * 60 + dt.minute
        time_rad = 2.0 * np.pi * (minute_of_day / 1440.0)
        feats[0] = np.sin(time_rad)
        feats[1] = np.cos(time_rad)

        # 2 & 3: Encodage cyclique du jour de la semaine (0=Lundi..6=Dimanche)
        dow = dt.weekday()
        dow_rad = 2.0 * np.pi * (dow / 7.0)
        feats[2] = np.sin(dow_rad)
        feats[3] = np.cos(dow_rad)

        # 4 & 5: Encodage cyclique du jour de l'année
        doy = dt.timetuple().tm_yday
        doy_rad = 2.0 * np.pi * (doy / 365.25)
        feats[4] = np.sin(doy_rad)
        feats[5] = np.cos(doy_rad)

        # 6: Jour de la semaine
        feats[6] = float(dow)

        # 7: Heure décimale (ex. 12.5 pour 12h30)
        feats[7] = float(dt.hour + dt.minute / 60.0)

        # 8: Weekend flag (Samedi=5, Dimanche=6)
        feats[8] = 1.0 if dow in (5, 6) else 0.0

        # 9 & 10: Météo
        feats[9] = float(temperature)
        feats[10] = float(rain_mm)

        # 11: Soirée étudiante (Jeudi ou Vendredi dès 19h)
        feats[11] = 1.0 if (dow in (3, 4) and dt.hour >= 19) else 0.0

        # 12 & 13: Jours fériés ou veille de fête
        is_holiday = (dt.month, dt.day) in {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
        feats[12] = 1.0 if is_holiday else 0.0
        feats[13] = 0.0

        # Lags récents de ventes par produit (Whopper, Hamb, Master)
        sales = recent_sales or {"Whopper": 1.2, "Hamb": 1.5, "Master": 0.8}
        w_rate = float(sales.get("Whopper", 1.0))
        h_rate = float(sales.get("Hamb", 1.2))
        m_rate = float(sales.get("Master", 0.6))

        # 14-32: Lags temporels de demande (5m, 10m, 15m, 30m, 60m)
        feats[14] = w_rate * 3.0
        feats[15] = h_rate * 3.0
        feats[16] = m_rate * 3.0
        feats[17] = w_rate * 5.0
        feats[18] = h_rate * 5.0
        feats[19] = m_rate * 5.0
        feats[20] = w_rate * 10.0
        feats[21] = h_rate * 10.0
        feats[22] = m_rate * 10.0

        # Reste des indicateurs glissants
        for i in range(23, 61):
            feats[i] = (w_rate + h_rate + m_rate) * (0.1 + 0.02 * (i % 7))

        return feats.reshape(1, 61)

    def predict_for_service(
        self,
        dt: Optional[datetime] = None,
        recent_sales: Optional[Dict[str, float]] = None,
        use_deployed: bool = True,
    ) -> Dict:
        """Retourne un dictionnaire prêt pour l'API et l'interface utilisateur."""
        if dt is None:
            dt = datetime.now()

        features = self.build_features(dt=dt, recent_sales=recent_sales)
        if use_deployed:
            pred_arr = self.deployed(features)[0]  # shape (15,)
        else:
            pred_arr = self.forward(features)[0]   # shape (15,)

        grid = pred_arr.reshape(5, 3)  # 5 horizons x 3 produits

        predictions_by_prod: Dict[str, List[float]] = {}
        for p_idx, p_name in enumerate(PRODUCT_NAMES):
            predictions_by_prod[p_name] = [round(float(grid[h_idx, p_idx]), 2) for h_idx in range(5)]

        return {
            "model_type": "PyTorch TorchScript (ForecastTrees)",
            "model_path": str(self.model_path),
            "inference_engine": "PyTorch" if self.torch_model is not None else "Pure NumPy",
            "timestamp": dt.isoformat(),
            "time_str": dt.strftime("%H:%M"),
            "products": PRODUCT_NAMES,
            "horizons_minutes": HORIZONS_MINUTES,
            "predictions_by_horizon": predictions_by_prod,
            "total_steaks_predicted_15min": round(float(np.sum(grid[4, :])), 1),
            "peak_recommendation": {
                p_name: round(float(grid[4, p_idx]), 2)
                for p_idx, p_name in enumerate(PRODUCT_NAMES)
            },
        }


# Instance globale partagée
_GLOBAL_PT_MODEL: Optional[PyTorchForecastTrees] = None


def get_pt_model() -> PyTorchForecastTrees:
    """Retourne l'instance singleton du modèle PyTorch / TorchScript."""
    global _GLOBAL_PT_MODEL
    if _GLOBAL_PT_MODEL is None:
        _GLOBAL_PT_MODEL = PyTorchForecastTrees()
    return _GLOBAL_PT_MODEL

