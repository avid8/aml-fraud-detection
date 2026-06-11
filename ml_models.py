"""
AML Pipeline — لایه ML
سه مدل موازی:
  1. LSTM Supervised — تشخیص تقلب با لیبل
  2. Autoencoder Unsupervised — anomaly detection بدون لیبل
  3. Behavioral Clustering — پروفایل رفتاری با K-Means
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# feature های ورودی به مدل‌ها
# ─────────────────────────────────────────────
SEQUENCE_FEATURES = [
    "amount_rial_norm",       # مبلغ نرمال‌شده
    "is_night",               # ساعت شب
    "ip_confidence",          # اعتماد IP
    "is_carrier_nat",         # NAT
    "hour_sin",               # ساعت (sin)
    "hour_cos",               # ساعت (cos)
    "day_of_week_sin",        # روز هفته
    "day_of_week_cos",
]

BEHAVIORAL_FEATURES = [
    "avg_amount_30d",
    "std_amount_30d",
    "night_ratio_30d",
    "fail_ratio_30d",
    "unique_ips_30d",
    "unique_devices_30d",
    "tx_count_30d",
    "avg_tx_gap_hours_30d",   # میانگین فاصله بین تراکنش‌ها
]

SEQ_LEN       = 20    # طول دنباله تراکنش‌ها
INPUT_DIM     = len(SEQUENCE_FEATURES)
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
BEHAVIOR_DIM  = len(BEHAVIORAL_FEATURES)
N_CLUSTERS    = 5     # تعداد پروفایل رفتاری


# ─────────────────────────────────────────────
# 1. LSTM Supervised
# ─────────────────────────────────────────────
class FraudLSTM(nn.Module):
    """
    ورودی: دنباله‌ای از تراکنش‌ها (batch, seq_len, input_dim)
    خروجی: احتمال تقلب (batch, 1)
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # attention روی timestep ها
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        return self.classifier(context)


class LSTMTrainer:
    def __init__(self, model: FraudLSTM, lr: float = 1e-3, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 64) -> float:
        """
        X: (N, seq_len, input_dim)
        y: (N,) — 0 یا 1
        """
        self.model.train()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        total_loss = 0.0
        n_batches  = 0
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size]
            yb = y_t[i:i+batch_size]
            self.opt.zero_grad()
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_t).cpu().numpy().flatten()
        return probs

    def save(self, path: str = "models/lstm.pt"):
        torch.save(self.model.state_dict(), path)
        logger.info(f"LSTM saved to {path}")

    def load(self, path: str = "models/lstm.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"LSTM loaded from {path}")


# ─────────────────────────────────────────────
# 2. Autoencoder Unsupervised
# ─────────────────────────────────────────────
class FraudAutoencoder(nn.Module):
    """
    ورودی: دنباله تراکنش (batch, seq_len * input_dim)
    خروجی: reconstruction — خطای بالا = anomaly
    """
    def __init__(self, input_dim: int = SEQ_LEN * INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderTrainer:
    def __init__(self, model: FraudAutoencoder,
                 lr: float = 1e-3, device: str = "cpu"):
        self.model    = model.to(device)
        self.device   = device
        self.opt      = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn  = nn.MSELoss()
        self.threshold: Optional[float] = None

    def train_epoch(self, X: np.ndarray, batch_size: int = 64) -> float:
        """X: (N, seq_len, input_dim) — فقط تراکنش‌های نرمال"""
        self.model.train()
        X_flat = X.reshape(len(X), -1)
        X_t    = torch.FloatTensor(X_flat).to(self.device)
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size]
            self.opt.zero_grad()
            recon = self.model(xb)
            loss  = self.loss_fn(recon, xb)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    def fit_threshold(self, X_normal: np.ndarray, percentile: float = 95.0):
        """آستانه رو از توزیع خطای داده نرمال تعیین می‌کنه"""
        errors = self.reconstruction_errors(X_normal)
        self.threshold = float(np.percentile(errors, percentile))
        logger.info(f"Autoencoder threshold set to {self.threshold:.6f} (p{percentile})")

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_flat = X.reshape(len(X), -1)
        with torch.no_grad():
            X_t   = torch.FloatTensor(X_flat).to(self.device)
            recon = self.model(X_t)
            errors = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict_anomaly(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        برگشت: (is_anomaly, scores)
        is_anomaly: bool array
        scores: خطای بازسازی نرمال‌شده به 0-1
        """
        errors = self.reconstruction_errors(X)
        if self.threshold is None:
            raise ValueError("اول fit_threshold رو صدا بزن")
        is_anomaly = errors > self.threshold
        scores     = np.clip(errors / (self.threshold * 2), 0, 1)
        return is_anomaly, scores

    def save(self, path: str = "models/autoencoder.pt"):
        data = {"state_dict": self.model.state_dict(), "threshold": self.threshold}
        torch.save(data, path)
        logger.info(f"Autoencoder saved to {path}")

    def load(self, path: str = "models/autoencoder.pt"):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["state_dict"])
        self.threshold = data.get("threshold")
        logger.info(f"Autoencoder loaded from {path}")


# ─────────────────────────────────────────────
# 3. Behavioral Clustering (پروفایل رفتاری)
# ─────────────────────────────────────────────
PROFILE_NAMES = {
    0: "محافظه‌کار",    # مبلغ پایین، فرکانس کم
    1: "عادی",          # رفتار معمول
    2: "پرتراکنش",      # فرکانس بالا
    3: "پرمبلغ",        # مبلغ بالا
    4: "مشکوک",         # ناشناخته / نامعمول
}


class BehavioralProfiler:
    """
    برای هر Person یه پروفایل رفتاری تعیین می‌کنه.
    اگه تراکنش جدید از پروفایل منحرف بشه — سیگنال تقلب.
    """
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans:  Optional[KMeans]         = None
        self.scaler:  Optional[StandardScaler] = None

    def fit(self, X: np.ndarray):
        """X: (N, BEHAVIOR_DIM) — ویژگی‌های رفتاری ۳۰ روزه"""
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        logger.info(f"BehavioralProfiler fitted on {len(X)} persons")

    def predict_profile(self, x: np.ndarray) -> tuple[int, str, float]:
        """
        x: (BEHAVIOR_DIM,) — ویژگی‌های رفتاری یه نفر
        برگشت: (cluster_id, profile_name, distance_to_center)
        """
        if self.kmeans is None or self.scaler is None:
            raise ValueError("اول fit() رو صدا بزن")
        x_scaled  = self.scaler.transform(x.reshape(1, -1))
        cluster   = int(self.kmeans.predict(x_scaled)[0])
        center    = self.kmeans.cluster_centers_[cluster]
        distance  = float(np.linalg.norm(x_scaled - center))
        name      = PROFILE_NAMES.get(cluster, "نامشخص")
        return cluster, name, distance

    def is_anomalous_behavior(self, x: np.ndarray,
                               threshold_multiplier: float = 2.5) -> tuple[bool, float]:
        """
        اگه فاصله از مرکز cluster بیش از threshold_multiplier برابر
        میانگین فاصله‌ها باشه — رفتار غیرعادیه
        """
        cluster, _, distance = self.predict_profile(x)
        x_scaled  = self.scaler.transform(x.reshape(1, -1))
        all_dists = np.linalg.norm(
            self.scaler.transform(
                self.kmeans.cluster_centers_
            ) - x_scaled, axis=1
        )
        mean_dist = float(np.mean(all_dists))
        is_anom   = distance > mean_dist * threshold_multiplier
        score     = min(1.0, distance / (mean_dist * threshold_multiplier))
        return is_anom, score

    def save(self, path: str = "models/profiler.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"kmeans": self.kmeans, "scaler": self.scaler}, f)
        logger.info(f"BehavioralProfiler saved to {path}")

    def load(self, path: str = "models/profiler.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.kmeans = data["kmeans"]
        self.scaler = data["scaler"]
        logger.info(f"BehavioralProfiler loaded from {path}")


# ─────────────────────────────────────────────
# 4. ML Scorer — ترکیب هر سه مدل
# ─────────────────────────────────────────────
class MLScorer:
    """
    هر سه مدل رو ترکیب می‌کنه و یه score نهایی برمی‌گردونه.
    وزن‌ها قابل تنظیم هستن.
    """
    WEIGHTS = {
        "lstm":        0.5,
        "autoencoder": 0.3,
        "behavioral":  0.2,
    }

    def __init__(self,
                 lstm_trainer:    Optional[LSTMTrainer]       = None,
                 ae_trainer:      Optional[AutoencoderTrainer] = None,
                 profiler:        Optional[BehavioralProfiler] = None):
        self.lstm   = lstm_trainer
        self.ae     = ae_trainer
        self.profiler = profiler

    def score(self,
              sequence:   Optional[np.ndarray] = None,
              behavioral: Optional[np.ndarray] = None) -> dict:
        """
        sequence:   (1, seq_len, input_dim) — دنباله تراکنش‌ها
        behavioral: (BEHAVIOR_DIM,) — ویژگی‌های رفتاری ۳۰ روزه

        خروجی:
        {
            lstm_score, ae_score, behavioral_score,
            final_score, is_anomaly, profile_name
        }
        """
        scores = {}
        total  = 0.0
        weight_sum = 0.0

        # LSTM
        if self.lstm and sequence is not None:
            lstm_prob = float(self.lstm.predict_proba(sequence)[0])
            scores["lstm_score"] = lstm_prob
            total      += lstm_prob * self.WEIGHTS["lstm"]
            weight_sum += self.WEIGHTS["lstm"]
        else:
            scores["lstm_score"] = None

        # Autoencoder
        if self.ae and sequence is not None:
            _, ae_scores = self.ae.predict_anomaly(sequence)
            ae_score = float(ae_scores[0])
            scores["ae_score"] = ae_score
            total      += ae_score * self.WEIGHTS["autoencoder"]
            weight_sum += self.WEIGHTS["autoencoder"]
        else:
            scores["ae_score"] = None

        # Behavioral
        if self.profiler and behavioral is not None:
            is_anom, beh_score = self.profiler.is_anomalous_behavior(behavioral)
            _, profile_name, _ = self.profiler.predict_profile(behavioral)
            scores["behavioral_score"] = beh_score
            scores["profile_name"]     = profile_name
            scores["behavioral_anomaly"] = is_anom
            total      += beh_score * self.WEIGHTS["behavioral"]
            weight_sum += self.WEIGHTS["behavioral"]
        else:
            scores["behavioral_score"]   = None
            scores["profile_name"]       = None
            scores["behavioral_anomaly"] = None

        final = total / weight_sum if weight_sum > 0 else 0.0
        scores["final_score"] = round(min(1.0, final), 4)
        scores["is_anomaly"]  = scores["final_score"] >= 0.5

        return scores
