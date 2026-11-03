"""
이상탐지 모델
  1) Isolation Forest  : 조건부 정규화된 특징 테이블 기반 (1차)
  2) 시퀀스 오토인코더 : 점 순서 시퀀스의 재구성 오차 기반 (PyTorch LSTM, 없으면 sklearn MLP)
  3) ScoreCalibrator   : 모델별 점수를 '정상 데이터 대비 백분위(0~1)'로 통일
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import Config


# ---------------------------------------------------------------------------
# 점수 보정: 정상 학습 데이터의 점수 분포 기준 백분위
# ---------------------------------------------------------------------------
class ScoreCalibrator:
    """raw 점수(클수록 이상) → 0~1. 0.95 = 정상 데이터의 95%보다 이상함"""

    def fit(self, raw: np.ndarray):
        self.ref = np.sort(np.asarray(raw, float))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, float)
        lo = np.searchsorted(self.ref, raw, side="left")
        hi = np.searchsorted(self.ref, raw, side="right")
        return ((lo + hi) / 2.0) / len(self.ref)


# ---------------------------------------------------------------------------
# 1) Isolation Forest
# ---------------------------------------------------------------------------
class IFDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = IsolationForest(
            n_estimators=cfg.if_n_estimators, max_samples=cfg.if_max_samples,
            contamination=cfg.if_contamination, random_state=cfg.random_state, n_jobs=-1)
        self.cal = ScoreCalibrator()

    def fit(self, Z_train: pd.DataFrame):
        self.model.fit(Z_train.to_numpy())
        self.cal.fit(self.raw(Z_train))
        return self

    def raw(self, Z: pd.DataFrame) -> np.ndarray:
        # score_samples: 클수록 정상 → 부호 반전
        return -self.model.score_samples(Z.to_numpy())

    def score(self, Z: pd.DataFrame) -> np.ndarray:
        return self.cal.transform(self.raw(Z))


# ---------------------------------------------------------------------------
# 2) 시퀀스 오토인코더
# ---------------------------------------------------------------------------
def _torch_available():
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


if _torch_available():
    import torch.nn as _nn

    class LSTMAE(_nn.Module):
        """LSTM 인코더 → 잠재벡터 → LSTM 디코더"""

        def __init__(self, F, hidden, latent):
            super().__init__()
            self.enc = _nn.LSTM(F, hidden, batch_first=True)
            self.to_z = _nn.Linear(hidden, latent)
            self.from_z = _nn.Linear(latent, hidden)
            self.dec = _nn.LSTM(hidden, hidden, batch_first=True)
            self.out = _nn.Linear(hidden, F)

        def forward(self, x):
            T = x.size(1)
            _, (h, _) = self.enc(x)
            z = self.to_z(h[-1])
            d = self.from_z(z).unsqueeze(1).repeat(1, T, 1)
            y, _ = self.dec(d)
            return self.out(y)


def _make_lstm_ae(F, hidden, latent):
    return LSTMAE(F, hidden, latent)


class SeqAutoencoder:
    """
    입력: X (N, T, F) — 점 순서대로 [속력, sin/cos 침로, 이동량 dx/dy, (시간간격)]
    정상 궤적만으로 학습 → 재구성 오차가 큰 궤적 = 평소와 다른 움직임
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        backend = cfg.ae_backend
        if backend == "auto":
            backend = "torch" if _torch_available() else "sklearn"
        self.backend = backend
        self.mu = None
        self.sd = None
        self.model = None

    # ---- 공통 ----
    def _norm(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def fit(self, X: np.ndarray):
        self.mu = X.reshape(-1, X.shape[2]).mean(0)
        self.sd = X.reshape(-1, X.shape[2]).std(0) + 1e-6
        Xn = self._norm(X)
        if self.backend == "torch":
            self._fit_torch(Xn)
        else:
            self._fit_sklearn(Xn)
        return self

    def recon_error(self, X: np.ndarray) -> np.ndarray:
        Xn = self._norm(X)
        R = self._predict_torch(Xn) if self.backend == "torch" else self._predict_sklearn(Xn)
        return ((Xn - R) ** 2).mean(axis=(1, 2))

    # ---- PyTorch LSTM AE ----
    def _build_torch(self, F):
        return _make_lstm_ae(F, self.cfg.ae_hidden, self.cfg.ae_latent)

    def _fit_torch(self, Xn):
        import torch
        torch.manual_seed(self.cfg.random_state)
        cfg = self.cfg
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.model = self._build_torch(Xn.shape[2]).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.ae_lr)
        loss_fn = torch.nn.MSELoss()
        data = torch.tensor(Xn)
        n_val = max(1, int(len(data) * 0.1))
        perm = torch.randperm(len(data))
        tr, va = data[perm[n_val:]], data[perm[:n_val]]
        best, best_state, patience = np.inf, None, 0
        for ep in range(cfg.ae_epochs):
            self.model.train()
            idx = torch.randperm(len(tr))
            for i in range(0, len(tr), cfg.ae_batch):
                xb = tr[idx[i:i + cfg.ae_batch]].to(dev)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), xb)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = loss_fn(self.model(va.to(dev)), va.to(dev)).item()
            if vl < best - 1e-5:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [AE] epoch {ep + 1:3d}  val_loss={vl:.4f}")
            if patience >= 5:
                print(f"  [AE] early stop at epoch {ep + 1}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _predict_torch(self, Xn):
        import torch
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xn), 512):
                xb = torch.tensor(Xn[i:i + 512]).to(self.device)
                outs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outs) if outs else Xn

    # ---- sklearn MLP AE (torch 없을 때 대체) ----
    def _fit_sklearn(self, Xn):
        from sklearn.neural_network import MLPRegressor
        cfg = self.cfg
        print("  [AE] PyTorch 없음 → sklearn MLP 오토인코더로 대체 학습")
        self.shape = Xn.shape[1:]
        flat = Xn.reshape(len(Xn), -1)
        self.model = MLPRegressor(hidden_layer_sizes=(cfg.ae_hidden * 2, cfg.ae_latent, cfg.ae_hidden * 2),
                                  max_iter=cfg.ae_epochs * 10, early_stopping=True,
                                  random_state=cfg.random_state, learning_rate_init=cfg.ae_lr)
        self.model.fit(flat, flat)

    def _predict_sklearn(self, Xn):
        return self.model.predict(Xn.reshape(len(Xn), -1)).reshape(Xn.shape)

    def save(self, path):
        import joblib
        if self.backend == "torch":
            import torch
            torch.save({"state": self.model.state_dict(), "mu": self.mu, "sd": self.sd}, str(path) + ".pt")
        else:
            joblib.dump({"model": self.model, "mu": self.mu, "sd": self.sd}, str(path) + ".joblib")


class GroupErrorNormalizer:
    """
    재구성 오차를 업종×시간대 그룹 기준으로 정규화
    (야간 통발처럼 원래 움직임이 복잡한 그룹은 오차가 커도 정상일 수 있으므로)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.st = {}

    @staticmethod
    def _ms(v):
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        return med, (mad if mad > 1e-9 else (np.std(v) if np.std(v) > 1e-9 else 1.0))

    def fit(self, log_err, types, bands):
        m = self.cfg.min_group_size
        self.st[("__all__",)] = self._ms(log_err)
        df = pd.DataFrame({"e": log_err, "t": types, "b": bands})
        for t, g in df.groupby("t"):
            if len(g) >= m:
                self.st[(t,)] = self._ms(g["e"].values)
        for (t, b), g in df.groupby(["t", "b"]):
            if len(g) >= m:
                self.st[(t, b)] = self._ms(g["e"].values)
        return self

    def transform(self, log_err, types, bands):
        out = np.zeros(len(log_err))
        for i, (e, t, b) in enumerate(zip(log_err, types, bands)):
            med, s = self.st.get((t, b)) or self.st.get((t,)) or self.st[("__all__",)]
            out[i] = (e - med) / s
        return out
