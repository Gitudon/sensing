import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 設定パラメータ ---
M = 3  # センサ数
J = 2  # 信号源数
I = 10  # スナップショット数
angles_deg = np.array([90, 45])  # 真の到来方向 (degree)
snr_db = 10  # SNR [dB]

# 物理設定
lam = 1.0  # 波長
d = lam / 2  # センサ間隔
m = np.arange(M).reshape(-1, 1)
theta_true = np.radians(angles_deg)

# --- 1. 信号データ生成 ---
# ステアリングベクトル行列 A
A = np.exp(-2j * np.pi * d / lam * m @ np.sin(theta_true).reshape(1, -1))

# 信号 (複素ガウス信号)
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)

# 雑音 (白色ガウス雑音)
sigma2 = 10 ** (-snr_db / 10)
N = np.sqrt(sigma2 / 2) * (np.random.randn(M, I) + 1j * np.random.randn(M, I))

# 観測データ X
X = A @ S + N

# 相関行列 R
R = (X @ X.conj().T) / I

# --- 2. 推定手法の実装 ---
search_theta = np.linspace(-90, 90, 181)
search_rad = np.radians(search_theta)


def get_a(theta):
    return np.exp(-2j * np.pi * d / lam * np.arange(M) * np.sin(theta)).reshape(-1, 1)


# (1) ビームフォーミング法 (Conventional Beamformer)
p_bf = []
for th in search_rad:
    a = get_a(th)
    p_bf.append(np.abs(a.conj().T @ R @ a).item())
p_bf = np.array(p_bf)

# (2) 最尤法 (Deterministic ML - 簡易グリッドサーチ)
# 本来はJ次元の最適化が必要だが、ここでは空間スペクトル表示のため
# 投影行列を用いたコスト関数を使用
p_ml = []
for th in search_rad:
    a = get_a(th)
    Pa = a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T
    p_ml.append(np.trace(Pa @ R).real)
p_ml = np.array(p_ml)

# (3) WSF法 (Weighted Subspace Fitting)
# 固有値分解
eig_vals, eig_vecs = np.linalg.eigh(R)
idx = eig_vals.argsort()[::-1]
Es = eig_vecs[:, idx[:J]]  # 信号部分空間
Ls = np.diag(eig_vals[idx[:J]])
W_wsf = (Ls - sigma2 * np.eye(J)) ** 2 @ np.linalg.inv(Ls)


def wsf_cost(thetas):
    # thetas: 推定対象の角度(J個)
    A_theta = np.hstack([get_a(t) for t in thetas])
    Pa = A_theta @ np.linalg.inv(A_theta.conj().T @ A_theta) @ A_theta.conj().T
    return np.trace((np.eye(M) - Pa) @ Es @ W_wsf @ Es.conj().T).real


# WSFは多次元検索なので初期値から最適化
res = minimize(wsf_cost, x0=theta_true + 0.05, method="Nelder-Mead")
est_wsf = np.degrees(res.x)

# --- 3. 結果の可視化 ---
plt.figure(figsize=(10, 6))
plt.plot(search_theta, 10 * np.log10(p_bf / np.max(p_bf)), label="Beamforming")
plt.plot(search_theta, 10 * np.log10(p_ml / np.max(p_ml)), label="ML (Spatial Scan)")
for val in est_wsf:
    plt.axvline(
        val,
        color="r",
        linestyle="--",
        alpha=0.6,
        label="WSF Estimate" if val == est_wsf[0] else "",
    )
for val in angles_deg:
    plt.axvline(
        val, color="k", linestyle=":", label="True DOA" if val == angles_deg[0] else ""
    )

plt.title(f"DOA Estimation Comparison (M={M}, J={J}, SNR={snr_db}dB)")
plt.xlabel("Angle [deg]")
plt.ylabel("Relative Power [dB]")
plt.legend()
plt.grid(True)
plt.show()
