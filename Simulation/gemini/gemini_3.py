import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 設定値 ---
M = 3  # センサ数
J = 2  # 信号源数
theta_true = np.array([-10, 15])  # 真の到来方向 (度)
SNR = 10  # 信号対雑音比 (dB)
snapshots = 100  # スナップショット数 (I)
d_lambda = 0.5  # センサ間隔 / 波長

# 角度をラジアンに変換
theta_rad = np.deg2rad(theta_true)


# ステアリングベクトルの生成関数
def steering_vector(theta, M, d_lambda):
    return np.exp(-1j * 2 * np.pi * d_lambda * np.arange(M) * np.sin(theta))


# 1. 信号と雑音の生成
S = np.exp(1j * 2 * np.pi * np.random.rand(J, snapshots))  # 位相がランダムな信号
A = np.column_stack([steering_vector(t, M, d_lambda) for t in theta_rad])
X = A @ S  # 観測信号（雑音なし）

# 雑音の付加
sigma2 = 10 ** (-SNR / 10)
noise = np.sqrt(sigma2 / 2) * (
    np.random.randn(M, snapshots) + 1j * np.random.randn(M, snapshots)
)
X += noise

# 相関行列の計算
R = (X @ X.conj().T) / snapshots

# --- 推定アルゴリズム ---

# 角度サーチ範囲
search_angles = np.linspace(-90, 90, 500)
search_rad = np.deg2rad(search_angles)


# A. ビームフォーミング法 (Bartlett)
def compute_beamforming(R, search_rad):
    p_bf = []
    for t in search_rad:
        a = steering_vector(t, M, d_lambda)
        p_bf.append(np.abs(a.conj().T @ R @ a))
    return np.array(p_bf)


# B. 最尤法 (Deterministic ML) - 簡易的に2波同時探索
def ml_cost(thetas):
    A_theta = np.column_stack([steering_vector(t, M, d_lambda) for t in thetas])
    # 射影行列 P = A(A^H A)^-1 A^H
    P_A = A_theta @ np.linalg.inv(A_theta.conj().T @ A_theta) @ A_theta.conj().T
    return -np.real(np.trace(P_A @ R))  # 最小化のためマイナス


# C. WSF法 (Weighted Subspace Fitting)
# 信号部分空間の抽出
eig_vals, eig_vecs = np.linalg.eigh(R)
idx = eig_vals.argsort()[::-1]
Es = eig_vecs[:, idx[:J]]  # 信号部分空間
Ls = np.diag(eig_vals[idx[:J]])  # 信号固有値
sigma_est = np.mean(eig_vals[idx[J:]])  # 雑音電力推定
W_wsf = (Ls - sigma_est * np.eye(J)) ** 2 @ np.linalg.inv(Ls)


def wsf_cost(thetas):
    A_theta = np.column_stack([steering_vector(t, M, d_lambda) for t in thetas])
    P_A_perp = (
        np.eye(M)
        - A_theta @ np.linalg.inv(A_theta.conj().T @ A_theta) @ A_theta.conj().T
    )
    return np.real(np.trace(P_A_perp @ Es @ W_wsf @ Es.conj().T))


# --- 最適化実行 (MLとWSF) ---
# ※ 初期値は真値の近くに設定（局所解回避のため）
initial_guess = np.deg2rad(theta_true + np.random.randn(J) * 2)
res_ml = minimize(ml_cost, initial_guess, method="Nelder-Mead")
res_wsf = minimize(wsf_cost, initial_guess, method="Nelder-Mead")

# --- 結果の可視化 ---
p_bf = compute_beamforming(R, search_rad)
p_bf = 10 * np.log10(p_bf / np.max(p_bf))  # 正規化

plt.figure(figsize=(10, 6))
plt.plot(search_angles, p_bf, label="Beamforming (Spectrum)", color="gray", alpha=0.5)
plt.axvline(theta_true[0], color="red", linestyle="--", label="True Angles")
plt.axvline(theta_true[1], color="red", linestyle="--")

# MLとWSFは点としてプロット
plt.scatter(
    np.rad2deg(res_ml.x), [0, 0], color="blue", marker="x", s=100, label="ML Estimates"
)
plt.scatter(
    np.rad2deg(res_wsf.x),
    [-2, -2],
    color="green",
    marker="o",
    s=100,
    label="WSF Estimates",
)

plt.title(f"DOA Estimation Comparison (M={M}, J={J}, SNR={SNR}dB)")
plt.xlabel("Angle [deg]")
plt.ylabel("Normalized Power / Cost [dB]")
plt.legend()
plt.grid(True)
plt.show()

print(f"真の値: {theta_true}")
print(f"ML推定値: {np.rad2deg(res_ml.x)}")
print(f"WSF推定値: {np.rad2deg(res_wsf.x)}")
