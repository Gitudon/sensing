import numpy as np
import matplotlib.pyplot as plt

# --- 1. センサと信号源の設定 ---
M = 3  # センサ数
J = 2  # 信号源数
theta_true = np.array([-20, 10])  # 真の到来角 (度)
SNR = 15  # SNR (dB)
I = 200  # スナップショット数
wavelength = 1.0

# センサ位置 (x, y): 線対称・回転対称を避ける
# 単位は波長。非一様な配置に設定
sensor_pos = np.array([[0.0, 0.0], [0.7, 0.1], [0.3, 0.8]])


def get_steering_vector(theta_deg, pos):
    theta_rad = np.deg2rad(theta_deg)
    # 2次元平面上での位相差計算
    # k = (2pi/lambda) * [cos(theta), sin(theta)]
    k = (2 * np.pi / wavelength) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    phases = pos @ k
    return np.exp(1j * phases)


# --- 2. 信号生成 ---
A = np.column_stack([get_steering_vector(t, sensor_pos) for t in theta_true])
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)
X = A @ S

# 雑音付加
sigma2 = 10 ** (-SNR / 10)
noise = np.sqrt(sigma2 / 2) * (np.random.randn(M, I) + 1j * np.random.randn(M, I))
X += noise

# 相関行列
R = (X @ X.conj().T) / I

# --- 3. スペクトラム計算用の関数 ---
search_angles = np.linspace(-90, 90, 400)


# A. Beamforming
def bf_spectrum(theta):
    a = get_steering_vector(theta, sensor_pos)
    return np.abs(a.conj().T @ R @ a)


# B. ML法 (Deterministic ML) の簡易スペクトラム
# 本来は多次元探索だが、可視化のため1波射影による評価関数を使用
def ml_pseudo_spectrum(theta):
    a = get_steering_vector(theta, sensor_pos).reshape(-1, 1)
    # 射影行列 P = a(a^H a)^-1 a^H
    P_a = a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T
    # 評価値: 1 / (Tr( (I-P)R )) -> 残差の逆数
    residual = np.real(np.trace((np.eye(M) - P_a) @ R))
    return 1.0 / residual


# C. WSF法 の簡易スペクトラム
eig_vals, eig_vecs = np.linalg.eigh(R)
idx = eig_vals.argsort()[::-1]
Es = eig_vecs[:, idx[:J]]
Ls = np.diag(eig_vals[idx[:J]])
sigma_est = np.mean(eig_vals[idx[J:]])
W_wsf = (Ls - sigma_est * np.eye(J)) ** 2 @ np.linalg.inv(Ls)


def wsf_pseudo_spectrum(theta):
    a = get_steering_vector(theta, sensor_pos).reshape(-1, 1)
    P_a_perp = np.eye(M) - a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T
    # 評価値: 1 / Tr( P_perp * Es * W * Es^H )
    val = np.real(np.trace(P_a_perp @ Es @ W_wsf @ Es.conj().T))
    return 1.0 / val


# --- 4. スペクトラムの計算と正規化 ---
p_bf = np.array([bf_spectrum(t) for t in search_angles])
p_ml = np.array([ml_pseudo_spectrum(t) for t in search_angles])
p_wsf = np.array([wsf_pseudo_spectrum(t) for t in search_angles])


def normalize(p):
    return 10 * np.log10(p / np.max(p))


p_bf_db = normalize(p_bf)
p_ml_db = normalize(p_ml)
p_wsf_db = normalize(p_wsf)

# --- 5. グラフ表示 ---
plt.figure(figsize=(12, 7))
plt.plot(search_angles, p_bf_db, label="Beamforming", color="blue", linewidth=1.5)
plt.plot(search_angles, p_ml_db, label="ML (Pseudo)", color="red", linewidth=2)
# plt.plot(
#     search_angles,
#     p_wsf_db,
#     label="WSF (Pseudo)",
#     color="green",
#     linestyle="--",
#     linewidth=2,
# )

for t in theta_true:
    plt.axvline(t, color="black", linestyle=":", alpha=0.6)
plt.text(theta_true[0], 2, "True DOA", ha="center")

plt.title(
    f"DOA Estimation Spectrum Comparison\n(Asymmetric Array, M={M}, SNR={SNR}dB, I={I})"
)
plt.xlabel("Angle [degrees]")
plt.ylabel("Normalized Power / Metric [dB]")
plt.ylim([-30, 5])
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.show()
