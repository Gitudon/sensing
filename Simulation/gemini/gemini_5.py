import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定数と非対称センサ配置 ---
M = 3
J = 2
theta_true = np.array([-15, 20])  # 真の角度
SNR = 15
I = 300
wavelength = 1.0

# 非対称・非回転対称な配置 (x, y)
sensor_pos = np.array([[0.0, 0.0], [0.6, 0.1], [0.2, 0.7]])


def get_steering_vector(theta_deg, pos):
    theta_rad = np.deg2rad(theta_deg)
    # 波数ベクトル k = (2pi/lambda) * [cos(theta), sin(theta)]
    k = (2 * np.pi / wavelength) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return np.exp(1j * pos @ k)


# --- 2. 信号生成 ---
A_true = np.column_stack([get_steering_vector(t, sensor_pos) for t in theta_true])
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)
X = A_true @ S + (
    np.sqrt(10 ** (-SNR / 10) / 2)
    * (np.random.randn(M, I) + 1j * np.random.randn(M, I))
)
R = (X @ X.conj().T) / I

# --- 3. スペクトラム計算関数 ---
search_angles = np.linspace(-90, 90, 200)


def compute_spectra():
    bf_spec = []
    ml_pseudo = []
    wsf_pseudo = []

    # WSF用準備
    eig_vals, eig_vecs = np.linalg.eigh(R)
    idx = eig_vals.argsort()[::-1]
    Es = eig_vecs[:, idx[:J]]
    Ls = np.diag(eig_vals[idx[:J]])
    sigma_est = np.mean(eig_vals[idx[J:]])
    W_wsf = (Ls - sigma_est * np.eye(J)) ** 2 @ np.linalg.inv(Ls)

    for t in search_angles:
        a = get_steering_vector(t, sensor_pos).reshape(-1, 1)
        # 射影行列 P_H = a(a^H a)^-1 a^H
        P_H = a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T

        # Beamforming: a^H R a
        bf_spec.append(np.abs(a.conj().T @ R @ a)[0, 0])

        # ML: Tr(P_H R)
        # ※1波でのTr(P_H R)はBFを規格化したものに近いため、
        # 高分解能化を見せるため逆数残差（1/(Tr(R)-Tr(P_H R))）として計算
        res_ml = np.real(np.trace(R - P_H @ R))
        ml_pseudo.append(1.0 / res_ml)

        # WSF: Tr(P_H Es W Es^H) の逆数
        res_wsf = np.real(np.trace((np.eye(M) - P_H) @ Es @ W_wsf @ Es.conj().T))
        wsf_pseudo.append(1.0 / res_wsf)

    return np.array(bf_spec), np.array(ml_pseudo), np.array(wsf_pseudo)


# --- 4. 2次元MLコスト関数 (Tr(P_H R) の真の姿) ---
def ml_2d_cost(t1, t2):
    a1 = get_steering_vector(t1, sensor_pos).reshape(-1, 1)
    a2 = get_steering_vector(t2, sensor_pos).reshape(-1, 1)
    H = np.column_stack([a1, a2])
    P_H = H @ np.linalg.inv(H.conj().T @ H) @ H.conj().T
    return np.real(np.trace(P_H @ R))


# 描画用データ作成
p_bf, p_ml, p_wsf = compute_spectra()

# --- 5. グラフ化 ---
fig = plt.figure(figsize=(15, 5))

# (1) 1次元スペクトラム比較
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(search_angles, 10 * np.log10(p_bf / max(p_bf)), label="Beamforming")
ax1.plot(search_angles, 10 * np.log10(p_ml / max(p_ml)), label="ML (Pseudo)")
ax1.plot(search_angles, 10 * np.log10(p_wsf / max(p_wsf)), label="WSF (Pseudo)")
ax1.axvline(theta_true[0], color="k", linestyle=":", label="True DOA")
ax1.axvline(theta_true[1], color="k", linestyle=":")
ax1.set_title("1D Spectra Comparison")
ax1.set_xlabel("Angle [deg]")
ax1.set_ylabel("Power [dB]")
ax1.legend()
ax1.grid(True)

# (2) 2次元MLコスト関数ヒートマップ
ax2 = fig.add_subplot(1, 2, 2)
T1, T2 = np.meshgrid(search_angles, search_angles)
Z = np.zeros_like(T1)
for i in range(len(search_angles)):
    for j in range(len(search_angles)):
        Z[i, j] = ml_2d_cost(search_angles[i], search_angles[j])

cp = ax2.contourf(T1, T2, Z, levels=30, cmap="viridis")
plt.colorbar(cp, ax=ax2, label="Tr(P_H R)")
ax2.scatter(theta_true[0], theta_true[1], color="red", marker="x", label="True DOA")
ax2.scatter(theta_true[1], theta_true[0], color="red", marker="x")
ax2.set_title(r"2D ML Cost Function: $Tr(P_H R)$")
ax2.set_xlabel(r"$\theta_1$ [deg]")
ax2.set_ylabel(r"$\theta_2$ [deg]")
ax2.legend()

plt.tight_layout()
plt.show()
