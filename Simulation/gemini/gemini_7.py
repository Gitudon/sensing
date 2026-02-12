import numpy as np
import matplotlib.pyplot as plt

# --- 1. 設定 ---
M = 3  # センサ数
J = 2  # 信号源数
theta_true = np.array([45, 135.0])  # 真の角度 (0-180度の範囲)
SNR = 20
I = 4
wavelength = 1.0

# 非対称・非回転対称なセンサ位置 (x, y)
sensor_pos = np.array([[0.0, 0.0], [0.65, 0.12], [0.25, 0.85]])


def get_steering_vector(theta_deg, pos):
    theta_rad = np.deg2rad(theta_deg)
    # 2次元平面: 波数は x*cos(theta) + y*sin(theta)
    k_vec = (2 * np.pi / wavelength) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return np.exp(1j * pos @ k_vec)


# --- 2. 信号生成 ---
A_true = np.column_stack([get_steering_vector(t, sensor_pos) for t in theta_true])
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)
Y = A_true @ S
noise = np.sqrt(10 ** (-SNR / 10) / 2) * (
    np.random.randn(M, I) + 1j * np.random.randn(M, I)
)
Y += noise

# 相関行列 R = (1/I) * YY*
R = (Y @ Y.conj().T) / I

# --- 3. 0-180度スキャン ---
search_angles = np.linspace(0, 180, 500)
p_bf = []
p_ml = []

for t in search_angles:
    h = get_steering_vector(t, sensor_pos).reshape(-1, 1)

    # --- ビームフォーミング法: h* R h (R = YY*/I) ---
    val_bf = np.abs(h.conj().T @ R @ h)[0, 0]
    p_bf.append(val_bf)

    # --- 最尤法: Tr(P_H R) ---
    # P_H = h(h*h)^-1 h*
    P_H = h @ np.linalg.inv(h.conj().T @ h) @ h.conj().T
    val_ml = np.real(np.trace(P_H @ R))
    p_ml.append(val_ml)

# デシベル変換（比較のため最大値で正規化）
p_bf_db = 10 * np.log10(np.array(p_bf) / np.max(p_bf))
p_ml_db = 10 * np.log10(np.array(p_ml) / np.max(p_ml))

# --- 4. 可視化 ---
plt.figure(figsize=(12, 6))
plt.plot(
    search_angles, p_bf_db, label=r"$P_{BF}(\theta) = h^* R h$", color="blue", alpha=0.7
)
plt.plot(
    search_angles,
    p_ml_db,
    label=r"$P_{ML}(\theta) = \text{Tr}(P_H R)$",
    color="red",
    linewidth=2,
)

for t in theta_true:
    plt.axvline(t, color="black", linestyle="--", alpha=0.6)
plt.text(theta_true[0], 1, "True DOA", ha="center")

plt.title(
    f"DOA Estimation Spectrum (0-180°)\nNon-symmetric Array (M={M}, J={J}, SNR={SNR}dB)"
)
plt.xlabel("Angle $\\theta$ [degrees]")
plt.ylabel("Normalized Output [dB]")
plt.xlim([0, 180])
plt.ylim([-25, 2])
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.show()
