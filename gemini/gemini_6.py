import numpy as np
import matplotlib.pyplot as plt

# --- 1. シミュレーション条件の設定 ---
M = 3  # センサ数
J = 2  # 信号源数
theta_true = np.array([10, 30])  # 真の到来角
SNR = 20  # SNR (dB)
I = 500  # スナップショット数
wavelength = 1.0

# センサ位置 (x, y): 非対称・非回転対称（単位：波長）
sensor_pos = np.array([[0.0, 0.0], [0.55, 0.15], [0.2, 0.75]])


def get_steering_vector(theta_deg, pos):
    theta_rad = np.deg2rad(theta_deg)
    # 2次元平面での波数ベクトルとの内積
    k = (2 * np.pi / wavelength) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return np.exp(1j * pos @ k)


# --- 2. 信号と雑音の生成 ---
A_true = np.column_stack([get_steering_vector(t, sensor_pos) for t in theta_true])
# 信号 (複素包絡線)
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)
# 観測データ Y
Y = A_true @ S
# 複素白色ガウス雑音の付加
sigma2 = 10 ** (-SNR / 10)
noise = np.sqrt(sigma2 / 2) * (np.random.randn(M, I) + 1j * np.random.randn(M, I))
Y += noise

# 相関行列 R = (1/I) * YY*
R = (Y @ Y.conj().T) / I

# --- 3. スペクトラム計算 ---
search_angles = np.linspace(-90, 90, 1000)
p_bf = []
p_ml_pseudo = []

# トレースの基準値（全電力）
tr_R = np.real(np.trace(R))

for t in search_angles:
    h = get_steering_vector(t, sensor_pos).reshape(-1, 1)

    # --- ビームフォーミング法: P_BF = h* R h ---
    # (正規化のため h*h で割っています)
    bf_val = np.abs(h.conj().T @ R @ h)[0, 0] / np.abs(h.conj().T @ h)[0, 0]
    p_bf.append(bf_val)

    # --- 最尤法 (1次元擬似スペクトル): P_ML = Tr(P_H R) の原理を利用 ---
    # 単一の θ に対する射影行列 P_h = h(h*h)^-1 h*
    P_h = h @ np.linalg.inv(h.conj().T @ h) @ h.conj().T
    # 残差 Tr(R - P_h R) が小さいほど、その角度に信号が存在する可能性が高い
    # グラフで山を作るため、1 / 残差 を計算
    residual = tr_R - np.real(np.trace(P_h @ R))
    p_ml_pseudo.append(1.0 / residual)


# デシベル変換と正規化
def to_db(data):
    data = np.array(data)
    return 10 * np.log10(data / np.max(data))


p_bf_db = to_db(p_bf)
p_ml_db = to_db(p_ml_pseudo)

# --- 4. 結果のグラフ化 ---
plt.figure(figsize=(12, 6))
plt.plot(
    search_angles, p_bf_db, label="Beamforming: $h^* R h$", color="blue", alpha=0.8
)
plt.plot(
    search_angles,
    p_ml_db,
    label="ML (Pseudo): $1 / Tr(R - P_H R)$",
    color="red",
    linewidth=2,
)

# 真の方向を点線で表示
for t in theta_true:
    plt.axvline(t, color="black", linestyle="--", alpha=0.5)
plt.text(theta_true[0], 2, "True DOA", ha="center", fontsize=10)

plt.title(
    f"DOA Estimation: Beamforming vs ML\n(M={M}, SNR={SNR}dB, Non-symmetric Array)"
)
plt.xlabel("Angle $\\theta$ [degrees]")
plt.ylabel("Normalized Output [dB]")
plt.ylim([-15, 5])
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend()
plt.show()
