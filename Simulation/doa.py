import numpy as np
import matplotlib.pyplot as plt

# --- 1. 設定パラメタ ---
M = 3  # センサ数
J = 2  # 信号源数
I = 1000  # スナップショット数
SNR = 20  # 雑音レベル (dB)
lambda_ = 2 ** (1.5)  # 波長
d = lambda_ / 2  # 基準となる素子間隔

# センサ位置 (3次元空間内の同一平面: z=0)
# 非対称・非回転対称にするための設定
sensor_pos = np.array([[0.0, 0.0], [1.2 * d, 0.3 * d], [0.5 * d, 1.5 * d]])

# 信号源の真の角度 (度)
true_thetas = np.array([30, 45])


# --- 2. 信号生成 ---
def steering_vector(theta, pos):
    """ステアリングベクトルの計算"""
    theta_rad = np.radians(theta)
    # 平面波の仮定: 進行方向ベクトルとの内積
    # k = 2 * pi / lambda * [cos(theta), sin(theta)]
    k = (2 * np.pi / lambda_) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    phases = np.dot(pos, k)
    return np.exp(1j * phases)


# Aマトリックス (M x J)
A = np.column_stack([steering_vector(t, sensor_pos) for t in true_thetas])

# 信号源 S (J x I): 複素ガウス信号
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)

# 雑音 N (M x I)
sigma2 = 10 ** (-SNR / 10)
N = np.sqrt(sigma2 / 2) * (np.random.randn(M, I) + 1j * np.random.randn(M, I))

# 観測信号 X
X = np.dot(A, S) + N

# 相関行列 Rxx
Rxx = np.dot(X, X.conj().T) / I

# --- 3. 定位アルゴリズム ---

# 探索範囲
search_thetas = np.linspace(0, 180, 181)


# (a) ビームフォーミング法 (BF)
def compute_bf(theta):
    a = steering_vector(theta, sensor_pos)
    return np.real(a.conj().T @ Rxx @ a)


p_bf = [compute_bf(t) for t in search_thetas]
p_bf = np.array(p_bf) / np.max(p_bf)  # 正規化
est_bf_idx = np.argmax(p_bf)
est_bf_theta = search_thetas[est_bf_idx]


# (b) 最尤法 (ML) - 決定論的ML
def compute_ml(t1, t2):
    if t1 == t2:
        return 0
    A_theta = np.column_stack(
        [steering_vector(t1, sensor_pos), steering_vector(t2, sensor_pos)]
    )
    # 投影行列 P_A = A * (A^H * A)^-1 * A^H
    Proj = A_theta @ np.linalg.inv(A_theta.conj().T @ A_theta) @ A_theta.conj().T
    return np.real(np.trace(Proj @ Rxx))


# 2次元探索
ml_search = np.linspace(0, 180, 181)
p_ml = np.zeros((len(ml_search), len(ml_search)))

for i, t1 in enumerate(ml_search):
    for j, t2 in enumerate(ml_search):
        p_ml[i, j] = compute_ml(t1, t2)

# MLの最大値
idx = np.unravel_index(np.argmax(p_ml), p_ml.shape)
est_ml_thetas = (ml_search[idx[0]], ml_search[idx[1]])

# --- 4. 可視化 ---
fig = plt.figure(figsize=(12, 5))

# BF法のグラフ
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(search_thetas, p_bf, label="P_BF(θ)")
ax1.axvline(true_thetas[0], color="red", linestyle="--", label=f"True θ1")
ax1.axvline(true_thetas[1], color="red", linestyle="--", label=f"True θ2")

ax1.set_title("Beamforming Method")
ax1.set_xlabel("θ [deg]")
ax1.set_ylabel("P_BF (normalized)")
ax1.legend()
ax1.grid(True)

# ML法のヒートマップ
custom_cmap = plt.get_cmap("jet", 256)
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(
    p_ml,
    extent=[ml_search[0], ml_search[-1], ml_search[0], ml_search[-1]],
    origin="lower",
    aspect="auto",
    cmap=custom_cmap,
)
ax2.scatter(
    true_thetas[1],
    true_thetas[0],
    color="red",
    marker="x",
    s=100,
    label="True (θ1, θ2)",
)
ax2.scatter(
    est_ml_thetas[1],
    est_ml_thetas[0],
    color="white",
    marker="o",
    edgecolors="black",
    label=f"Est: ({est_ml_thetas[0]:.1f}, {est_ml_thetas[1]:.1f})",
)
ax2.scatter(
    est_ml_thetas[0],
    est_ml_thetas[1],
    color="white",
    marker="o",
    edgecolors="black",
    label=f"Est: ({est_ml_thetas[1]:.1f}, {est_ml_thetas[0]:.1f})",
)
ax2.set_title("Maximum Likelihood Method")
ax2.set_xlabel("θ2 [deg]")
ax2.set_ylabel("θ1 [deg]")
plt.colorbar(im, ax=ax2, label="P_ML")
ax2.legend()

plt.tight_layout()
plt.show()
