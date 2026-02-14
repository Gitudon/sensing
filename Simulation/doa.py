import numpy as np
import matplotlib.pyplot as plt


# 1. 課題の条件設定

# 条件1: センサ数と信号源数
# センサ数
M = 3
# 信号源数
J = 2

# 条件2, 3: センサと信号源の位置
# 波長λ
LAMBDA = 2 ** (1.5)
# 基準となる素子間隔
D = LAMBDA / 2
# センサ位置 (3次元空間内の同一平面 (z=0))
# 線対称・回転対称にならない位置に配置
SENSOR_POS = np.array([[0.0, 0.0], [1.2 * D, 0.3 * D], [0.5 * D, 1.5 * D]])

# 条件4: 信号の到来方向θ
TRUE_THETAS = np.array([30, 45])

# 条件6: スナップショット数と雑音レベル
# スナップショット数
I = 1000
# 雑音レベル (dB)
SNR = 20

# 条件5: 雑音は全センサについて独立同分布なる白色複素ガウス雑音とする
sigma = 10 ** (-SNR / 10)
N = np.sqrt(sigma / 2) * (np.random.randn(M, I) + 1j * np.random.randn(M, I))


# 2. シミュレーションに使用する関数の定義


# ステアリングベクトルの計算
def steering_vector(theta, pos):
    theta_rad = np.radians(theta)
    # 平面波を仮定、進行方向ベクトルとの内積を取る
    k = (2 * np.pi / LAMBDA) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    phases = np.dot(pos, k)
    return np.exp(1j * phases)


# ビームフォーミング法
def beam_forming_method(theta):
    a = steering_vector(theta, SENSOR_POS)
    return np.real(a.conj().T @ R @ a)


# 最尤法
def maximum_likelihood_method(t1, t2):
    if t1 == t2:
        return 0
    A_theta = np.column_stack(
        [steering_vector(t1, SENSOR_POS), steering_vector(t2, SENSOR_POS)]
    )
    # 投影行列 P_A = A * (A^H * A)^-1 * A^H
    Proj = A_theta @ np.linalg.inv(A_theta.conj().T @ A_theta) @ A_theta.conj().T
    return np.real(np.trace(Proj @ R))


# 3. シミュレーションに用いるデータの生成

# 行列H (M x J)
H = np.column_stack([steering_vector(t, SENSOR_POS) for t in TRUE_THETAS])
# 信号源 S (J x I): 複素ガウス信号
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)
# 観測信号 Y
Y = np.dot(H, S) + N
# 相関行列 R
R = np.dot(Y, Y.conj().T) / I


# 4. ビームフォーミング法によるDOA推定

# 探索範囲(0°から180°まで0.5°刻み)
search_thetas = np.linspace(0, 180, 360 + 1)
# ビームフォーミング法の出力
p_bf = [beam_forming_method(t) for t in search_thetas]
# 正規化
p_bf = np.array(p_bf) / np.max(p_bf)
# p_bfが最大となるθを推定値とする
est_bf_idx = np.argmax(p_bf)
est_bf_theta = search_thetas[est_bf_idx]


# 2次元探索
ml_search = np.linspace(0, 180, 181)
p_ml = np.zeros((len(ml_search), len(ml_search)))

for i, t1 in enumerate(ml_search):
    for j, t2 in enumerate(ml_search):
        p_ml[i, j] = maximum_likelihood_method(t1, t2)

# MLの最大値
idx = np.unravel_index(np.argmax(p_ml), p_ml.shape)
est_ml_thetas = (ml_search[idx[0]], ml_search[idx[1]])

# グラフの作成
fig = plt.figure(figsize=(12, 5))

# BF法のグラフ
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(search_thetas, p_bf, label="P_BF(θ)")
ax1.axvline(TRUE_THETAS[0], color="red", linestyle="--", label=f"True θ1")
ax1.axvline(TRUE_THETAS[1], color="red", linestyle="--", label=f"True θ2")

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
    TRUE_THETAS[1],
    TRUE_THETAS[0],
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

# レイアウト調整・グラフの表示
plt.tight_layout()
plt.show()
