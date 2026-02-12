import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. シミュレーション条件の設定
# ==========================================
M = 3  # センサ数
J = 2  # 信号源数
I = 500  # スナップショット数
SNR = 20  # 信号対雑音比 (dB)
wavelength = 1.0

# 真の到来方向 (0-180度の範囲)
# BFでは分離が難しい「近接した角度」に設定
theta_true = np.array([75.0, 95.0])

# センサ位置 (x, y): 線対称・回転対称を避けた非一様配置
sensor_pos = np.array([[0.0, 0.0], [0.62, 0.15], [0.28, 0.82]])


def get_steering_vector(theta_deg, pos):
    """指定した角度のステアリングベクトルを返す"""
    theta_rad = np.deg2rad(theta_deg)
    # 2次元平面における波数ベクトルとの内積計算
    k_vec = (2 * np.pi / wavelength) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return np.exp(1j * pos @ k_vec)


# ==========================================
# 2. 観測データの生成
# ==========================================
# ステアリング行列 A
A_true = np.column_stack([get_steering_vector(t, sensor_pos) for t in theta_true])

# 信号 S (複素ガウス信号)
S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)

# 観測データ Y (信号 + 雑音)
noise_std = np.sqrt(10 ** (-SNR / 10) / 2)
N = noise_std * (np.random.randn(M, I) + 1j * np.random.randn(M, I))
Y = A_true @ S + N

# 相関行列 R = (1/I) * YY*
R = (Y @ Y.conj().T) / I

# ==========================================
# 3. 各手法の評価関数
# ==========================================
search_angles = np.linspace(0, 180, 360)

# --- A. ビームフォーミング法 (1次元スキャン) ---
p_bf = []
for t in search_angles:
    h = get_steering_vector(t, sensor_pos).reshape(-1, 1)
    # P_BF = h* R h
    val = np.abs(h.conj().T @ R @ h)[0, 0]
    p_bf.append(val)
p_bf = np.array(p_bf)

# --- B. 最尤法 (条件付き1次元スキャン) の修正 ---
p_ml_cond = []
h_fixed = get_steering_vector(theta_true[0], sensor_pos).reshape(-1, 1)

for t in search_angles:
    h_scan = get_steering_vector(t, sensor_pos).reshape(-1, 1)
    H = np.column_stack([h_fixed, h_scan])

    # inv の代わりに pinv (Moore-Penrose 擬似逆行列) を使用
    # これにより、t が 75度（固定角）に重なってもエラーにならず計算を継続できます
    try:
        # P_H = H @ (H*H)^-1 @ H*
        H_hermitian = H.conj().T
        # pinv は特異行列でも近似的な逆行列を返します
        P_H = H @ np.linalg.pinv(H_hermitian @ H) @ H_hermitian
        val = np.real(np.trace(P_H @ R))
    except Exception as e:
        val = 0
    p_ml_cond.append(val)
p_ml_cond = np.array(p_ml_cond)

# 2次元MLコスト関数の計算 (視覚化用)
scan_2d = np.linspace(60, 110, 50)
Z_ml = np.zeros((len(scan_2d), len(scan_2d)))
for i, t1 in enumerate(scan_2d):
    for j, t2 in enumerate(scan_2d):
        h1 = get_steering_vector(t1, sensor_pos).reshape(-1, 1)
        h2 = get_steering_vector(t2, sensor_pos).reshape(-1, 1)
        H = np.column_stack([h1, h2])
        H_hermitian = H.conj().T
        # pinv は特異行列でも近似的な逆行列を返します
        P_H = H @ np.linalg.pinv(H_hermitian @ H) @ H_hermitian
        Z_ml[i, j] = np.real(np.trace(P_H @ R))

# ==========================================
# 4. グラフ作成
# ==========================================
fig = plt.figure(figsize=(15, 6))

# --- 左：1次元比較グラフ ---
ax1 = fig.add_subplot(1, 2, 1)
# 正規化してデシベル表示
p_bf_db = 10 * np.log10(p_bf / np.max(p_bf))
p_ml_db = 10 * np.log10(p_ml_cond / np.max(p_ml_cond))

ax1.plot(
    search_angles, p_bf_db, label="Beamforming (Standard 1D)", color="blue", alpha=0.7
)
ax1.plot(
    search_angles,
    p_ml_db,
    label="ML",
    color="red",
    linewidth=2,
)

for t in theta_true:
    ax1.axvline(t, color="black", linestyle="--", alpha=0.5)
ax1.set_title("1D Spectrum: BF vs ML (Conditional)")
ax1.set_xlabel("Angle [deg]")
ax1.set_ylabel("Normalized Power [dB]")
ax1.set_ylim([-30, 2])
ax1.grid(True, linestyle=":")
ax1.legend()

# --- 右：2次元MLコスト関数ヒートマップ ---
ax2 = fig.add_subplot(1, 2, 2)
X_grid, Y_grid = np.meshgrid(scan_2d, scan_2d)
cp = ax2.contourf(X_grid, Y_grid, Z_ml, levels=30, cmap="hot")
ax2.scatter(
    theta_true[0], theta_true[1], color="cyan", marker="x", s=100, label="True Pair"
)
ax2.scatter(theta_true[1], theta_true[0], color="cyan", marker="x", s=100)
ax2.set_title("2D ML Cost Function: $Tr(P_H R)$")
ax2.set_xlabel("$\\theta_1$ [deg]")
ax2.set_ylabel("$\\theta_2$ [deg]")
plt.colorbar(cp, ax=ax2, label="Trace value")
ax2.legend()

plt.tight_layout()
plt.show()
