import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 基本設定 ---
M = 3
J = 2
I = 100
true_angles_deg = np.array([10.0, 25.0])
theta_true = np.radians(true_angles_deg)
lam = 1.0
d = lam / 2
snr_range_db = np.arange(-5, 21, 5)
trials = 100


def get_a_vec(theta):
    return np.exp(-2j * np.pi * d / lam * np.arange(M) * np.sin(theta)).reshape(-1, 1)


def get_A_mat(thetas):
    return np.hstack([get_a_vec(t) for t in thetas])


# 各手法の推定関数
def estimate_doa(X, method="BF"):
    R = (X @ X.conj().T) / I
    search_deg = np.linspace(-90, 90, 181)

    if method == "BF":
        p = [
            np.abs(
                get_a_vec(np.radians(d)).conj().T @ R @ get_a_vec(np.radians(d))
            ).item()
            for d in search_deg
        ]
        # ピークを2つ取得
        idx = np.argsort(p)[-J:]
        return np.sort(search_deg[idx])

    elif method == "ML":

        def ml_obj(thetas):
            A = get_A_mat(thetas)
            Pa = A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T
            return -np.trace(Pa @ R).real

        res = minimize(ml_obj, x0=theta_true, method="Nelder-Mead")
        return np.sort(np.degrees(res.x))

    elif method == "WSF":
        eig_vals, eig_vecs = np.linalg.eigh(R)
        idx = eig_vals.argsort()[::-1]
        Es = eig_vecs[:, idx[:J]]
        # 雑音分散の推定（最小固有値の平均）
        sigma2_est = np.mean(eig_vals[idx[J:]])
        Ls = np.diag(eig_vals[idx[:J]])
        W = (Ls - sigma2_est * np.eye(J)) ** 2 @ np.linalg.inv(Ls)

        def wsf_obj(thetas):
            A = get_A_mat(thetas)
            Pa = A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T
            return np.trace((np.eye(M) - Pa) @ Es @ W @ Es.conj().T).real

        res = minimize(wsf_obj, x0=theta_true, method="Nelder-Mead")
        return np.sort(np.degrees(res.x))


# --- シミュレーション実行 ---
results = {m: [] for m in ["BF", "ML", "WSF"]}

for snr in snr_range_db:
    errors = {m: [] for m in ["BF", "ML", "WSF"]}
    sigma2 = 10 ** (-snr / 10)

    for _ in range(trials):
        # データ生成
        A = get_A_mat(theta_true)
        S = (np.random.randn(J, I) + 1j * np.random.randn(J, I)) / np.sqrt(2)
        N = np.sqrt(sigma2 / 2) * (np.random.randn(M, I) + 1j * np.random.randn(M, I))
        X = A @ S + N

        for m in ["BF", "ML", "WSF"]:
            est = estimate_doa(X, method=m)
            errors[m].append(np.sum((est - true_angles_deg) ** 2))

    for m in ["BF", "ML", "WSF"]:
        rmse = np.sqrt(np.sum(errors[m]) / (trials * J))
        results[m].append(rmse)

# --- グラフ出力 ---
plt.figure(figsize=(8, 5))
for m, style in zip(["BF", "ML", "WSF"], ["o-", "s--", "^-."]):
    plt.semilogy(snr_range_db, results[m], style, label=m)

plt.title(f"RMSE vs SNR ({trials} Trials, M={M}, J={J})")
plt.xlabel("SNR [dB]")
plt.ylabel("RMSE [degrees]")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()
