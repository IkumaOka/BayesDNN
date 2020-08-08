# 「ベイズ深層学習」ベイズ線形回帰実装
# 本においてベクトルは縦ベクトル表記だから転置すると横ベクトルになる
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

x = np.linspace(-1.0, 1.0, 100)
len_data = len(x)
ndim = 1
sigma_w = 1.0
sigma_y = 1.0

# 式3.68
# wは基底関数φと同じ要素数のベクトル（式2.1より）
w = multivariate_normal(np.zeros(1), (sigma_w**2)*np.eye(1)).rvs(size=4)
# 特徴量関数φ
phi_x = np.array([x**3, x**2, x, np.ones(100)])

y_ = np.dot(w.T, phi_x)

# モデルf(x;w)のサンプル(式3.67)
y = multivariate_normal(np.dot(w.T, phi_x), np.ones(100)).rvs(size=1)

# plt.plot(x, y_)
# plt.show()

# 事後分布の計算
# 式3.72
inv_sigma_hat = (sigma_y**(-2)) * np.dot(phi_x, phi_x.T).sum(axis=0) + sigma_w**(-2)
sigma_hat = 1.0 / inv_sigma_hat
# y_nとphi(x_n)の積を計算
sum_multi_y_phi = np.array([y * phi_x[i] for i in range(4)])
# 式3.73
mu_hat = sigma_hat * (sigma_y**(-2)) * sum_multi_y_phi.sum(axis=1)

x_new = 0.8
phi_x_new = np.array([x_new**3, x_new**2, x_new, 1])
# 予測分布の計算
# 式3.76
mu_new = np.dot(mu_hat, phi_x_new)
sigma_new = sigma_y**2 + np.dot(phi_x_new.T, sigma_hat*phi_x_new)
print(sigma_new)









