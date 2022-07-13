from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from numpy.linalg import inv, slogdet
from scipy.optimize import minimize


class GPR(GaussianProcessRegressor):
    def __init__(self, noise=0.0):
        super().__init__(kernel=self.kernel(), alpha=noise)
        self.name = 'GPR'
        self.x_train = None
        self.length_scale = None
        self.constant_value = None
        self.inv_K = None
        self.noise = noise
    
    def kernel(self):
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0, 1e2))
        return kernel

    def fit(self, x, y):
        self.x_train = x
        super().fit(x, y)
        self.save_params()
    
    def save_params(self):
        params = self.kernel_.get_params()
        self.constant_value = params['k1__constant_value']
        self.length_scale = params['k2__length_scale']
        self.alpha = self.alpha_.ravel()
        K = self.kernel_(self.x_train, self.x_train) + np.eye(self.x_train.shape[0]) * self.noise
        self.inv_K = inv(K)
    
    def predict(self, x, return_std=False):
        if return_std:
            return super().predict(x, return_std=True)
        else:
            return super().predict(x, return_std=False)

    def formulation(self, x, return_std=False):
        n = self.x_train.shape[0]
        m = self.x_train.shape[1]
        sq_exp = np.exp(
            -sum(0.5 / self.length_scale ** 2 * (x[:, j] - self.x_train[:, j]) ** 2 for j in range(m))
            )
        k_s = self.constant_value * sq_exp
        pred = sum(k_s[i] * self.alpha[i] for i in range(n))
        if return_std:
            vMv = sum(k_s[i] * sum(self.inv_K[i, j] * k_s[j] for j in range(n)) for i in range(n))
            var = self.constant_value + self.noise - vMv
            std = np.sqrt(var)
            return pred, std
        else:
            return pred


class GPC:
    def __init__(self):
        self.name = 'GPC'
        self.x_train = None
        self.t_train = None
        self.l = None
        self.kappa = None
        self.delta = None
        self.invP = None

    def kernel(self, x1, x2):
        l = 0.5 / self.l ** 2
        sq_dist = np.sum(l * x1 ** 2, 1).reshape(-1, 1) + np.sum(l * x2 ** 2, 1) - 2 * np.dot(x1, l.T * x2.T)
        sq_exp = self.kappa * np.exp(-sq_dist)
        return sq_exp

    def fit(self, x, t):
        self.x_train = x
        self.t_train = t
        self.calculate_params(x, t)

    def predict(self, x, return_std=False):
        a = self.posterior_mode(self.x_train, self.t_train)
        k_s = self.kernel(self.x_train, x)
        mu = k_s.T.dot(self.t_train - self.sigmoid(a))
        s2 = self.kappa - k_s.T.dot(self.invP).dot(k_s)
        s2 = np.diag(s2).clip(min=0).reshape(-1, 1)
        beta = np.sqrt(1.0 + 3.1416 / 8 * s2)
        prediction = self.sigmoid(mu / beta)
        if return_std:
            return prediction, np.sqrt(s2)   # uncertainty is not 1.96*s due to not normal distribution here.
        else:
            return prediction
    
    def formulation(self, x):
        l = 0.5 / self.l ** 2
        mu = self.kappa * sum(self.delta[j] * np.exp(-sum(l * (x[i] - self.x_train[j, i]) ** 2 for i in range(self.x_train.shape[1]))) for j in range(self.x_train.shape[0]))
        var = self.kappa * (1 - sum(np.exp(-sum(l * (x[i] - self.x_train[j, i]) ** 2 for i in range(self.x_train.shape[1]))) * self.kappa * sum(np.exp(-sum(l * (x[j] - self.x_train[k, j]) ** 2 for j in range(self.x_train.shape[1]))) * self.invP[j, k] for k in range(self.x_train.shape[0])) for j in range(self.x_train.shape[0])))
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        pred = 1 / (1 + np.exp(- mu / beta))
        return pred
    
    def posterior_mode(self, x, t, max_iter=10, tol=1e-9):
        K = self.kernel(self.x_train, self.x_train)
        a = np.zeros_like(t)
        I = np.eye(x.shape[0])
        for i in range(max_iter):
            W = self.sigmoid(a) * (1 - self.sigmoid(a))
            W = np.diag(W.ravel())
            invQ = inv(I + W @ K)
            a_new = (K @ invQ).dot(t - self.sigmoid(a) + W.dot(a))
            a_diff = np.abs(a_new - a)
            a = a_new
            if not np.any(a_diff > tol):
                break
        return a
    
    def calculate_params(self, x, t):
        param_init = np.array([1.0, 1.0])  # l is a scalar parameter here
        param_bounds = [(1e-6, None), (1e-6, None)]
        params = minimize(self.opt_fun(x, t), param_init, bounds=param_bounds, method='L-BFGS-B', options={'iprint': -1})
        self.l = params.x[0]
        self.kappa = params.x[1]
        K = self.kernel(x, x)
        a = self.posterior_mode(x, t)
        W = self.sigmoid(a) * (1 - self.sigmoid(a))
        W = np.diag(W.ravel())
        invW = inv(W)
        P = invW + K
        self.invP = inv(P)
        delta = t - self.sigmoid(a)
        self.delta = delta.ravel()

    def opt_fun(self, x, t):
        def f(theta):
            I = np.eye(x.shape[0])
            self.l = theta[0]
            self.kappa = theta[1]
            K = self.kernel(x, x) + 1e-5 * I
            invK = inv(K)
            a = self.posterior_mode(x, t)
            W = self.sigmoid(a) * (1 - self.sigmoid(a))
            W = np.diag(W.ravel())
            ll = -0.5 * (a.T.dot(invK).dot(a) + slogdet(K)[1] + slogdet(W+invK)[1]) + np.inner(t.T, a.T) - np.sum(np.log(1.0 + np.exp(a)))
            return -ll.ravel()
        return f

    @staticmethod
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))


def main():
    x = np.random.rand(20, 2)
    y = np.random.rand(20, 1) * 10
    x_new = np.array([[0.420, 0.069]])

    gpr = GPR()
    gpr.fit(x, y)
    pred, s2 = gpr.predict(x_new, return_std=True)
    print(pred, s2)
    val, u = gpr.formulation(x_new, return_std=True)
    print(val, u)

    from sklearn.datasets import make_moons
    x, y = make_moons(noise=0.5, n_samples=100)

    gpc = GPC()
    gpc.fit(x, y)
    pred = gpc.predict(x_new)
    print(pred)
    val = gpc.formulation(x_new[0])
    print(val)


if __name__ == '__main__':
    main()
