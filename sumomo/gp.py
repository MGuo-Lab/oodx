from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from numpy.linalg import inv, slogdet
from scipy.optimize import minimize


class GPR(GaussianProcessRegressor):
    def __init__(self, noise=0.0):
        super().__init__(kernel=self._kernel(), alpha=noise)
        self.name = 'GPR'
        self.x_train = None
        self.length_scale = None
        self.constant_value = None
        self.inv_K = None
        self.noise = noise
    
    def _kernel(self):
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0, 1e2))
        return kernel

    def fit(self, x, y):
        self.x_train = x
        super().fit(x, y)
        self._save_params()
    
    def _save_params(self):
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
        self.sigma_f = None
        self.delta = None
        self.invP = None

    def _kernel(self, x1, x2):
        sq_dist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        sq_exp = self.sigma_f ** 2 * np.exp( - 0.5 / self.l ** 2 * sq_dist )
        return sq_exp

    def fit(self, x, t):
        self.x_train = x
        self.t_train = t
        self._calculate_params()

    def predict(self, x, return_std=False):
        a = self._posterior_mode()
        k_s = self._kernel(self.x_train, x)
        mu = k_s.T.dot(self.t_train - self._sigmoid(a))
        var = self.sigma_f ** 2 - k_s.T.dot(self.invP).dot(k_s)
        var = np.diag(var).clip(min=0).reshape(-1, 1)
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        prediction = self._sigmoid(mu / beta)
        if return_std:
            return prediction, np.sqrt(var)
        else:
            return prediction
    
    def formulation(self, x):
        n = self.x_train.shape[0]
        m = self.x_train.shape[1]
        sq_exp = np.exp(
            -sum(0.5 / self.l ** 2 * (x[j] - self.x_train[:, j]) ** 2 for j in range(m))
            )
        mu = self.sigma_f ** 2 * sum(self.delta[i] * sq_exp[i] for i in range(n))
        var = self.sigma_f ** 2 * (1 - sum(
            sq_exp[i] * self.sigma_f ** 2 * sum(
                sq_exp[i_] * self.invP[i, i_] for i_ in range(n)) for i in range(n)))
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        prediction = 1 / (1 + np.exp(- mu / beta))
        return prediction
    
    def _posterior_mode(self, max_iter=10, tol=1e-9):
        K = self._kernel(self.x_train, self.x_train)
        a = np.zeros_like(self.t_train)
        I = np.eye(self.x_train.shape[0])
        for i in range(max_iter):
            W = self._sigmoid(a) * (1 - self._sigmoid(a))
            W = np.diag(W.ravel())
            invQ = inv(I + W @ K)
            a_new = (K @ invQ).dot(self.t_train - self._sigmoid(a) + W.dot(a))
            a_diff = np.abs(a_new - a)
            a = a_new
            if not np.any(a_diff > tol):
                break
        return a
    
    def _calculate_params(self):
        
        params = minimize(
            fun=self._opt_fun, 
            x0=[1.0, 1.0], 
            bounds=[(1e-6, None), (1e-6, None)], 
            method='L-BFGS-B', 
            options={'iprint': -1})
        
        self.l = params.x[0]
        self.sigma_f = params.x[1]
        a = self._posterior_mode()
        W = self._sigmoid(a) * (1 - self._sigmoid(a))
        W = np.diag(W.ravel())
        K = self._kernel(self.x_train, self.x_train)
        P = inv(W) + K
        self.invP = inv(P)
        self.delta = self.t_train - self._sigmoid(a)
    
    def _opt_fun(self, theta):
        I = np.eye(self.x_train.shape[0])
        self.l = theta[0]
        self.sigma_f = theta[1]
        K = self._kernel(self.x_train, self.x_train) + 1e-5 * I
        inv_K = inv(K)
        a = self._posterior_mode()
        W = self._sigmoid(a) * (1 - self._sigmoid(a))
        W = np.diag(W.ravel())
        ll = self.t_train.T.dot(a) - np.sum(np.log(1.0 + np.exp(a))) - 0.5 * (
            a.T.dot(inv_K).dot(a) + 
            slogdet(K)[1] + 
            slogdet(W+inv_K)[1])
        return -ll

    @staticmethod
    def _sigmoid(a):
        return 1 / (1 + np.exp(-a))


def main():
    x = np.random.rand(20, 2)
    y = np.random.rand(20, 1) * 10
    x_new = np.array([[0.420, 0.069]])

    gpr = GPR()
    # with np.errstate(divide='ignore'):
    #     gpr.fit(x, y)
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
