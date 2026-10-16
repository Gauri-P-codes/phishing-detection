"""Unit VI - Optimisation algorithms: Logistic Regression trained FROM SCRATCH (NumPy)
with Batch GD, SGD(+momentum), RMSprop and Adam. Loss = binary cross-entropy."""
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

class LogReg:
    def __init__(self, optimizer="adam", lr=0.01, epochs=30, batch=64, beta=0.9, beta2=0.999, eps=1e-8, seed=0):
        self.opt, self.lr, self.epochs, self.batch = optimizer, lr, epochs, batch
        self.b1, self.b2, self.eps, self.seed = beta, beta2, eps, seed
        self.history = []
    def _loss(self, X, y):
        p = np.clip(sigmoid(X @ self.w + self.b), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        P = np.zeros(d + 1); m = np.zeros(d + 1); v = np.zeros(d + 1); t = 0     # params / moments
        bs = n if self.opt == "gd" else self.batch
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, bs):
                j = idx[s:s + bs]; Xb, yb = X[j], y[j]; t += 1
                err = sigmoid(Xb @ self.w + self.b) - yb
                g = np.append(Xb.T @ err / len(j), err.mean())
                if self.opt in ("gd", "sgd"):
                    upd = self.lr * g
                elif self.opt == "momentum":
                    m = self.b1 * m + g; upd = self.lr * m
                elif self.opt == "rmsprop":
                    v = 0.9 * v + 0.1 * g ** 2; upd = self.lr * g / (np.sqrt(v) + self.eps)
                elif self.opt == "adam":
                    m = self.b1 * m + (1 - self.b1) * g; v = self.b2 * v + (1 - self.b2) * g ** 2
                    upd = self.lr * (m / (1 - self.b1 ** t)) / (np.sqrt(v / (1 - self.b2 ** t)) + self.eps)
                else: raise ValueError(self.opt)
                self.w -= upd[:-1]; self.b -= upd[-1]
            self.history.append(self._loss(X, y))
        return self
    def predict_proba(self, X): return sigmoid(X @ self.w + self.b)
    def predict(self, X): return (self.predict_proba(X) >= .5).astype(int)
