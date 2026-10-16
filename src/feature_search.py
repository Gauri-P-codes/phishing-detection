"""Unit II + Self-study 'Search Space Management':
Feature-subset selection treated as a SEARCH PROBLEM.
 State  = binary mask over features      Move = flip one bit (neighbour)
 Score  = CV-F1 of a small decision tree - lambda * (#features)   (heuristic h(n))
Implements Hill Climbing, Simulated Annealing, Beam Search, Tabu Search.
Hill-climbing's disadvantage (local maxima / plateaus) is what Beam, SA and Tabu try to fix."""
import math, random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class FeatureSearch:
    def __init__(self, X, y, lam=0.004, seed=0):
        self.X, self.y, self.lam, self.rng = np.asarray(X), np.asarray(y), lam, random.Random(seed)
        self.n = self.X.shape[1]; self.cache = {}; self.evals = 0
    def score(self, mask):
        key = tuple(mask)
        if key not in self.cache:
            idx = [i for i, m in enumerate(mask) if m]
            if not idx: self.cache[key] = 0.0
            else:
                self.evals += 1
                f1 = cross_val_score(DecisionTreeClassifier(max_depth=6, random_state=0),
                                     self.X[:, idx], self.y, cv=3, scoring="f1").mean()
                self.cache[key] = f1 - self.lam * len(idx)
        return self.cache[key]
    def neighbours(self, mask):
        for i in range(self.n):
            m = list(mask); m[i] ^= 1; yield tuple(m)
    def random_state(self): return tuple(self.rng.randint(0, 1) for _ in range(self.n))

    def hill_climbing(self, start=None):
        cur = start or self.random_state(); best = self.score(cur)
        while True:
            nb = max(self.neighbours(cur), key=self.score)
            if self.score(nb) <= best: return cur, best          # stuck at local maximum
            cur, best = nb, self.score(nb)

    def simulated_annealing(self, T0=0.05, cool=0.95, steps=150):
        cur = self.random_state(); best, best_s = cur, self.score(cur); T = T0
        for _ in range(steps):
            i = self.rng.randrange(self.n); nxt = list(cur); nxt[i] ^= 1; nxt = tuple(nxt)
            d = self.score(nxt) - self.score(cur)
            if d > 0 or self.rng.random() < math.exp(d / max(T, 1e-9)): cur = nxt     # accept worse sometimes
            if self.score(cur) > best_s: best, best_s = cur, self.score(cur)
            T *= cool
        return best, best_s

    def beam_search(self, width=3, depth=10):
        beam = [self.random_state() for _ in range(width)]; best = max(beam, key=self.score)
        for _ in range(depth):
            cand = {n for m in beam for n in self.neighbours(m)}
            beam = sorted(cand, key=self.score, reverse=True)[:width]
            if self.score(beam[0]) > self.score(best): best = beam[0]
        return best, self.score(best)

    def tabu_search(self, tenure=7, iters=30):
        cur = self.random_state(); best, best_s = cur, self.score(cur); tabu = []
        for _ in range(iters):
            moves = [(i, n) for i, n in enumerate(self.neighbours(cur)) if i not in tabu]
            i, cur = max(moves, key=lambda t: self.score(t[1]))      # best non-tabu move, even if worse
            tabu.append(i); tabu = tabu[-tenure:]
            if self.score(cur) > best_s: best, best_s = cur, self.score(cur)
        return best, best_s
