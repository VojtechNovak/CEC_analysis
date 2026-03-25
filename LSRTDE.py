import numpy as np

class LSRTDE_Optimizer:
    """
    L-SRTDE — CEC 2024 Winning Algorithm
    Ultra-accurate Python port of the original C++ code.
    Fully compatible with: run_lsrtde(f, dim, bounds, max_evals, seed)
    """
    def __init__(self, func, dim, bounds, max_evals, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.func = func
        self.dim = dim
        self.max_evals = max_evals

        # Robust bounds handling
        b = np.array(bounds, dtype=float)
        if b.ndim == 1 and b.size == 2:
            self.bounds = np.tile(b, (dim, 1))
        else:
            self.bounds = b  # assume (dim, 2)

        # Population settings
        self.base_pop = 20
        self.pop_size = self.base_pop * dim
        self.pop_size_max = self.pop_size
        self.pop_size_min = 4

        # State
        self.eval_count = 0
        self.best_fit = float('inf')
        self.best_ind = None

        # Cr memory
        self.memory_size = 5
        self.memory_pos = 0
        self.memory_cr = np.ones(self.memory_size)  # initialized to 1.0

        # Populations
        self.pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                     (2 * self.pop_size_max, self.dim))
        self.front = np.zeros((self.pop_size_max, self.dim))
        self.fitness = np.zeros(2 * self.pop_size_max)
        self.front_fitness = np.zeros(self.pop_size_max)

        self.trial = np.zeros(self.dim)
        self.success_cr = []
        self.success_delta = []

        # Initialize
        self._evaluate_initial_population()

    def _evaluate_initial_population(self):
        for i in range(self.pop_size):
            fit = self.func(self.pop[i])
            self.fitness[i] = fit
            self.eval_count += 1
            if fit < self.best_fit:
                self.best_fit = fit
                self.best_ind = self.pop[i].copy()

    def _build_front(self):
        """Copy current best individuals into the elite front (sorted)"""
        idx = np.argsort(self.fitness[:self.pop_size])
        active = slice(0, self.pop_size)
        self.front[active] = self.pop[idx]
        self.front_fitness[active] = self.fitness[idx]

    def optimize(self):
        self._build_front()

        success_rate = 0.5

        while self.eval_count < self.max_evals:
            # Adaptive scaling factor
            mean_f = 0.4 + np.tanh(success_rate * 5) * 0.25
            sigma_f = 0.02

            # Adaptive p-size for pbest selection
            p_size = max(2, int(self.pop_size * 0.7 * np.exp(-success_rate * 7)))

            # Exponential weights for r1 selection from elite front
            ranks = np.arange(self.pop_size)
            weights_front = np.exp(-3.0 * ranks / self.pop_size)
            weights_front /= weights_front.sum()

            # Reset success buffers
            self.success_cr.clear()
            self.success_delta.clear()
            success_count = 0
            front_write_pos = 0

            for _ in range(self.pop_size):
                if self.eval_count >= self.max_evals:
                    break

                # === Index selection ===
                x_idx = np.random.randint(self.pop_size)           # current individual (from front)
                mem_idx = np.random.randint(self.memory_size)

                # pbest from top p_size individuals in front (already sorted!)
                p_idx = np.random.randint(p_size)

                # r1 ranked selection from elite front
                r1 = np.random.choice(self.pop_size, p=weights_front)
                while r1 == p_idx:  # avoid same as pbest
                    r1 = np.random.choice(self.pop_size, p=weights_front)

                # r2 random from full population (not front!)
                r2 = np.random.randint(self.pop_size)
                while r2 in (p_idx, r1, x_idx):
                    r2 = np.random.randint(self.pop_size)

                # === Parameters ===
                F = np.random.normal(mean_f, sigma_f)
                while not (0 < F <= 1):
                    F = np.random.normal(mean_f, sigma_f)

                Cr = np.random.normal(self.memory_cr[mem_idx], 0.05)
                Cr = np.clip(Cr, 0.0, 1.0)

                # === Mutation: current-to-pbest/1 with external difference ===
                diff1 = self.front[p_idx] - self.front[x_idx]           # pbest - current
                diff2 = self.front[r1]   - self.pop[r2]                 # elite - random
                mutant = self.front[x_idx] + F * (diff1 + diff2)

                # === Crossover + boundary handling ===
                j_rand = np.random.randint(self.dim)
                actual_cr_count = 0
                for j in range(self.dim):
                    if np.random.random() < Cr or j == j_rand:
                        self.trial[j] = mutant[j]
                        actual_cr_count += 1
                    else:
                        self.trial[j] = self.front[x_idx][j]

                    # Boundary violation random reinitialization (as in original C++)
                    if self.trial[j] < self.bounds[j, 0] or self.trial[j] > self.bounds[j, 1]:
                        self.trial[j] = np.random.uniform(self.bounds[j, 0], self.bounds[j, 1])

                actual_cr = actual_cr_count / self.dim

                # === Evaluation ===
                trial_fit = self.func(self.trial)
                self.eval_count += 1

                if trial_fit < self.best_fit:
                    self.best_fit = trial_fit
                    self.best_ind = self.trial.copy()

                # === Selection ===
                if trial_fit <= self.front_fitness[x_idx]:
                    # Store offspring in auxiliary space
                    aux_idx = self.pop_size + success_count
                    self.pop[aux_idx] = self.trial.copy()
                    self.fitness[aux_idx] = trial_fit

                    # Update elite front using ring buffer
                    self.front[front_write_pos] = self.trial.copy()
                    self.front_fitness[front_write_pos] = trial_fit
                    front_write_pos = (front_write_pos + 1) % self.pop_size

                    self.success_cr.append(actual_cr)
                    self.success_delta.append(abs(self.front_fitness[x_idx] - trial_fit))
                    success_count += 1

            # === End of generation ===
            success_rate = success_count / self.pop_size

            # Update Cr memory using Lehmer mean on actual_cr
            if success_count > 0:
                w = np.array(self.success_delta)
                w /= w.sum()
                lehmer_cr = np.sum(w * np.array(self.success_cr)**2) / np.sum(w * np.array(self.success_cr) + 1e-16)
                self.memory_cr[self.memory_pos] = 0.5 * (lehmer_cr + self.memory_cr[self.memory_pos])
                self.memory_pos = (self.memory_pos + 1) % self.memory_size

            # Linear Population Size Reduction (LPSR)
            progress = self.eval_count / self.max_evals
            new_pop_size = int((self.pop_size_min - self.pop_size_max) * progress + self.pop_size_max)
            new_pop_size = max(self.pop_size_min, new_pop_size)

            # Merge survivors + successful offspring
            if success_count > 0:
                cand_pop = np.vstack((self.pop[:self.pop_size],
                                      self.pop[self.pop_size:self.pop_size + success_count]))
                cand_fit = np.hstack((self.fitness[:self.pop_size],
                                      self.fitness[self.pop_size:self.pop_size + success_count]))
                best_idx = np.argsort(cand_fit)[:new_pop_size]
                self.pop[:new_pop_size] = cand_pop[best_idx]
                self.fitness[:new_pop_size] = cand_fit[best_idx]
                self.pop_size = new_pop_size

            self._build_front()

        return self.best_ind, self.best_fit