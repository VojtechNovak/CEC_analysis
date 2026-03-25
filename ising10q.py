import os
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from CMAES import CMAESOptimizer
# Suppress mathematically harmless divide-by-zero warnings from the optimizers' adaptive memory updates
warnings.filterwarnings("ignore", category=RuntimeWarning)

class VQAIsingLogger:
    """
    Generalized Batched Numpy Simulator for N-Qubit Hardware-Efficient VQA.
    1D Ising Model (no external field): H = -sum_{i} Z_i Z_{i+1}
    Ansatz: Ry layers interleaved with linear CNOT entangling layers.
    """
    def __init__(self, dim, noise_std, rng=None, n_qubits=10):
        self.n_qubits = n_qubits
        self.layers = max(1, dim // self.n_qubits - 1)
        self.noise_std = noise_std
        self.rng = rng if rng is not None else np.random.RandomState()
        self.eval_count = 0
        self.best = np.inf
        self.history = []

        # Ground state energy is -(N-1)
        self.H_diag = np.zeros(2**self.n_qubits)
        for i in range(2**self.n_qubits):
            bits = [(i >> (self.n_qubits - 1 - j)) & 1 for j in range(self.n_qubits)]
            energy = 0
            for q in range(self.n_qubits - 1):
                z1 = 1 if bits[q] == 0 else -1
                z2 = 1 if bits[q+1] == 0 else -1
                energy += -(z1 * z2) 
            self.H_diag[i] = energy

    def _apply_ry(self, state, thetas, target):
        B = state.shape[0]
        c = np.cos(thetas / 2.0)
        s = np.sin(thetas / 2.0)
        U = np.zeros((B, 2, 2))
        U[:, 0, 0] = c; U[:, 0, 1] = -s
        U[:, 1, 0] = s; U[:, 1, 1] = c
        
        # Dynamically generate einsum string based on N qubits
        letters = [chr(97 + i) for i in range(self.n_qubits)]
        state_str = "Z" + "".join(letters) # 'Z' represents the batch dimension
        out_letters = letters.copy()
        out_letters[target] = 'X'          # 'X' represents the transformed target
        out_str = "Z" + "".join(out_letters)
        
        einsum_str = f"ZX{letters[target]},{state_str}->{out_str}"
        return np.einsum(einsum_str, U, state)

    def _apply_cnot(self, state, control, target):
        # Dynamically slice N-dimensional batch array
        s0 = [slice(None)] * (self.n_qubits + 1)
        s0[control + 1] = 1; s0[target + 1] = 0
        s1 = [slice(None)] * (self.n_qubits + 1)
        s1[control + 1] = 1; s1[target + 1] = 1
        
        temp = state[tuple(s0)].copy()
        state[tuple(s0)] = state[tuple(s1)]
        state[tuple(s1)] = temp
        return state

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        is_1d = (x.ndim == 1)
        if is_1d: x = x.reshape(1, -1)
        
        B = x.shape[0]
        # Initialize state |0...0> for N qubits
        state = np.zeros((B,) + (2,) * self.n_qubits)
        idx = [slice(None)] + [0] * self.n_qubits
        state[tuple(idx)] = 1.0 
        
        param_idx = 0
        for _ in range(self.layers):
            for q in range(self.n_qubits):
                state = self._apply_ry(state, x[:, param_idx], q)
                param_idx += 1
            for q in range(self.n_qubits - 1):
                state = self._apply_cnot(state, q, q + 1)
                
        for q in range(self.n_qubits):
            state = self._apply_ry(state, x[:, param_idx], q)
            param_idx += 1
            
        probs = state.reshape(B, 2**self.n_qubits) ** 2
        exact_values = np.sum(probs * self.H_diag, axis=1)
        
        if self.noise_std > 0.0:
            noise = self.rng.normal(0.0, self.noise_std, size=exact_values.shape)
            fx = exact_values + noise
        else:
            fx = exact_values
        
        self.eval_count += B
        batch_best = np.min(fx)
        if batch_best < self.best: self.best = batch_best
        self.history.extend([self.best] * B)
        
        return fx[0] if is_1d else fx
    
# ============================================================
# Vectorized Objective Logger
# ============================================================
class NoisyQuadraticLogger:
    """
    f(x) = Shifted, Rotated Rastrigin's Function + Normal(0, noise_std)
    *Supports both 1D (single ind) and 2D (population) inputs!*
    """
    def __init__(self, dim, noise_std, rng=None):
        self.dim = dim
        self.noise_std = noise_std
        self.history = []      
        self.eval_count = 0
        self.best = np.inf
        self.rng = rng if rng is not None else np.random.RandomState()

        self.shift = np.zeros(self.dim)
        H = self.rng.standard_normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(H)
        self.M = Q
        self.bias = 0.0

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.reshape(1, -1)

        x_shifted = x - self.shift
        z = x_shifted.dot(self.M.T) 
        
        terms = (z**2 - 10 * np.cos(2 * np.pi * z) + 10)
        base = np.sum(terms, axis=1) + self.bias
        noise = self.rng.normal(0.0, self.noise_std, size=base.shape)
        fx = base + noise
        
        self.eval_count += x.shape[0]
        current_batch_best = np.min(fx)
        if current_batch_best < self.best:
            self.best = current_batch_best
            
        self.history.extend([self.best] * x.shape[0])

        return fx[0] if is_1d else fx

# ============================================================
# Optimizer Wrappers
# ============================================================
def run_spsa(f, dim, bounds, max_evals, seed, a=0.1, c=0.1, alpha=0.602, gamma=0.101):
    rng = np.random.RandomState(seed)
    x = rng.uniform(bounds[0], bounds[1], size=dim)
    k = 0
    while f.eval_count + 2 <= max_evals:
        ak = a / ((k + 1) ** alpha)
        ck = c / ((k + 1) ** gamma)
        delta = rng.choice([-1.0, 1.0], size=dim)
        x_plus = x + ck * delta
        x_minus = x - ck * delta
        y_plus = f(x_plus)
        y_minus = f(x_minus)
        g_hat = (y_plus - y_minus) / (2.0 * ck * delta)
        x = x - ak * g_hat
        k += 1
    return np.array(f.history, dtype=float), f.eval_count

def run_scipy_lbfgsb(f, dim, bounds, max_evals, seed):
    from scipy.optimize import minimize
    np.random.seed(seed)
    # Start at a random location within bounds
    x0 = np.random.uniform(bounds[0], bounds[1], size=dim)
    bnds = [(bounds[0], bounds[1]) for _ in range(dim)]
    
    # maxfun strictly limits function evaluations for local solvers
    res = minimize(
        fun=f, 
        x0=x0, 
        method='L-BFGS-B', 
        bounds=bnds, 
        options={'maxfun': max_evals, 'ftol': 1e-10}
    )
    return np.array(f.history, dtype=float), f.eval_count

def run_cmaes(f, dim, bounds, max_evals, seed):
    from CMAES import CMAESOptimizer
    import numpy as np
    
    np.random.seed(seed)
    # Define population size based on dimensionality
    pop_size = 4 + int(3 * np.log(dim))
    
    # maxiter in CMA-ES (pycma) usually refers to generations
    # To stay within max_evals: max_generations = max_evals / popsize
    max_generations = max(1, max_evals // pop_size)
    
    # Initialize your custom class
    # Note: bounds in your class are expected as a tuple (low, high)
    lower_bnds = np.full(dim, bounds[0])
    upper_bnds = np.full(dim, bounds[1])
    
    optimizer = CMAESOptimizer(
        sigma0=0.15,
        maxiter=max_generations,
        popsize=pop_size,
        tolx=1e-10,
        tolfun=1e-10,
        bounds=(lower_bnds, upper_bnds),
        verbose=False
    )
    
    x0 = np.random.uniform(bounds[0], bounds[1], size=dim)
    
    # Execute minimization
    optimizer.minimize(fun=f, x0=x0)
    
    return np.array(f.history, dtype=float), f.eval_count

def run_rde(f, dim, bounds, max_evals, seed):
    from RDE import RDE_Optimizer
    np.random.seed(seed)
    opt = RDE_Optimizer(func=f, dim=dim, bounds=bounds, max_evals=max_evals)
    opt.optimize()
    return np.array(f.history, dtype=float), f.eval_count

def run_lsrtde(f, dim, bounds, max_evals, seed):
    from LSRTDE import LSRTDE_Optimizer
    np.random.seed(seed)
    opt = LSRTDE_Optimizer(func=f, dim=dim, bounds=bounds, max_evals=max_evals)
    opt.optimize()
    return np.array(f.history, dtype=float), f.eval_count

def run_gampc(f, dim, bounds, max_evals, seed):
    from GA11 import ga_mpc
    np.random.seed(seed)
    bnds = [(bounds[0], bounds[1]) for _ in range(dim)]
    ga_mpc(objective_func=f, bounds=bnds, max_fes=max_evals)
    return np.array(f.history, dtype=float), f.eval_count

def run_imode(f, dim, bounds, max_evals, seed):
    from IMODE import imode_optimizer_final 
    np.random.seed(seed)
    imode_optimizer_final(fhd=f, D=dim, Xmin=bounds[0], Xmax=bounds[1], Max_FEs=max_evals)
    return np.array(f.history, dtype=float), f.eval_count

def run_hses(f, dim, bounds, max_evals, seed):
    from HS_ES import hses_optimizer
    np.random.seed(seed)
    hses_optimizer(fhd=f, D=dim, Xmin=bounds[0], Xmax=bounds[1], Max_FEs=max_evals)
    return np.array(f.history, dtype=float), f.eval_count

def run_jso(f, dim, bounds, max_evals, seed):
    from jso import JSO
    np.random.seed(seed)
    JSO(dim, bounds, max_evals, seed).run(f)
    return np.array(f.history, dtype=float), f.eval_count

def run_nl_shade_lbc(f, dim, bounds, max_evals, seed):
    from NL_SHADE_LBC import nl_shade_lbc_optimizer
    np.random.seed(seed)
    nl_shade_lbc_optimizer(fhd=f, D=dim, Xmin=bounds[0], Xmax=bounds[1], Max_FEs=max_evals)
    return np.array(f.history, dtype=float), f.eval_count

def run_lshade_rsp(f, dim, bounds, max_evals, seed):
    from LSHADE_RSP import lshade_rsp_optimizer
    np.random.seed(seed)
    lshade_rsp_optimizer(fhd=f, D=dim, Xmin=bounds[0], Xmax=bounds[1], Max_FEs=max_evals)
    return np.array(f.history, dtype=float), f.eval_count

def run_ilshade(f, dim, bounds, max_evals, seed):
    from pyade import ilshade
    params = ilshade.get_default_params(dim)
    params["max_evals"] = max_evals
    params["bounds"] = np.array([[bounds[0], bounds[1]]] * dim, dtype=float)
    params["func"] = lambda x: f(x)
    params["seed"] = int(seed)
    params["population_size"] = dim * 4 
    ilshade.apply(**params)
    return np.array(f.history, dtype=float), f.eval_count

def run_scipy_de(f, dim, bounds, max_evals, seed):
    bnds = [(bounds[0], bounds[1]) for _ in range(dim)]
    popsize = 15
    # Scipy DE uses maxiter (generations). Total evals approx = maxiter * popsize * dim
    max_generations = int(max_evals / (popsize * dim)) 
    
    differential_evolution(
        f, 
        bounds=bnds, 
        maxiter=max_generations, 
        seed=seed, 
        strategy='best1bin',
        popsize=popsize, 
        tol=0.0, # Prevent early stopping
        disp=False
    )
    return np.array(f.history, dtype=float), f.eval_count

def run_experiments():
    dim = 20
    bounds = (-5.0, 5.0) 
    max_evals_global = 100000 
    n_runs = 1  
    
    problems = [
        ("VQA_Ising_10Q_L1", lambda d, n, rng=None: VQAIsingLogger(d, n, rng=rng, n_qubits=10))
    ]
    noise_levels = {"exact": 0.0, "64_shots": 1.0 / np.sqrt(64)}

    optimizers = [
        ("L-BFGS-B", run_scipy_lbfgsb),
        ("SPSA", run_spsa),
        ("CMA-ES", run_cmaes),
        ("SciPy DE (best1bin)", run_scipy_de),
        ("GA-MPC", run_gampc),
        ("LSHADE-RSP", run_lshade_rsp),
        ("HS-ES", run_hses),
        ("jSO", run_jso),
        ("iL-SHADE", run_ilshade),
        ("IMODE", run_imode),
        ("NL-SHADE-LBC", run_nl_shade_lbc),
        ("L-SRTDE", run_lsrtde)
    ]

    out_dir = "results_data"
    
    for prob_name, prob_class in problems:
        print(f"\n{'='*60}\nRunning: {prob_name}\n{'='*60}")
        
        for noise_label, noise_std in noise_levels.items():
            print(f"\n--- Noise Level: {noise_label} ---")

            for opt_name, opt_runner in optimizers:
                opt_dir = os.path.join(out_dir, prob_name, noise_label, opt_name)
                os.makedirs(opt_dir, exist_ok=True)
                
                final_values = []
                elapsed_times = []

                for run_idx in range(n_runs):
                    seed = np.random.randint(1, 999999)
                    rng = np.random.RandomState(seed)
                    f_logger = prob_class(dim, noise_std, rng=rng)

                    start_time = time.time()
                    try:
                        history, fes = opt_runner(f_logger, dim, bounds, max_evals_global, seed)
                        elapsed = time.time() - start_time
                        
                        # Save individual run data to .txt
                        # Format: Col 1 = FEs, Col 2 = Best Value
                        run_data = np.column_stack((np.arange(1, len(history) + 1), history))
                        txt_path = os.path.join(opt_dir, f"run_{run_idx + 1}.txt")
                        np.savetxt(txt_path, run_data, header="FE Best_Value", comments="")
                        
                        final_values.append(history[-1])
                        elapsed_times.append(elapsed)
                        
                    except Exception as e:
                        if run_idx == 0:
                            print(f"{opt_name:<20} | [SKIPPED] Error/Missing dependency: {str(e).splitlines()[0]}")
                        break 
                
                if final_values:
                    mean_best = np.mean(final_values)
                    std_best = np.std(final_values) # Added Standard Deviation
                    avg_time = np.mean(elapsed_times)
                    print(f"{opt_name:<20} | Mean Best: {mean_best:.4e} ± {std_best:.4e} | Avg Time/Run: {avg_time:.2f}s")

if __name__ == "__main__":
    run_experiments()