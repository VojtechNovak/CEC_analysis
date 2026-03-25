import numpy as np

def hses_optimizer(fhd, D, Xmin=-100.0, Xmax=100.0, Max_FEs=None, callback=None):
    """
    Python translation of the HSES optimization algorithm.
    
    Parameters:
    - fhd: Objective function. Should take a 2D numpy array of shape (N, D) and return a 1D array of fitnesses (N,).
    - D: Dimensionality of the problem.
    - Xmin: Minimum boundary.
    - Xmax: Maximum boundary.
    - Max_FEs: Maximum number of function evaluations (defaults to 10000 * D).
    - callback: Function called at tracked intervals -> callback(iteration, FEs, best_fitness, best_solution)
    """
    if Max_FEs is None:
        Max_FEs = 10000 * D
        
    FEs = 0
    iteration = 0
    
    def evaluate(pos):
        nonlocal FEs
        fitness = fhd(pos)
        FEs += pos.shape[0]
        return fitness

    def trigger_callback(best_val, best_vec):
        if callback is not None:
            callback(iteration, FEs, best_val, best_vec)

    # ==========================================
    # Phase 1: Initial Univariate Sampling
    # ==========================================
    total = 200
    mu = 100
    
    # Initialize population
    pos = np.random.uniform(Xmin, Xmax, (total, D))
    e = evaluate(pos)
    
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    
    sorted_indices = np.argsort(e)
    bestval = e[sorted_indices[0]]
    bestvec = pos[sorted_indices[0], :].copy()
    
    trigger_callback(bestval, bestvec)
    
    pos_top = pos[sorted_indices[:mu], :]
    meanval = np.mean(pos_top, axis=0)
    stdval = np.std(pos_top, axis=0)
    
    cc1 = 0
    FV = np.zeros(100)
    
    for kk in range(1, 101):
        iteration += 1
        
        # Resample
        for k in range(total):
            a = 0.96 * np.random.randn(D) if cc1 == 1 else np.random.randn(D)
            pos[k, :] = meanval + stdval * a
            
        # Boundary handling for Phase 1
        out_of_bounds = (pos > Xmax) | (pos < Xmin)
        while np.any(out_of_bounds):
            rows, cols = np.where(out_of_bounds)
            pos[rows, cols] = meanval[cols] + stdval[cols] * np.random.randn(len(rows))
            out_of_bounds = (pos > Xmax) | (pos < Xmin)

        e = evaluate(pos)
        sorted_indices = np.argsort(e)
        
        if e[sorted_indices[0]] < bestval:
            bestval = e[sorted_indices[0]]
            bestvec = pos[sorted_indices[0], :].copy()
            
        newpos = pos[sorted_indices[:mu], :]
        meanval = np.dot(weights, newpos)
        stdval = 1.0 * np.std(newpos, axis=0)
        
        FV[kk-1] = e[sorted_indices[0]]
        if kk > 30 and kk % 20 == 0:
            aa2 = np.argmin(FV[:kk])
            if aa2 < kk - 20:
                cc1 = 1
                
        trigger_callback(bestval, bestvec)

    # ==========================================
    # Phase 2: CMA-ES Exploration
    # ==========================================
    previousbest = e[sorted_indices[0]]
    Times = 2 if D <= 30 else 1
    
    arfitnessbest = np.full(Times, bestval)
    xvalbest = np.tile(bestvec, (Times, 1)).T 
    
    for kkk in range(Times):
        sigma = 0.2
        stopeval = Max_FEs / 4 if D <= 30 else Max_FEs / 2
        
        lambda_ = int(np.floor(3 * np.log(D))) + 80
        mu_cma = int(np.floor(lambda_ / 2))
        weights_cma = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
        weights_cma = weights_cma / np.sum(weights_cma)
        mueff = np.sum(weights_cma)**2 / np.sum(weights_cma**2)
        
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3)**2 + mueff)
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((D + 2)**2 + 2 * mueff / 2)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        
        pc = np.zeros(D)
        ps = np.zeros(D)
        B = np.eye(D)
        DD = np.eye(D)
        C = B @ DD @ (B @ DD).T
        eigenval = 0
        chiN = D**0.5 * (1 - 1 / (4 * D) + 1 / (21 * D**2))
        
        counteval = 0
        xmean = bestvec.copy()
        
        arz = np.zeros((D, lambda_))
        arxx = np.zeros((D, lambda_))
        arfitness = np.zeros(lambda_)
        
        while counteval < stopeval and FEs < Max_FEs:
            iteration += 1
            for k in range(lambda_):
                arz[:, k] = np.random.randn(D)
                arxx[:, k] = xmean + 1 * sigma * (B @ DD @ arz[:, k])
                
                # Phase 2 Boundary handling (modulo logic from MATLAB)
                arxx[:, k] = np.where(arxx[:, k] > Xmax, arxx[:, k] % Xmax, arxx[:, k])
                arxx[:, k] = np.where(arxx[:, k] < Xmin, arxx[:, k] % Xmin, arxx[:, k])
                
                arfitness[k] = evaluate(arxx[:, k].reshape(1, -1))[0]
                counteval += 1
                
            arindex = np.argsort(arfitness)
            
            if abs(arfitness[arindex[0]] - previousbest) < 1e-11:
                break
            else:
                previousbest = arfitness[arindex[0]]
                
            if arfitnessbest[kkk] > arfitness[arindex[0]]:
                arfitnessbest[kkk] = arfitness[arindex[0]]
                xvalbest[:, kkk] = arxx[:, arindex[0]]
                
            xmean = arxx[:, arindex[:mu_cma]] @ weights_cma
            zmean = arz[:, arindex[:mu_cma]] @ weights_cma
            
            ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff)) * (B @ zmean)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lambda_)) / chiN < 1.4 + 2 / (D + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ DD @ zmean)
            
            artmp = (B @ DD @ arz[:, arindex[:mu_cma]])
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * artmp @ np.diag(weights_cma) @ artmp.T
                
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            if counteval - eigenval > lambda_ / cmu / D / 10:
                eigenval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                evals, B = np.linalg.eigh(C)
                evals = np.maximum(0, evals) # prevent negative due to numerical issues
                DD = np.diag(np.sqrt(evals))
                
            if arfitness[arindex[0]] == arfitness[arindex[int(np.ceil(0.7 * lambda_)) - 1]]:
                sigma = sigma * np.exp(0.2 + cs / damps)

            if arfitnessbest[kkk] < bestval:
                bestval = arfitnessbest[kkk]
                bestvec = xvalbest[:, kkk].copy()
                
            trigger_callback(bestval, bestvec)

    # ==========================================
    # Phase 3: Final Univariate Refinement
    # ==========================================
    if D <= 30:
        total, mu = 200, 160
    elif D == 50:
        total, mu = 450, 360
    else:
        total, mu = 600, 480
        
    if D >= 50 and FEs <= 0.3 * Max_FEs:
        total += 200
        mu += 160
        
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    
    dividevalue = 0
    bbpbb = np.ones(D)
    
    # MATLAB dimension splitting logic
    if D <= 30:
        ppp1 = np.std(xvalbest.T, axis=0)
        ppp2 = np.sort(ppp1)
        if ppp2[0] > 0.2:
            dividevalue = 0
        elif np.max(ppp2) < 0.01:
            dividevalue = 1
        else:
            indicatorppp = np.zeros(D)
            for dd in range(1, D):
                indicatorppp[dd] = (ppp2[dd] - ppp2[dd-1]) / ppp2[dd-1]
            indicatorppp[0] = np.min(indicatorppp) - 0.001
            value2 = np.argsort(indicatorppp)[::-1]
            
            for dd in range(D):
                idx = value2[dd]
                if ppp2[idx] < 10 and ppp2[idx] > 0.1:
                    dividevalue = ppp2[idx] - 0.001
                    break
                elif ppp2[idx-1] < 0.01:
                    dividevalue = ppp2[idx] - 0.001
                    break
                if dd == D - 1:
                    dividevalue = ppp2[idx] - 0.001
    else:
        spos = np.tile(xvalbest[:, 0], (total // 5, 1))
        bbpbbp = np.zeros(D)
        for d in range(D):
            for k in range(total // 5):
                spos[k, d] = xvalbest[d, 0] - 0.1 * total + 1 * (k + 1)
            e_val = evaluate(spos)
            bbpbbp[d] = abs(np.max(e_val) / arfitnessbest[0])
            spos[:, d] = xvalbest[d, 0] # Reset
            
        if np.max(bbpbbp) >= 3.1:
            aaa1 = np.sort(bbpbbp)
            diaaa1 = aaa1[1:] / aaa1[:-1]
            aab2 = np.argsort(diaaa1)[::-1]
            division = 0
            
            if aaa1[D // 2] <= 2:
                for d in range(D - 1):
                    if aaa1[aab2[d]] < 1.8:
                        division = aaa1[aab2[d]] + 0.01
                        break
            else:
                for d in range(D - 1):
                    if aaa1[aab2[d]] < 4:
                        division = aaa1[aab2[d]] + 0.01
                        break
                        
            bbpbb = (bbpbbp <= division).astype(int)

    pos = np.random.uniform(Xmin, Xmax, (total, D))
    cc2 = 0
    xmin_hist = []
    kk = 1
    
    seq = np.argmin(arfitnessbest)
    
    while FEs < Max_FEs - total:
        iteration += 1
        e1 = evaluate(pos)
        sorted_indices = np.argsort(e1)
        xmin_hist.append(e1[sorted_indices[0]])
        
        if e1[sorted_indices[0]] < arfitnessbest[seq]:
            bestval = e1[sorted_indices[0]]
            bestvec = pos[sorted_indices[0], :].copy()
        else:
            bestval = arfitnessbest[seq]
            bestvec = xvalbest[:, seq].copy()
            
        trigger_callback(bestval, bestvec)

        newpos = pos[sorted_indices[:mu], :]
        meanval = np.dot(weights, newpos)
        stdval = 1.0 * np.std(newpos, axis=0)
        
        if kk == 1:
            if D >= 50:
                mask = bbpbb == 0
                stdval[mask] = 0.001
                meanval[mask] = xvalbest[mask, seq]
            else:
                mask = ppp1 < dividevalue
                stdval[mask] = 0.001
                meanval[mask] = xvalbest[mask, seq]
                
        if kk > 30 and kk % 20 == 0:
            if np.argmin(xmin_hist) < kk - 20:
                cc2 = 1
            else:
                cc2 = 0
                
        kk += 1
        
        # Resample Phase 3
        for k in range(total):
            a = 0.96 * np.random.randn(D) if cc2 == 1 else np.random.randn(D)
            pos[k, :] = meanval + stdval * a
            
        # Boundary handling Phase 3
        out_of_bounds = (pos > Xmax) | (pos < Xmin)
        while np.any(out_of_bounds):
            rows, cols = np.where(out_of_bounds)
            pos[rows, cols] = meanval[cols] + stdval[cols] * np.random.randn(len(rows))
            out_of_bounds = (pos > Xmax) | (pos < Xmin)

    return bestval, bestvec

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Define a dummy objective function (e.g., Sphere function)
    # The optimizer expects a vectorized function: input shape (N, D) -> output shape (N,)
    def sphere_function(x):
        return np.sum(x**2, axis=1)

    # Define the tracking callback
    def tracking_callback(iteration, FEs, best_fitness, best_solution):
        # Print every 500th iteration to avoid flooding the console
        if iteration % 500 == 0 or FEs >= 10000 * 10 - 600: 
            print(f"Iter: {iteration:4d} | FEs: {FEs:6d} | Best Fitness: {best_fitness:.8f}")

    print("Starting Optimization...")
    best_val, best_vec = hses_optimizer(
        fhd=sphere_function,
        D=10, 
        Xmin=-100.0, 
        Xmax=100.0, 
        Max_FEs=10000 * 10, 
        callback=tracking_callback
    )
    
    print("\nOptimization Finished!")
    print(f"Final Best Value: {best_val}")
    print(f"Final Best Vector: {best_vec}")