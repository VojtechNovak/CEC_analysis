import numpy as np

def lshade_rsp_optimizer(fhd, D, Xmin=-100.0, Xmax=100.0, Max_FEs=None, callback=None):
    """
    Python translation of LSHADE-RSP (CEC 2018).
    
    Parameters:
    - fhd: Objective function, takes an (N, D) numpy array and returns an (N,) array of fitnesses.
    - D: Dimensionality of the problem.
    - Xmin, Xmax: Boundaries of the search space.
    - Max_FEs: Maximum function evaluations.
    - callback: Function called at tracked intervals -> callback(iteration, FEs, best_fitness, best_solution)
    """
    if Max_FEs is None:
        Max_FEs = D * 10000
        
    # --- Parameter Initialization ---
    NIndsMax = int(75 * (D ** (2/3)))
    NInds = NIndsMax
    NIndsMin = 4
    ArchiveSizeParam = 1.0
    psizeParam = 0.17
    
    MemorySize = 5
    MemoryIter = 0
    MemoryCr = np.full(MemorySize, 0.8)
    MemoryF = np.full(MemorySize, 0.3)
    
    Archive = np.empty((0, D))
    
    # Initialize Population
    Popul = np.random.uniform(Xmin, Xmax, (NInds, D))
    FitMass = fhd(Popul)
    FEs = NInds
    iteration = 0
    
    besti = np.argmin(FitMass)
    bestfit = FitMass[besti]
    bestvec = Popul[besti].copy()
    
    if callback is not None:
        callback(iteration, FEs, bestfit, bestvec)
        
    # --- Main Loop ---
    while FEs < Max_FEs:
        iteration += 1
        
        # Rank based calculations
        sort_idx = np.argsort(FitMass)
        ranks = np.empty_like(sort_idx)
        ranks[sort_idx] = np.arange(NInds) # 0 is best, NInds-1 is worst
        
        fit_temp = 3.0 * (NInds - ranks)
        probs = fit_temp / np.sum(fit_temp)
        
        psize = ((psizeParam / 2.0) / Max_FEs * FEs + psizeParam / 2.0)
        psizeval = max(2, int(NInds * psize))
        
        # Memory indices and parameters for the generation
        mem_idx = np.random.randint(0, MemorySize + 1, size=NInds)
        
        # Generate F
        F_arr = np.zeros(NInds)
        for i in range(NInds):
            mu_f = MemoryF[mem_idx[i]] if mem_idx[i] < MemorySize else 0.9
            while F_arr[i] <= 0:
                F_arr[i] = mu_f + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5)) # Cauchy
            if F_arr[i] > 1.0: F_arr[i] = 1.0
            
            # RSP specific F-capping
            if (FEs / Max_FEs) < 0.6 and F_arr[i] > 0.7:
                F_arr[i] = 0.7
                
        # Generate F2
        F2_arr = np.zeros(NInds)
        ratio = FEs / Max_FEs
        mask1 = ratio < 0.2
        mask2 = (ratio >= 0.2) & (ratio < 0.4)
        mask3 = ratio >= 0.4
        F2_arr[mask1] = 0.7 * F_arr[mask1]
        F2_arr[mask2] = 0.8 * F_arr[mask2]
        F2_arr[mask3] = 1.2 * F_arr[mask3]
        
        # Generate Cr
        Cr_arr = np.zeros(NInds)
        for i in range(NInds):
            if mem_idx[i] < MemorySize:
                mu_cr = MemoryCr[mem_idx[i]]
                if mu_cr < 0:
                    Cr_arr[i] = 0.0
                else:
                    Cr_arr[i] = np.random.normal(mu_cr, 0.1)
            else:
                Cr_arr[i] = np.random.normal(0.9, 0.1)
                
        Cr_arr = np.clip(Cr_arr, 0.0, 1.0)
        if ratio < 0.25: Cr_arr = np.maximum(Cr_arr, 0.7)
        elif ratio < 0.5: Cr_arr = np.maximum(Cr_arr, 0.6)
        
        # Vectorized Parent Selection
        prand = np.zeros(NInds, dtype=int)
        r1 = np.zeros(NInds, dtype=int)
        r2 = np.zeros(NInds, dtype=int)
        use_archive = np.random.rand(NInds) < (len(Archive) / (len(Archive) + NInds) if len(Archive) > 0 else 0)
        
        for i in range(NInds):
            # Select pbest
            while True:
                p_idx = np.random.choice(sort_idx[:psizeval])
                if p_idx != i or ratio >= 0.5:
                    prand[i] = p_idx
                    break
            
            # Select r1
            while True:
                r1_idx = np.random.choice(NInds, p=probs)
                if r1_idx != prand[i]:
                    r1[i] = r1_idx
                    break
                    
            # Select r2
            if use_archive[i]:
                r2[i] = np.random.randint(0, len(Archive))
            else:
                while True:
                    r2_idx = np.random.choice(NInds, p=probs)
                    if r2_idx != prand[i] and r2_idx != r1[i]:
                        r2[i] = r2_idx
                        break
        
        # Build Trial Vectors
        Donor = np.zeros_like(Popul)
        for i in range(NInds):
            p1 = Popul[prand[i]]
            p2 = Popul[r1[i]]
            p3 = Archive[r2[i]] if use_archive[i] else Popul[r2[i]]
            Donor[i] = Popul[i] + F2_arr[i] * (p1 - Popul[i]) + F_arr[i] * (p2 - p3)
            
        # Boundary Handling (Midpoint)
        mask_lower = Donor < Xmin
        Donor[mask_lower] = (Xmin + Popul[mask_lower]) / 2.0
        mask_upper = Donor > Xmax
        Donor[mask_upper] = (Xmax + Popul[mask_upper]) / 2.0
        
        # Binomial Crossover
        j_rand = np.random.randint(0, D, size=NInds)
        cross_mask = (np.random.rand(NInds, D) < Cr_arr[:, None]) | (np.arange(D) == j_rand[:, None])
        Trial = np.where(cross_mask, Donor, Popul)
        
        # Evaluation
        TrialFit = fhd(Trial)
        FEs += NInds
        
        # Selection & Success Storage
        success_idx = np.where(TrialFit <= FitMass)[0]
        
        if len(success_idx) > 0:
            success_Cr = Cr_arr[success_idx]
            success_F = F_arr[success_idx]
            FitDelta = np.abs(FitMass[success_idx] - TrialFit[success_idx])
            
            # Append replaced parents to archive
            new_archive_items = Popul[success_idx]
            Archive = np.vstack((Archive, new_archive_items))
            
            # Update Population
            Popul[success_idx] = Trial[success_idx]
            FitMass[success_idx] = TrialFit[success_idx]
            
            # Memory Update (Weighted Lehmer Mean)
            if np.max(success_Cr) == 0 or MemoryCr[MemoryIter] == -1:
                MemoryCr[MemoryIter] = -1
            else:
                weights = FitDelta / np.sum(FitDelta)
                meanWL_Cr = np.sum(weights * success_Cr**2) / np.sum(weights * success_Cr)
                MemoryCr[MemoryIter] = (meanWL_Cr + MemoryCr[MemoryIter]) / 2.0
                
            weights = FitDelta / np.sum(FitDelta)
            meanWL_F = np.sum(weights * success_F**2) / np.sum(weights * success_F)
            MemoryF[MemoryIter] = (meanWL_F + MemoryF[MemoryIter]) / 2.0
            
            MemoryIter = (MemoryIter + 1) % MemorySize
            
        # Track global best
        current_besti = np.argmin(FitMass)
        if FitMass[current_besti] < bestfit:
            bestfit = FitMass[current_besti]
            bestvec = Popul[current_besti].copy()
            
        if callback is not None:
            callback(iteration, FEs, bestfit, bestvec)
            
        # Maintain Archive Size
        ArchiveSize = int((Max_FEs - FEs) / Max_FEs * (ArchiveSizeParam * (NIndsMax - NIndsMin)))
        ArchiveSize = max(ArchiveSize, NIndsMin)
        if len(Archive) > ArchiveSize:
            keep_indices = np.random.choice(len(Archive), ArchiveSize, replace=False)
            Archive = Archive[keep_indices]
            
        # Population Size Reduction (LPSR)
        newNInds = int((NIndsMin - NIndsMax) / Max_FEs * FEs + NIndsMax)
        newNInds = np.clip(newNInds, NIndsMin, NIndsMax)
        
        if newNInds < NInds:
            keep_idx = np.argsort(FitMass)[:newNInds]
            Popul = Popul[keep_idx]
            FitMass = FitMass[keep_idx]
            NInds = newNInds

    return bestfit, bestvec

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    def rastrigin_function(x):
        # f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))
        A = 10
        return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

    def callback_tracker(iteration, FEs, best_fitness, best_solution):
        if iteration % 100 == 0 or FEs >= 10000 * 10 - 20: 
            print(f"Iter: {iteration:4d} | FEs: {FEs:6d} | Best Fit: {best_fitness:.8f}")

    print("Starting LSHADE-RSP...")
    best_val, best_vec = lshade_rsp_optimizer(
        fhd=rastrigin_function,
        D=10, 
        Xmin=-100.0, 
        Xmax=100.0, 
        Max_FEs=10000 * 10, 
        callback=callback_tracker
    )
    
    print("\nOptimization Complete!")
    print(f"Global Best Value: {best_val}")