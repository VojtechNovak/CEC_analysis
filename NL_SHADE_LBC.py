import numpy as np

def nl_shade_lbc_optimizer(fhd, D, Xmin=-100.0, Xmax=100.0, Max_FEs=None, callback=None):
    """
    Python translation of NL-SHADE-LBC (CEC 2022).
    
    Parameters:
    - fhd: Objective function, takes an (N, D) numpy array and returns an (N,) array of fitnesses.
    - D: Dimensionality of the problem.
    - Xmin, Xmax: Boundaries of the search space.
    - Max_FEs: Maximum function evaluations.
    - callback: Function called at tracked intervals -> callback(iteration, FEs, best_fitness, best_solution)
    """
    if Max_FEs is None:
        if D == 10:
            Max_FEs = 200000
        elif D == 20:
            Max_FEs = 1000000
        else:
            Max_FEs = D * 10000
            
    # --- Parameter Initialization ---
    NIndsMax = 23 * D
    NIndsMin = 4
    NInds = NIndsMax
    ArchiveSizeParam = 1.0
    
    MemorySize = 20 * D
    MemoryIter = 0
    MemoryCr = np.full(MemorySize, 0.9)
    MemoryF = np.full(MemorySize, 0.5)
    
    MWLp1 = 3.5
    MWLp2 = 1.0
    MWLm = 1.5
    LBC_fin = 1.5
    
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
        
    def mean_wl_general(vec, weights_diff, p, m):
        """Calculates the generalized Weighted Lehmer Mean."""
        weights = weights_diff / np.sum(weights_diff)
        num = np.sum(weights * (vec ** p))
        den = np.sum(weights * (vec ** (p - m)))
        if abs(den) > 1e-6:
            return num / den
        return 0.5
        
    # --- Main Loop ---
    while FEs < Max_FEs:
        iteration += 1
        
        # Rank calculations for sorting Cr
        sort_idx = np.argsort(FitMass)
        ranks = np.empty_like(sort_idx)
        ranks[sort_idx] = np.arange(NInds) # 0 is best, NInds-1 is worst
        
        # Selection probabilities for r2
        FitTemp3 = np.exp(-np.arange(NInds) / NInds)
        probs_r2 = FitTemp3 / np.sum(FitTemp3)
        
        # pbest size increases linearly
        psizeval = max(2, int(NInds * (0.1 / Max_FEs * FEs + 0.2)))
        
        # Generate Cr and F parameters
        mem_idx = np.random.randint(0, MemorySize, size=NInds)
        
        Cr_gen = np.clip(np.random.normal(MemoryCr[mem_idx], 0.1), 0.0, 1.0)
        # NL-SHADE trait: sorting Cr so that better individuals get lower Cr
        Cr_sorted = np.sort(Cr_gen)
        Cr_arr = Cr_sorted[ranks]
        
        F_arr = np.zeros(NInds)
        for i in range(NInds):
            while F_arr[i] <= 0:
                F_arr[i] = MemoryF[mem_idx[i]] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5)) # Cauchy
        F_arr = np.clip(F_arr, 0.0, 1.0)
        
        # Vectorized Parent Selection
        prand = np.zeros(NInds, dtype=int)
        r1 = np.zeros(NInds, dtype=int)
        r2 = np.zeros(NInds, dtype=int)
        use_archive = (np.random.rand(NInds) < 0.5) & (len(Archive) > 0)
        
        for i in range(NInds):
            # Select pbest
            while True:
                p_idx = np.random.choice(sort_idx[:psizeval])
                if p_idx != i:
                    prand[i] = p_idx
                    break
            
            # Select r1 (Uniform)
            while True:
                r1_idx = np.random.randint(0, NInds)
                if r1_idx != prand[i] and r1_idx != i:
                    r1[i] = r1_idx
                    break
                    
            # Select r2 (Rank-based from Pop, or Uniform from Archive)
            if use_archive[i]:
                r2[i] = np.random.randint(0, len(Archive))
            else:
                while True:
                    r2_idx = np.random.choice(sort_idx, p=probs_r2)
                    if r2_idx != prand[i] and r2_idx != r1[i] and r2_idx != i:
                        r2[i] = r2_idx
                        break
        
        # Build Trial Vectors (current-to-pbest/1)
        Donor = np.zeros_like(Popul)
        for i in range(NInds):
            p1 = Popul[prand[i]]
            p2 = Popul[r1[i]]
            p3 = Archive[r2[i]] if use_archive[i] else Popul[r2[i]]
            Donor[i] = Popul[i] + F_arr[i] * (p1 - Popul[i]) + F_arr[i] * (p2 - p3)
            
        # Boundary Handling (Midpoint towards base parent)
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
            
            # Memory Update (Adaptive Generalized Lehmer Mean)
            ratio = FEs / Max_FEs
            FMWL = LBC_fin + (MWLp1 - LBC_fin) * (1.0 - ratio)
            CrMWL = LBC_fin + (MWLp2 - LBC_fin) * (1.0 - ratio)
            
            MemoryF[MemoryIter] = (MemoryF[MemoryIter] + mean_wl_general(success_F, FitDelta, FMWL, MWLm)) * 0.5
            MemoryCr[MemoryIter] = (MemoryCr[MemoryIter] + mean_wl_general(success_Cr, FitDelta, CrMWL, MWLm)) * 0.5
            
            MemoryIter = (MemoryIter + 1) % MemorySize
        else:
            # If no success, regress toward default
            MemoryF[MemoryIter] = 0.5
            MemoryCr[MemoryIter] = 0.5
            MemoryIter = (MemoryIter + 1) % MemorySize
            
        # Track global best
        current_besti = np.argmin(FitMass)
        if FitMass[current_besti] < bestfit:
            bestfit = FitMass[current_besti]
            bestvec = Popul[current_besti].copy()
            
        if callback is not None:
            callback(iteration, FEs, bestfit, bestvec)
            
        # Non-linear Population Size Reduction (LPSR)
        ratio = FEs / Max_FEs
        # Apply the specific power curve from the C++ file
        newNInds = int(round((NIndsMin - NIndsMax) * (ratio ** (1.0 - ratio)) + NIndsMax))
        newNInds = np.clip(newNInds, NIndsMin, NIndsMax)
        
        # Shrink Archive
        ArchiveSize = int(newNInds * ArchiveSizeParam)
        ArchiveSize = max(ArchiveSize, NIndsMin)
        if len(Archive) > ArchiveSize:
            keep_indices = np.random.choice(len(Archive), ArchiveSize, replace=False)
            Archive = Archive[keep_indices]
            
        # Shrink Population
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
    def sphere_function(x):
        return np.sum(x**2, axis=1)

    def callback_tracker(iteration, FEs, best_fitness, best_solution):
        if iteration % 100 == 0 or FEs >= 200000 - 100: 
            print(f"Iter: {iteration:4d} | FEs: {FEs:6d} | Best Fit: {best_fitness:.12f}")

    print("Starting NL-SHADE-LBC...")
    best_val, best_vec = nl_shade_lbc_optimizer(
        fhd=sphere_function,
        D=10, 
        Xmin=-100.0, 
        Xmax=100.0, 
        Max_FEs=200000, 
        callback=callback_tracker
    )
    
    print("\nOptimization Complete!")
    print(f"Global Best Value: {best_val}")