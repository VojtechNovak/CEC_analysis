import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def imode_optimizer_final(fhd, D, Xmin=-100.0, Xmax=100.0, Max_FEs=None, callback=None):
    """
    Python translation of the final IMODE algorithm based on the CEC2020 paper.
    """
    if Max_FEs is None:
        Max_FEs = D * 100000 
        
    # --- Initialization Parameters ---
    PopSize = min(18 * D, int(6 * D**2)) # Scaled default for practicality (Paper suggests 6*D^2)
    MinPopSize = 4
    InitPop = PopSize
    arch_rate = 2.6
    
    memory_size = 20 * D
    archive_f = np.full(memory_size, 0.2)
    archive_Cr = np.full(memory_size, 0.2)
    hist_pos = 0
    
    archive_pop = np.empty((0, D))
    archive_fit = np.empty(0)
    
    prob = np.full(3, 1.0 / 3.0)
    
    # Initialize Population
    x = np.random.uniform(Xmin, Xmax, (PopSize, D))
    fitx = fhd(x)
    current_eval = PopSize
    iteration = 0
    
    sort_idx = np.argsort(fitx)
    x = x[sort_idx]
    fitx = fitx[sort_idx]
    
    bestx = x[0].copy()
    bestold = fitx[0]

    prob_ls = 0.1  # Local Search Probability

    if callback is not None:
        callback(iteration, current_eval, bestold, bestx)

    def get_indices(pop_size, total_size):
        r1 = np.zeros(pop_size, dtype=int)
        r2 = np.zeros(pop_size, dtype=int)
        r3 = np.zeros(pop_size, dtype=int)
        for i in range(pop_size):
            choices = np.delete(np.arange(total_size), i if i < total_size else -1)
            idx = np.random.choice(choices, 3, replace=False)
            r1[i], r2[i], r3[i] = idx[0], idx[1], idx[2]
            if r1[i] >= pop_size: r1[i] = r1[i] % pop_size
            if r3[i] >= pop_size: r3[i] = r3[i] % pop_size
        return r1, r2, r3

    # --- Main Loop ---
    while current_eval < Max_FEs:
        iteration += 1
        
        # 1. Linear Population Size Reduction (LPSR)
        UpdPopSize = int(round((((MinPopSize - InitPop) / Max_FEs) * current_eval) + InitPop))
        if PopSize > UpdPopSize:
            reduction_num = min(PopSize - UpdPopSize, PopSize - MinPopSize)
            if reduction_num > 0:
                PopSize -= reduction_num
                x = x[:PopSize]
                fitx = fitx[:PopSize]
                
            archive_NP = int(round(arch_rate * PopSize))
            if len(archive_pop) > archive_NP:
                keep_idx = np.random.choice(len(archive_pop), archive_NP, replace=False)
                archive_pop = archive_pop[keep_idx]
                archive_fit = archive_fit[keep_idx]
        
        # 2. Parameter Adaptation (F and Cr)
        mem_rand_index = np.random.randint(0, memory_size, size=PopSize)
        mu_sf = archive_f[mem_rand_index]
        mu_cr = archive_Cr[mem_rand_index]
        
        cr = np.random.normal(mu_cr, 0.1)
        cr[mu_cr == -1] = 0
        cr = np.clip(cr, 0.0, 1.0)
        cr = np.sort(cr) # Better individuals get lower Cr
        
        F = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(PopSize) - 0.5))
        mask_F_invalid = F <= 0
        while np.any(mask_F_invalid):
            F[mask_F_invalid] = mu_sf[mask_F_invalid] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_F_invalid)) - 0.5))
            mask_F_invalid = F <= 0
        F = np.clip(F, 0.0, 1.0)
        
        # 3. Assign Sub-populations based on Probabilities
        popAll = np.vstack((x, archive_pop)) if len(archive_pop) > 0 else x
        r1, r2, r3 = get_indices(PopSize, len(popAll))
        
        bb = np.random.rand(PopSize)
        op_1 = bb <= prob[0]
        op_2 = (bb > prob[0]) & (bb <= np.sum(prob[:2]))
        op_3 = bb > np.sum(prob[:2])
        
        vi = np.zeros_like(x)
        
        # Strategy 1: DE/current-to-best with archive/1
        if np.any(op_1):
            pNP1 = max(int(round(0.1 * PopSize)), 1) # Best 10%
            rand_idx1 = np.random.randint(0, pNP1, size=np.sum(op_1))
            phix1 = x[rand_idx1]
            vi[op_1] = x[op_1] + F[op_1][:, None] * (phix1 - x[op_1] + x[r1[op_1]] - popAll[r2[op_1]])
            
        # Strategy 2: DE/current-to-best without archive/1
        if np.any(op_2):
            pNP2 = max(int(round(0.1 * PopSize)), 1)
            rand_idx2 = np.random.randint(0, pNP2, size=np.sum(op_2))
            phix2 = x[rand_idx2]
            vi[op_2] = x[op_2] + F[op_2][:, None] * (phix2 - x[op_2] + x[r1[op_2]] - x[r3[op_2]])
            
        # Strategy 3: DE weighted-rand-to-best
        if np.any(op_3):
            pNP3 = max(int(round(0.1 * PopSize)), 1)
            rand_idx3 = np.random.randint(0, pNP3, size=np.sum(op_3))
            phix3 = x[rand_idx3]
            vi[op_3] = F[op_3][:, None] * x[r1[op_3]] + F[op_3][:, None] * (phix3 - x[r3[op_3]])
            
        # Boundary Handling (Midpoint)
        mask_lower = vi < Xmin
        vi[mask_lower] = (Xmin + x[mask_lower]) / 2.0
        mask_upper = vi > Xmax
        vi[mask_upper] = (Xmax + x[mask_upper]) / 2.0
        
        # 4. Crossover (Randomly choosing Binomial or Exponential)
        ui = np.copy(x)
        if np.random.rand() < 0.5:
            # Binomial
            j_rand = np.random.randint(0, D, size=PopSize)
            cross_mask = (np.random.rand(PopSize, D) <= cr[:, None]) | (np.arange(D) == j_rand[:, None])
            ui = np.where(cross_mask, vi, x)
        else:
            # Exponential
            startLoc = np.random.randint(0, D, size=PopSize)
            for i in range(PopSize):
                l = startLoc[i]
                while np.random.rand() < cr[i] and l < D - 1:
                    l += 1
                ui[i, startLoc[i]:l+1] = vi[i, startLoc[i]:l+1]
                
        # 5. Evaluation
        fitx_new = fhd(ui)
        current_eval += PopSize
        
        # 6. Quality (QR) and Diversity (DR) Calculation for Operator Probabilities
        D_op = np.zeros(3)
        Fit_op_best = np.zeros(3)
        
        def calc_diversity_quality(op_mask):
            if not np.any(op_mask): return 0.0, 0.0
            sub_pop = ui[op_mask]
            sub_fit = fitx_new[op_mask]
            best_idx = np.argmin(sub_fit)
            best_sol = sub_pop[best_idx]
            div = np.mean(np.linalg.norm(sub_pop - best_sol, axis=1))
            return div, sub_fit[best_idx]

        D_op[0], Fit_op_best[0] = calc_diversity_quality(op_1)
        D_op[1], Fit_op_best[1] = calc_diversity_quality(op_2)
        D_op[2], Fit_op_best[2] = calc_diversity_quality(op_3)
        
        # Normalize DR and QR
        DR_op = D_op / np.sum(D_op) if np.sum(D_op) > 0 else np.full(3, 1/3)
        
        # Shift fitness to positive for QR division if needed
        shifted_fit = Fit_op_best - np.min(Fit_op_best) + 1e-8 
        QR_op = shifted_fit / np.sum(shifted_fit)
        
        # Calculate Improvement Rate Value (IRV)
        IRV_op = (1.0 - QR_op) + DR_op
        IRV_op = np.maximum(0.1, np.minimum(0.9, IRV_op / np.sum(IRV_op)))
        prob = IRV_op / np.sum(IRV_op)

        # 7. Selection & Memory Update
        improved = fitx_new < fitx
        diff = np.abs(fitx - fitx_new)
        
        goodCR = cr[improved]
        goodF = F[improved]
        
        if len(goodCR) > 0:
            archive_pop = np.vstack((archive_pop, x[improved]))
            archive_fit = np.concatenate((archive_fit, fitx[improved]))
            archive_NP = int(round(arch_rate * PopSize))
            if len(archive_pop) > archive_NP:
                keep_idx = np.random.choice(len(archive_pop), archive_NP, replace=False)
                archive_pop = archive_pop[keep_idx]
                archive_fit = archive_fit[keep_idx]
            
            weightsDE = diff[improved] / np.sum(diff[improved])
            archive_f[hist_pos] = np.sum(weightsDE * (goodF ** 2)) / np.sum(weightsDE * goodF)
            
            if np.max(goodCR) == 0 or archive_Cr[hist_pos] == -1:
                archive_Cr[hist_pos] = -1
            else:
                archive_Cr[hist_pos] = np.sum(weightsDE * (goodCR ** 2)) / np.sum(weightsDE * goodCR)
                
            hist_pos = (hist_pos + 1) % memory_size
        else:
            archive_Cr[hist_pos] = 0.5
            archive_f[hist_pos] = 0.5
            
        # Update Population
        x[improved] = ui[improved]
        fitx[improved] = fitx_new[improved]
        
        sort_idx = np.argsort(fitx)
        x = x[sort_idx]
        fitx = fitx[sort_idx]
        
        if fitx[0] < bestold:
            bestold = fitx[0]
            bestx = x[0].copy()
            
        # 8. Local Search (SQP) Phase
        if current_eval >= 0.85 * Max_FEs and current_eval < Max_FEs:
            if np.random.rand() <= prob_ls:
                CFE_ls = min(int(np.ceil(0.02 * Max_FEs)), Max_FEs - current_eval)
                evals_in_ls = 0
                
                def fhd_1d(x_1d):
                    nonlocal evals_in_ls
                    evals_in_ls += 1
                    return fhd(np.array([x_1d]))[0]
                
                bounds = [(Xmin, Xmax) for _ in range(D)]
                res = minimize(fhd_1d, bestx, method='SLSQP', bounds=bounds, options={'maxiter': CFE_ls})
                
                current_eval += evals_in_ls
                
                if res.fun < bestold:
                    prob_ls = 0.1
                    bestold = res.fun
                    bestx = res.x.copy()
                    x[-1] = bestx
                    fitx[-1] = bestold
                    
                    sort_idx = np.argsort(fitx)
                    x = x[sort_idx]
                    fitx = fitx[sort_idx]
                else:
                    prob_ls = 0.0001

        if callback is not None:
            callback(iteration, current_eval, bestold, bestx)

    return bestold, bestx