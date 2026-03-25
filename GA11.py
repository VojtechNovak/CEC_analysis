import numpy as np

def ga_mpc(objective_func, bounds, max_fes=150000, ps=90, p=0.1, cr=1.0, callback=None):
    """
    Genetic Algorithm with a New Multi-Parent Crossover (GA-MPC)
    Based on the IEEE-CEC2011 competition winner by Elsayed et al.
    
    Parameters:
    - objective_func: Callable, the objective function to minimize.
    - bounds: List of tuples [(min, max), ...] for each dimension.
    - max_fes: Maximum number of Function Evaluations (FEs).
    - ps: Population size (default 90).
    - p: Probability for the randomized operator.
    - cr: Crossover rate.
    - callback: Optional function `callback(iteration, fes, best_fitness, best_solution)`.
    
    Returns:
    - best_solution: The best variable vector found.
    - best_fitness: The minimum objective value.
    - history: Dictionary with 'fes' and 'best_fitness' lists for plotting.
    """
    D = len(bounds)
    lower_bound = np.array([b[0] for b in bounds])
    upper_bound = np.array([b[1] for b in bounds])
    
    # Trackers for visualization
    history_fes = []
    history_best = []
    
    # STEP 1: Initialization
    pop = lower_bound + np.random.rand(ps, D) * (upper_bound - lower_bound)
    fitness = np.array([objective_func(ind) for ind in pop])
    fes = ps
    
    # Archive size m = 50% of PS
    m = int(0.5 * ps)
    iteration = 0
    
    while fes < max_fes:
        # STEP 2: Sort individuals based on objective function
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # Save the best m individuals in the archive pool (A)
        archive = pop[:m].copy()
        archive_fitness = fitness[:m].copy()
        
        # Track convergence state
        history_fes.append(fes)
        history_best.append(fitness[0])
        if callback is not None:
            callback(iteration, fes, fitness[0], pop[0])
            
        # STEP 3: Tournament selection
        # Fill the selection pool indices
        pool_indices = np.zeros(ps, dtype=int)
        for i in range(ps):
            tc = np.random.choice([2, 3]) # Tournament size 2 or 3
            competitors = np.random.choice(ps, tc, replace=False)
            # Since population is sorted, the lowest index has the best fitness
            best_idx = np.min(competitors)
            pool_indices[i] = best_idx
            
        offspring = np.zeros_like(pop)
        
        # STEP 4: Crossover
        # Process in chunks of 3 consecutive individuals
        for i in range(0, ps, 3):
            # If PS is not a multiple of 3, handle the remainder gracefully
            if i + 3 > ps:
                offspring[i:] = pop[pool_indices[i:]].copy()
                break
                
            p_idx = list(pool_indices[i:i+3])
            
            # If one selected individual is the same as another, replace with a random one
            for j in range(3):
                for k in range(j+1, 3):
                    if np.linalg.norm(pop[p_idx[j]] - pop[p_idx[k]]) < 1e-8:
                        p_idx[k] = pool_indices[np.random.randint(ps)]
                        
            # Rank the three individuals from best to worst
            p_idx = sorted(p_idx, key=lambda idx: fitness[idx])
            x1, x2, x3 = pop[p_idx[0]], pop[p_idx[1]], pop[p_idx[2]]
            
            if np.random.rand() < cr:
                # Calculate beta = N(0.7, 0.1)
                beta = np.random.normal(0.7, 0.1)
                
                # Generate three offspring
                o1 = x1 + beta * (x2 - x3)
                o2 = x2 + beta * (x3 - x1)
                o3 = x3 + beta * (x1 - x2)
            else:
                o1, o2, o3 = x1.copy(), x2.copy(), x3.copy()
                
            # STEP 5: Randomized Operator
            for o in [o1, o2, o3]:
                for d in range(D):
                    if np.random.rand() < p:
                        arch_idx = np.random.randint(m)
                        o[d] = archive[arch_idx, d]
                        
            offspring[i], offspring[i+1], offspring[i+2] = o1, o2, o3
            
        # Ensure offspring are within bounds
        offspring = np.clip(offspring, lower_bound, upper_bound)
        
        # Evaluate offspring
        offspring_fitness = np.zeros(ps)
        for i in range(ps):
            if fes >= max_fes:
                offspring_fitness[i] = np.inf
                continue
            offspring_fitness[i] = objective_func(offspring[i])
            fes += 1
            
        # Merge archive pool and offspring to build the new population
        merged_pop = np.vstack((archive, offspring))
        merged_fit = np.concatenate((archive_fitness, offspring_fitness))
        
        # Select best PS individuals
        survivor_idx = np.argsort(merged_fit)[:ps]
        pop = merged_pop[survivor_idx]
        fitness = merged_fit[survivor_idx]
        
        # STEP 6: Shift duplicate individuals to preserve diversity
        for i in range(1, ps):
            # Because pop is sorted, duplicates will be adjacent
            if np.linalg.norm(pop[i] - pop[i-1]) < 1e-8:
                u = np.random.rand(D)
                # Shift by Normal dist N(0.5*u, 0.25*u)
                shift = np.random.normal(0.5 * u, 0.25 * u)
                pop[i] = np.clip(pop[i] + shift, lower_bound, upper_bound)
                
                if fes < max_fes:
                    fitness[i] = objective_func(pop[i])
                    fes += 1
                    
        iteration += 1

    # Final record
    sort_idx = np.argsort(fitness)
    history_fes.append(fes)
    history_best.append(fitness[sort_idx[0]])

    return pop[sort_idx[0]], fitness[sort_idx[0]], {'fes': history_fes, 'best_fitness': history_best}