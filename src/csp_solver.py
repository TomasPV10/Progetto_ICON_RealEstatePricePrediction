import os
import sys
import time
import numpy as np
import pandas as pd

def run_csp_solver():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    
    file_path = os.path.join(data_dir, 'housing_matching_supervised.csv')
    if not os.path.exists(file_path):
        print(f"\n[ERRORE] File non trovato in: {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    # Parametri del CSP
    SHORTLIST_SIZE = 10
    MAX_PRICE = 45.0
    MAX_DISTANCE_KM = 3.0
    MIN_BUDGET_FIT = 0.50
    MAX_SAME_TYPE = 4
    MAX_SAME_ENERGY = 4
    
    # Funzione di valutazione / Penalità
    def calculate_penalty(solution_indices):
        subset = df.iloc[solution_indices]
        penalty = 0.0
        
        # Hard Constraint 1: Prezzo massimo
        price_violations = (subset['unit_price'] > MAX_PRICE).sum()
        penalty += price_violations * 50.0
        
        # Hard Constraint 2: Distanza massima
        dist_violations = (subset['distance_km'] > MAX_DISTANCE_KM).sum()
        penalty += dist_violations * 30.0
        
        # Hard Constraint 3: Budget Fit medio
        avg_fit = subset['budget_fit'].mean()
        if avg_fit < MIN_BUDGET_FIT:
            penalty += (MIN_BUDGET_FIT - avg_fit) * 100.0
            
        # Soft Constraint 1: Diversità tipologia immobile
        type_counts = subset['property_type'].value_counts()
        for count in type_counts:
            if count > MAX_SAME_TYPE:
                penalty += (count - MAX_SAME_TYPE) * 15.0
                
        # Soft Constraint 2: Diversità classe energetica
        energy_counts = subset['energy_class'].value_counts()
        for count in energy_counts:
            if count > MAX_SAME_ENERGY:
                penalty += (count - MAX_SAME_ENERGY) * 10.0
                
        return penalty

    n_samples = len(df)
    max_iter = 1000

    # 1. Random Walk
    def random_walk(max_iterations=1000):
        best_sol = np.random.choice(n_samples, size=SHORTLIST_SIZE, replace=False)
        best_pen = calculate_penalty(best_sol)
        
        for _ in range(max_iterations):
            cand_sol = np.random.choice(n_samples, size=SHORTLIST_SIZE, replace=False)
            cand_pen = calculate_penalty(cand_sol)
            if cand_pen < best_pen:
                best_sol, best_pen = cand_sol, cand_pen
                if best_pen == 0:
                    break
        return best_sol, best_pen

    # 2. Simulated Annealing
    def simulated_annealing(max_iterations=1000, initial_temp=1000.0, alpha=0.95):
        current_sol = np.random.choice(n_samples, size=SHORTLIST_SIZE, replace=False)
        current_pen = calculate_penalty(current_sol)
        best_sol, best_pen = current_sol, current_pen
        temp = initial_temp
        
        for _ in range(max_iterations):
            neighbor_sol = current_sol.copy()
            swap_idx = np.random.randint(0, SHORTLIST_SIZE)
            new_house = np.random.randint(0, n_samples)
            while new_house in neighbor_sol:
                new_house = np.random.randint(0, n_samples)
            neighbor_sol[swap_idx] = new_house
            
            neighbor_pen = calculate_penalty(neighbor_sol)
            delta = neighbor_pen - current_pen
            
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-5)):
                current_sol, current_pen = neighbor_sol, neighbor_pen
                if current_pen < best_pen:
                    best_sol, best_pen = current_sol, current_pen
                    
            temp *= alpha
            if best_pen == 0:
                break
                
        return best_sol, best_pen

    # Benchmark su 10 Esecuzioni Indipendenti
    N_RUNS = 10
    
    rw_times, rw_penalties = [], []
    for _ in range(N_RUNS):
        t0 = time.time()
        _, pen = random_walk(max_iter)
        rw_times.append(time.time() - t0)
        rw_penalties.append(pen)
        
    sa_times, sa_penalties = [], []
    best_sa_sol = None
    best_sa_pen = float('inf')
    for _ in range(N_RUNS):
        t0 = time.time()
        sol, pen = simulated_annealing(max_iter)
        sa_times.append(time.time() - t0)
        sa_penalties.append(pen)
        if pen < best_sa_pen:
            best_sa_sol, best_sa_pen = sol, pen

    # Esportazione della Short-List Ottimale
    df_shortlist = df.iloc[best_sa_sol].copy()
    os.makedirs(results_dir, exist_ok=True)
    shortlist_path = os.path.join(results_dir, 'csp_shortlist_optimal.csv')
    df_shortlist.to_csv(shortlist_path, index=False)
    
    print("\nPASSO 5 COMPLETATO CON SUCCESSO!")
    print("\n--- BENCHMARK CSP (10 ESECUZIONI) ---")
    print(f"[Random Walk]        Tempo medio: {np.mean(rw_times):.4f}s | Penalità media: {np.mean(rw_penalties):.2f}")
    print(f"[Simulated Annealing] Tempo medio: {np.mean(sa_times):.4f}s | Penalità media: {np.mean(sa_penalties):.2f}")
    print(f"\n - Short-list ottimale di 10 immobili salvata in: {shortlist_path}")

if __name__ == '__main__':
    run_csp_solver()