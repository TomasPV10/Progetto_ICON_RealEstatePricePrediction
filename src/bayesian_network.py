import os
import sys
import numpy as np
import pandas as pd

def run_bayesian_network():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    
    file_path = os.path.join(data_dir, 'housing_matching_supervised.csv')
    if not os.path.exists(file_path):
        print(f"\n[ERRORE] File non trovato in: {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    # 1. Discretizzazione delle variabili numeriche
    df_discrete = pd.DataFrame()
    
    df_discrete['budget_fit_cat'] = pd.qcut(df['budget_fit'], q=3, labels=['basso', 'medio', 'alto'])
    df_discrete['distance_cat'] = pd.qcut(df['distance_km'], q=3, labels=['vicino', 'medio', 'lontano'])
    df_discrete['age_cat'] = pd.qcut(df['house_age'], q=3, labels=['nuova', 'media', 'datata'])
    df_discrete['price_cat'] = pd.qcut(df['unit_price'], q=3, labels=['economico', 'medio', 'costoso'])
    df_discrete['property_type'] = df['property_type']
    df_discrete['match'] = df['match']
    
    # 2. Stima delle Tabella di Probabilità Condizionata (CPT) per P(Match = 1 | Evidenze)
    group_cols = ['budget_fit_cat', 'distance_cat', 'price_cat']
    
    cpt_match = df_discrete.groupby(group_cols)['match'].agg(['count', 'mean']).reset_index()
    cpt_match.rename(columns={'mean': 'prob_match_1', 'count': 'sample_count'}, inplace=True)
    
    # Funzione di query probabilistica
    def query_bayes(budget_level, distance_level, price_level):
        match_row = cpt_match[
            (cpt_match['budget_fit_cat'] == budget_level) &
            (cpt_match['distance_cat'] == distance_level) &
            (cpt_match['price_cat'] == price_level)
        ]
        
        if not match_row.empty:
            prob = match_row['prob_match_1'].values[0]
            count = match_row['sample_count'].values[0]
            return prob, count
        else:
            # Fallback a probabilità a priori se la combinazione non è presente
            return df['match'].mean(), 0

    # 3. Test dell'inferenza con vari scenari di evidenze
    scenarios = [
        {'name': 'Scenario Ideale', 'budget': 'alto', 'dist': 'vicino', 'price': 'costoso'},
        {'name': 'Scenario Flessibile', 'budget': 'medio', 'dist': 'medio', 'price': 'medio'},
        {'name': 'Scenario Sfavorevole', 'budget': 'basso', 'dist': 'lontano', 'price': 'costoso'}
    ]
    
    bayes_results = []
    print("\n--- RISULTATI INFERENZA BAYESIANA P(Match=1 | Evidenze) ---")
    
    for sc in scenarios:
        prob, n_obs = query_bayes(sc['budget'], sc['dist'], sc['price'])
        bayes_results.append({
            'Scenario': sc['name'],
            'Budget_Fit': sc['budget'],
            'Distanza': sc['dist'],
            'Prezzo': sc['price'],
            'Probabilita_Match_1': round(prob, 4),
            'Istanze_Campione': n_obs
        })
        print(f"[{sc['name']}] P(Match=1 | Budget={sc['budget']}, Distanza={sc['dist']}, Prezzo={sc['price']}) = {prob:.2%} (campioni: {n_obs})")
        
    df_bayes = pd.DataFrame(bayes_results)
    os.makedirs(results_dir, exist_ok=True)
    bayes_csv_path = os.path.join(results_dir, 'bayesian_inference_results.csv')
    df_bayes.to_csv(bayes_csv_path, index=False)
    
    print("\nPASSO 6 COMPLETATO CON SUCCESSO!")
    print(f" - Tabella CPT e risultati dell'inferenza salvati in: {bayes_csv_path}")

if __name__ == '__main__':
    run_bayesian_network()