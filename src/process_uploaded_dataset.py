import os
import sys
import numpy as np
import pandas as pd

def process_data():
    # Determinazione delle cartelle di progetto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    
    # Percorso preciso del file
    file_path = os.path.join(data_dir, 'real_estate.csv')
    
    # Fallback su altri percorsi se non lo trova esattamente lì
    if not os.path.exists(file_path):
        alternative_paths = [
            os.path.join(data_dir, 'Real estate.csv'),
            os.path.join(parent_dir, 'real_estate.csv'),
            'data/real_estate.csv',
            'data/Real estate.csv'
        ]
        for alt in alternative_paths:
            if os.path.exists(alt):
                file_path = alt
                break

    if not os.path.exists(file_path):
        print(f"\n[ERRORE] Impossibile trovare 'real_estate.csv' in: {file_path}")
        sys.exit(1)

    print(f"Caricamento dati da: {file_path}")
    df = pd.read_csv(file_path)

    # Rinomina colonne originarie
    df = df.rename(columns={
        'X2 house age': 'house_age',
        'X3 distance to the nearest MRT station': 'distance_mrt_m',
        'X4 number of convenience stores': 'convenience_stores',
        'Y house price of unit area': 'unit_price'
    })

    np.random.seed(42)
    n_samples = len(df)

    # Converti distanza in km
    df['distance_km'] = np.round(df['distance_mrt_m'] / 1000.0, 2)

    # Mappatura tipologia immobile in base al prezzo
    df['property_type'] = np.where(df['unit_price'] > 45, 'villa',
                          np.where(df['unit_price'] > 35, 'apartment',
                          np.where(df['unit_price'] > 25, 'loft', 'studio')))

    df['contract_type'] = np.random.choice(['sale', 'rent'], size=n_samples, p=[0.5, 0.5])
    df['energy_class'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples, p=[0.25, 0.35, 0.25, 0.15])

    # Calcolo dello score e della variabile target 'match' (0 o 1)
    target_score = (
        (df['unit_price'] / df['unit_price'].max()) * 2.0
        - (df['distance_km'] / df['distance_km'].max()) * 2.5
        + (df['convenience_stores'] / 10.0) * 1.5
        - (df['house_age'] / 45.0) * 1.0
        + np.random.normal(0, 0.3, n_samples)
    )

    df['budget_fit'] = np.round(np.clip((target_score - target_score.min()) / (target_score.max() - target_score.min()), 0.1, 1.0), 2)
    df['match'] = np.where(target_score > target_score.median(), 1, 0)

    # Assicuriamo la creazione della cartella data/
    os.makedirs(data_dir, exist_ok=True)

    # Esportazione Dataset Supervisionato (414 istanze)
    df_supervised = df[[
        'budget_fit', 'house_age', 'unit_price', 'distance_km',
        'convenience_stores', 'property_type', 'contract_type', 'energy_class', 'match'
    ]]
    df_supervised.to_csv(os.path.join(data_dir, 'housing_matching_supervised.csv'), index=False)

    # Esportazione Dataset per la Knowledge Base (100 immobili)
    cities = ['taipei', 'new_taipei', 'taichung', 'kaohsiung']
    features_list = ['wifi', 'elevator', 'balcony', 'parking', 'air_conditioning']

    kb_rows = []
    for i in range(100):
        row = df.iloc[i]
        k_feats = np.random.randint(2, 4)
        req_feats = ",".join(np.random.choice(features_list, size=k_feats, replace=False))
        
        kb_rows.append({
            'house_id': f"house_{i+1}",
            'property_type': row['property_type'],
            'contract_type': row['contract_type'],
            'energy_class': row['energy_class'],
            'convenience_stores': int(row['convenience_stores']),
            'monthly_price': int(row['unit_price'] * 100),
            'location_city': np.random.choice(cities),
            'features': req_feats
        })

    df_kb = pd.DataFrame(kb_rows)
    df_kb.to_csv(os.path.join(data_dir, 'housing_kb.csv'), index=False)

    print("\nPASSO 1 COMPLETATO CON SUCCESSO!")
    print(f" - Creato {os.path.join(data_dir, 'housing_matching_supervised.csv')} (414 righe)")
    print(f" - Creato {os.path.join(data_dir, 'housing_kb.csv')} (100 righe)")

if __name__ == '__main__':
    process_data()