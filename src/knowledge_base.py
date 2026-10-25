import os
import sys
import pandas as pd

def build_and_query_kb():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    
    kb_csv_path = os.path.join(data_dir, 'housing_kb.csv')
    if not os.path.exists(kb_csv_path):
        print(f"\n[ERRORE] File non trovato in: {kb_csv_path}")
        sys.exit(1)
        
    df_kb = pd.read_csv(kb_csv_path)
    
    # 1. Generazione del file della Knowledge Base in sintassi Prolog (.pl)
    prolog_file_path = os.path.join(data_dir, 'housing_kb.pl')
    
    with open(prolog_file_path, 'w', encoding='utf-8') as f:
        f.write("% ===========================================\n")
        f.write("% KNOWLEDGE BASE HOUSING MATCHING (Prolog)\n")
        f.write("% ===========================================\n\n")
        
        for _, row in df_kb.iterrows():
            hid = row['house_id']
            f.write(f"house({hid}).\n")
            f.write(f"house_type({hid}, {row['property_type']}).\n")
            f.write(f"house_contract({hid}, {row['contract_type']}).\n")
            f.write(f"house_energy({hid}, {row['energy_class']}).\n")
            f.write(f"house_price({hid}, {row['monthly_price']}).\n")
            f.write(f"house_city({hid}, {row['location_city']}).\n")
            
            features = str(row['features']).split(',')
            for feat in features:
                f.write(f"house_has_feature({hid}, {feat.strip()}).\n")
            f.write("\n")
            
        # Regole Prolog di base
        f.write("% REGOLE DI ELEGIBILITA' E RACCOMANDAZIONE\n")
        f.write("eligible_price(House, MaxBudget) :- house_price(House, Price), Price =< MaxBudget.\n")
        f.write("eligible_city(House, City) :- house_city(House, City).\n")
        f.write("recommend_house(House, MaxBudget, City) :-\n")
        f.write("    house(House),\n")
        f.write("    eligible_price(House, MaxBudget),\n")
        f.write("    eligible_city(House, City).\n")

    # 2. Motore di Ragionamento Simbolico in Python (Valutazione Deterministica)
    def query_recommendations(max_budget, target_city, required_features=[]):
        recommended = []
        for _, row in df_kb.iterrows():
            # Vincolo Hard 1: Prezzo <= MaxBudget
            if row['monthly_price'] > max_budget:
                continue
            # Vincolo Hard 2: Città
            if row['location_city'].lower() != target_city.lower():
                continue
            # Vincolo Hard 3: Feature richieste presenti
            house_feats = [f.strip().lower() for f in str(row['features']).split(',')]
            missing_feats = [f for f in required_features if f.lower() not in house_feats]
            if missing_feats:
                continue
                
            recommended.append({
                'house_id': row['house_id'],
                'property_type': row['property_type'],
                'monthly_price': row['monthly_price'],
                'location_city': row['location_city'],
                'features': row['features']
            })
        return pd.DataFrame(recommended)

    # 3. Test del motore simbolico con un profilo cliente d'esempio
    sample_budget = 3500
    sample_city = 'taipei'
    sample_features = ['elevator']
    
    recs_df = query_recommendations(sample_budget, sample_city, sample_features)
    
    os.makedirs(results_dir, exist_ok=True)
    recs_csv_path = os.path.join(results_dir, 'kb_recommendations_sample.csv')
    recs_df.to_csv(recs_csv_path, index=False)
    
    print("\nPASSO 4 COMPLETATO CON SUCCESSO!")
    print(f" - Knowledge Base Prolog creata in: {prolog_file_path}")
    print(f" - Motore di ragionamento eseguito per un cliente con Budget <= {sample_budget}, Città={sample_city}, Feature={sample_features}")
    print(f" - Immobili idonei trovati: {len(recs_df)}")
    print(f" - Risultati salvati in: {recs_csv_path}")

if __name__ == '__main__':
    build_and_query_kb()