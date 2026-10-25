import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def run_unsupervised_analysis():
    # Determinazione delle cartelle di progetto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Caricamento del dataset supervisionato
    file_path = os.path.join(data_dir, 'housing_matching_supervised.csv')
    if not os.path.exists(file_path):
        print(f"\n[ERRORE] File non trovato in: {file_path}")
        print("Esegui prima lo script process_uploaded_dataset.py!")
        sys.exit(1)
        
    print(f"Caricamento dataset da: {file_path}")
    df = pd.read_csv(file_path)
    
    # Escludiamo la variabile target 'match' per l'analisi non supervisionata
    X = df.drop(columns=['match'])
    
    num_cols = ['budget_fit', 'house_age', 'unit_price', 'distance_km', 'convenience_stores']
    cat_cols = ['property_type', 'contract_type', 'energy_class']
    
    # Preprocessing: StandardScaler per numeriche + OneHotEncoder per categoriali
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop=None, sparse_output=False), cat_cols)
        ]
    )
    
    X_prep = preprocessor.fit_transform(X)
    
    # Analisi del numero di cluster (Elbow e Silhouette)
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_prep)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_prep, labels))
        
    # Salvataggio grafici di analisi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_title('Elbow Curve (KMeans) - Housing Matching')
    ax1.set_xlabel('Numero di Cluster (K)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_title('Silhouette Score (KMeans) - Housing Matching')
    ax2.set_xlabel('Numero di Cluster (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(plots_dir, 'clustering_metrics.png')
    plt.savefig(metrics_plot_path)
    plt.close()
    
    # Fit del modello finale con K=4 (segmentazione bilanciata)
    K_chosen = 4
    kmeans = KMeans(n_clusters=K_chosen, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_prep)
    
    # Visualizzazione PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_prep)
    
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.title('KMeans Clustering (PCA 2D) - Housing Matching')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster Assegnato')
    plt.grid(True)
    
    pca_plot_path = os.path.join(plots_dir, 'kmeans_pca_2d.png')
    plt.savefig(pca_plot_path)
    plt.close()
    
    # Rilevamento Anomalie (95° percentile delle distanze dai centroidi)
    distances = np.linalg.norm(X_prep - kmeans.cluster_centers_[cluster_labels], axis=1)
    threshold_95 = np.percentile(distances, 95)
    anomalies_mask = distances > threshold_95
    
    df_anomalies = df[anomalies_mask].copy()
    df_anomalies['assigned_cluster'] = cluster_labels[anomalies_mask]
    df_anomalies['distance_from_centroid'] = np.round(distances[anomalies_mask], 3)
    
    anomalies_csv_path = os.path.join(results_dir, 'anomalies_detected.csv')
    df_anomalies.to_csv(anomalies_csv_path, index=False)
    
    print("\nPASSO 2 COMPLETATO CON SUCCESSO!")
    print(f" - Grafici salvati in: {plots_dir}")
    print(f" - Cluster applicati: K={K_chosen}")
    print(f" - Anomalie rilevate (> 95° percentile, soglia={threshold_95:.3f}): {sum(anomalies_mask)} istanze")
    print(f" - CSV anomalie salvato in: {anomalies_csv_path}")

if __name__ == '__main__':
    run_unsupervised_analysis()