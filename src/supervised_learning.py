import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def run_supervised_learning():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    plots_dir = os.path.join(results_dir, 'plots_supervised')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'housing_matching_supervised.csv')
    if not os.path.exists(file_path):
        print(f"\n[ERRORE] File non trovato in: {file_path}")
        sys.exit(1)
        
    print(f"Caricamento dataset da: {file_path}")
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=['match'])
    y = df['match']
    
    num_cols = ['budget_fit', 'house_age', 'unit_price', 'distance_km', 'convenience_stores']
    cat_cols = ['property_type', 'contract_type', 'energy_class']
    
    # 1. Train / Test Split (70% train, 30% test) con stratificazione
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # 2. Definiamo il Model Zoo e le relative griglie di iperparametri
    models_config = {
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 5, 10],
                'clf__min_samples_split': [2, 5]
            }
        ),
        'SVM_RBF': (
            SVC(probability=True, random_state=42),
            {
                'clf__C': [0.5, 1, 5],
                'clf__gamma': ['scale', 0.1, 0.01]
            }
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42),
            {
                'clf__max_depth': [None, 5, 10],
                'clf__min_samples_split': [2, 5, 10]
            }
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'clf__n_estimators': [100, 150],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [2, 3]
            }
        ),
        'KNN': (
            KNeighborsClassifier(),
            {
                'clf__n_neighbors': [5, 11, 15],
                'clf__weights': ['uniform', 'distance']
            }
        )
    }
    
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    summary_results = []
    
    plt.figure(figsize=(9, 7))
    
    print("\n--- INIZIO ADDESTRAMENTO E GRID SEARCH ---")
    
    for model_name, (clf, param_grid) in models_config.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', clf)
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv_stratified,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predizioni sul Test Set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calcolo Metriche
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        summary_results.append({
            'Model': model_name,
            'Best_CV_ROC_AUC': round(grid_search.best_score_, 4),
            'Test_Accuracy': round(acc, 4),
            'Test_ROC_AUC': round(roc_auc, 4),
            'Test_PR_AUC': round(ap, 4),
            'Best_Params': grid_search.best_params_
        })
        
        # Tracciamento Curva ROC per il confronto
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        
        print(f"\n[+] Modello: {model_name}")
        print(f"    Migliori Iperparametri: {grid_search.best_params_}")
        print(f"    Test Accuracy: {acc:.4f} | Test ROC-AUC: {roc_auc:.4f} | Test PR-AUC: {ap:.4f}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Casuale (AUC = 0.500)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Confronto Curve ROC - Housing Matching')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    roc_plot_path = os.path.join(plots_dir, 'roc_curves_comparison.png')
    plt.savefig(roc_plot_path)
    plt.close()
    
    # Salvataggio report riepilogativo in CSV
    df_summary = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(results_dir, 'supervised_results.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    
    print("\nPASSO 3 COMPLETATO CON SUCCESSO!")
    print(f" - Report riepilogativo salvato in: {summary_csv_path}")
    print(f" - Grafico confronto ROC salvato in: {roc_plot_path}")

if __name__ == '__main__':
    run_supervised_learning()