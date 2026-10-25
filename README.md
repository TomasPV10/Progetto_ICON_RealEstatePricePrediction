# Real Estate Housing Matching System  
Progetto di Ingegneria della Conoscenza (A.A. 2025-2026)

**Università degli Studi di Bari Aldo Moro**

##  Gruppo di lavoro
- **Tommaso Pio**

---

##  Descrizione del Progetto
Questo progetto implementa un Knowledge-Based System (KBS) ibrido per modellare ed analizzare il problema dell'**Housing Matching** (valutazione della compatibilità ed elegibilità tra acquirente/affittuario e immobile residenziale).

Il sistema integra 5 moduli distinti:
- **Apprendimento non supervisionato:** Clustering K-Means, PCA 2D e rilevamento anomalie oltre il 95° percentile per la segmentazione del mercato immobiliare.
- **Apprendimento supervisionato:** Model Zoo (Gradient Boosting, Random Forest, Decision Tree, SVM RBF, KNN) integrato in una Pipeline scikit-learn con GridSearchCV e 5-Fold Stratified Cross-Validation.
- **Knowledge Base (KB):** Rappresentazione simbolica dichiarativa in formato Prolog (`housing_kb.pl`) e motore di raccomandazione deterministico basato su regole ed elegibilità.
- **Constraint Satisfaction Problem (CSP):** Generazione di una shortlist ottimale di 10 immobili tramite algoritmi di ricerca locale e metaeuristiche (Random Walk vs Simulated Annealing).
- **Rete Bayesiana:** Modello probabilistico discreto per il calcolo delle probabilità posteriori $P(\text{Match}=1 \mid \text{Evidenze})$ in regime d'incertezza.

---

## 🛠️ Requisiti e Installazione

### Requisiti
- Python **3.9 o superiore**
- `pip` aggiornato

### 1. Creazione e attivazione dell'ambiente virtuale
Dalla cartella principale del progetto:

```bash
python -m venv venv