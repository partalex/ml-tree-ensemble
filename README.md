# Izveštaj: Analiza stabala odlučivanja i ensemble metoda

## 1. Uvod

Ovaj projekat obuhvata eksperimente sa stablima odlučivanja i ensemble metodama za klasifikaciju. Fokus je na
razumevanju problema underfittinga, dobrog fitovanja i overfittinga, kao i poređenju Random Forest i Gradient Boosting
algoritama.

## 2. Zadatak 1: Stabla odlučivanja i selekcija karakteristika

### 2.1. Selekcija karakteristika

Implementirane su dve metode za rangiranje karakteristika:

- **Filter metoda**: Rangiranje pomoću apsolutne Pearsonove korelacije sa ciljnom promenljivom
- **Wrapper metoda**: Rangiranje na osnovu cross-validated tačnosti logističke regresije za svaku karakteristiku
  pojedinačno

Za trening stabla odlučivanja odabrane su TOP-2 karakteristike prema korelacionom rangiranju.

### 2.2. Poređenje modela

Trenirano je tri modela stabla odlučivanja sa različitim nivoom kompleksnosti:

| Model       | `max_depth` | Karakteristike                   |
|-------------|-------------|----------------------------------|
| Underfitted | 1           | Previše jednostavan, visok bias  |
| Well-fitted | 4           | Balansirana kompleksnost         |
| Overfitted  | None        | Previše složen, visoka varijansa |

Za svaki model generisane su dve vizualizacije:

- Granice odlučivanja (*decision boundary*)
- Struktura stabla (*tree structure*)

## 3. Zadatak 2: Ensemble metode

### 3.1. Random Forest

Analiziran je uticaj broja stabala (`n_estimators`) na performanse Random Forest klasifikatora. Testirana je lista
vrednosti: [10, 50, 100, 200].

Rezultati pokazuju kako tačnost raste sa povećanjem broja stabala, uz smanjenje marginalne koristi nakon određenog
broja.

### 3.2. Gradient Boosting

Ispitan je uticaj stope učenja (`learning_rate`) na performanse Gradient Boosting klasifikatora. Testirana je lista
vrednosti: [0.01, 0.05, 0.1, 0.2].

Rezultati ilustruju trade-off između brzine konvergencije i stabilnosti modela.

### 3.3. Feature Importance

Generisan je grafik važnosti karakteristika korišćenjem Random Forest klasifikatora sa 100 stabala. Ovo pomaže u
razumevanju koje karakteristike najviše doprinose predikciji.

## 4. Struktura projekta

```text
ml-tree-ensemble/ 
├── res/
│   ├── data_1.csv
│   └── data_2.csv
├── src/
│   ├── data_utils.py
│   ├── decision_tree_experiments.py
│   ├── ensemble_experiments.py
│   ├── feature_selection.py
│   ├── shared.py
│   ├── main-1-tree.py
│   └── main-2-ensamble.py
└── out/
    ├── 01/  # Rezultati zadatka 1
    └── 02/  # Rezultati zadatka 2
    
```

## 5. Zaključak

Projekat demonstrira praktičnu primenu stabala odlučivanja i ensemble metoda, uključujući tehnike selekcije
karakteristika, razumevanje bias-variance trade-off-a i optimizaciju hiperparametara.

Svi grafici su sačuvani u `out/` direktorijumu sa visokom rezolucijom (300 DPI) radi jasne vizualizacije rezultata.