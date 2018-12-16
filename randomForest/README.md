# [TEST] SRS Projet de reconnaissance facile

## Index
+ **[Classificateur Random Forest](#classificateur-random-forest)**	
	- [Définition](#définition)
	- [Implémentation](#implémentation)

## Classificateur Random Forest

### Définition
Afin de bien comprendre l'utilisation de l'algorithme Random Forest, il est nécéssaire de connaitre la notion d'arbre binaire de décision ([doc](#https://perso.univ-rennes1.fr/valerie.monbet/doc/cours/IntroDM/Chapitre6.pdf). En effet, l'algorithme Random Forest utilise une multitude d'arbres binaires de décision pour fonctionner.

Ainsi chaque arbre binaire va permettre d'analyser une composante de notre input. En utilisant l'exemple fourni dans l'explication de l'algorithme Nearest Neighbour, on peut assimiler chaque arbre de décision à une colonne du tableau. 
L'input va donc être soumise à chaque arbre de décision de la forêt, chaque arbre va pouvoir donner une conclusion sur la caractéristique recherchée. En faisant la moyenne des décisions des arbres, l'algorithme va sortir un prédiction sur l'input. 

De cette explication, on peut extraire 2 paramètres majeurs de ce classificateur :
* Le nombre d'arbre de décision utilisés : `n_estimators`
* La taille maximale des arbres de décision : `max_depth`

### Implémentation
```
clf = RandomForestClassifier(n_estimators=1, max_depth=2)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print("Average accuracy %0.3f" % accuracy_score(y_test, y_pred))
```
Nous allons modifier les 2 paramètres `n_estimators` et `max_depth` afin de determiner la meilleure combinaison dans notre cas. On concerve les autres paramètres de test constant, c'est à dire `n_components = 100` et `test % = 18`.

| n_estimators | max_depth |  Average accuracy |
| ------------ | --------- | ----------------- |
| 50           | 10        | 0.438             |
| 50           | 15        | 0.465             |
| 50           | 20        | 0.427             |
| 100          | 10        | 0.435             |
| 100          | 15        | 0.470             |
| 100          | 20        | 0.465             |
| 200          | 10        | 0.438             |
| 200          | 15        | 0.459             |
| 200          | 30        | 0.462             |

On peut voir que la combinaison la plus pertinente dans notre cas d'usage est le suivant : `RandomForestClassifier(n_estimators=100, max_depth=15)`.
On remarque que les performances optimales de l'algorithme sont visibles lorsque `n_estimators = n_components`. De plus, la limitation de la profondeur des arbres permet d'éviter la prise de décision parasites.

