# SRS Projet de reconnaissance facile

Dans le cadre de l'UV SRS à l'IMT Lille Douai (ex Telecom Lille), nous avons eu l'occasion de nous inicier à l'analyse biométrique basée sur l'IA.
Nous allons utiliser le code fourni par la biobliotèque [scikit-learn](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py) et nous allons analyser son fonctionnement afin de l'améliorer.

## TODO
- [ ] Implementation de courbe ROC
- [ ] Implementation de courbe rappel-précision
- [ ] Nearest Neighbour définition
- [ ] Utilisation Random Forest

## Index
+ **[Analyse de RecoFaciale](#analyse-de-RecoFaciale)**
	- [Recupération des données](#recupération-des-données) 
	- [Prétraitement des données](#prétraitement-des-données)
	- [Classificateur](#classificateur)
		- [Définition de SVM](#définition-de-SVM)
		- [Implémentation](#implémentation)
	- [Test de validation](#test-de-validation)
+ **[Classificateur Nearest Neighbour](#classificateur-nearest-neighbour)
	- [Définition](#définition)
	- [Implémentation](#implémentation)


## Analyse de RecoFaciale

### Recupération des données
La première étape du programme est la récupération des données. Ces données proviennent de la base données [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).
Lors de l'importation des données, il est possible de jouer sur 2 paramètres principaux : `min_faces_per_person` et `resize`. 
#### Nombre de visages par classes
```
lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.9, data_home=".")
```

| Faces per person | # of personns | n_components |  Average accuracy | Time (s) |
| ---------------- | ------------- | -------------| ----------------- | -------- |
| 10               | 158           | 200          | 0.480             | 4553     |
| 35               | 24            | 200          | 0.743             | 717      |
| 60               | 8             | 200          | 0.828             | 203      |

Naturellement, on remarque que *plus le nombre d'images par classe est élevé, plus la précision de la prédiction est bonne*. 

#### Redimentionnement des images sources
D'après nos tests, le redimmentionnement des images n'a pas beaucoup d'influence sur la précision de notre prédiction. Nous allons donc conserver le paramètre basique de 0,9. 
 
 
Les étapes suivantes sont destinées à extraires les caractèristiques du data-set que nous avons séléctionnées.
```
n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
``` 

### Prétraitement des données
Les données vont ensuite subire une série de prétraitement avat d'être transmise aux algorithmes d'apprentissage.
On va premièrement séparer notre ensemble en 2 : une sample de d'entrainement et un sample de test. Ce dernier permettra de valider les performance de notre entrainement. 
On définit ensuite un nombre de composantes (eigenfaces) à extraire de nos classes pour pouvoir entrainer notre modèle.

#### Proportion de test
Par défault, la fonction `sklearn.model_selection.train_test_split` mélange l'ensemble du dataset avant de le séparer. Nous allons garder ce mélange en conservant la même graine aléatoire entre chaque essai.

| Faces per person | n_components | test %      | Average accuracy | Time (s) |
| ---------------- | ------------ | ----------- | ---------------- | -------- |
| 35               | 200          | 10          | 0.782            | 1028     |
| 35               | 200          | 18          | 0.773            | !460!    |
| 35               | 200          | 25          | 0.728            | 606      |
| 35               | 200          | 40          | 0.702            | 571      |

Naturellement, on peux voir que plus la proportion de test augmente moins le model produit sera précis. En effet, cela donne au moteur d'entrainement moins de données pour travailer.
Il est donc nécéssaire de trouver un juste milieu entre le temps d'apprentissage et l'efficacité des modèles.
On fixe le pourcentage de test à 18 `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)`

#### Nombre de composantes
Le nombre de composantes correspond au nombre de filtre (eigenfaces) que nous allons appliquer pour prédire l'identité d'une personne. Nous allons faire varier ce nombre de filtres afin d'évaluer leur impact.

| Faces per person | n_components | test %      | Average accuracy | Time (s) |
| ---------------- | ------------ | ----------- | ---------------- | -------- |
| 35               | 50           | 18          | 0.722            | 102      |
| 35               | 100          | 18          | 0.789            | 229      |
| 35               | 150          | 18          | 0.751            | 342      |
| 35               | 200          | 18          | 0.768            | 506      |
| 35               | 250          | 18          | 0.746            | 625      |

On peux remarquer le nombre de filtre joue fortement sur la précision du model. Cependant, à partir d'un certain nombre les performance baissent fortement.
On fixera un nombre de eigenfaces à 100 pour des questions de performance temporelles et de précision : `n_components = 100`.

#### Reduction de dimension
Grâce à la librairie `sklearn`, on peut facilement générer des eigenfaces et les adapter au dataset d'entrainement.
```
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True) #model creation
pca.fit(X_train) #fit model to the training dataset
```
L'analyse en composante principale (PCA) va permettre de diminuer le nombre de dimension d'analyse au nombre d'eigenfaces. Il va donc falloir adapter nos 2 datasets (train et test) à nos PCA.
```
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

### Classificateur

#### Définition de SVM
Les machines à vecteurs de support sont des techniques d'apprentissage supervisé qui permettent de classifier des données de grandes dimensions. Les SVMs sont donc adaptés pour traiter des données tel que les visages.
Le fonctionnement des SVM est défini en fonction de 2 paramètres : un noyau (*kernel*) et une marge maximale (*soft margin*).
* Kernel. Cet élément permet de résoudre des problèmes linéairement non séparables. Il semble qu'il n'existe pas de méthode deterministe pour choisir un kernel, cependant il est possible de determiner le bon kernel de manière empirique (cross-validation). 
Enfin, le kernel peut être configuré en utlisant un paramètre &gamma;. Ce paramètre sera configuré de manière empirique, cependant il est de fournir un ensemble de valeurs croissantes exponentiellement ([doc](https://en.wikibooks.org/wiki/Support_Vector_Machines))
* Soft Margin. De la même manière que &gamma;, il est conseillé de fournir un ensembles de valeurs croissantes exponentiellement.

#### Implémentation
Voici le code remanié du code original, cela pour plus de clareté
```
K_param = 'rbf'
G_param = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5]
C_param = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

param_grid = {'C': C_param, 'gamma': G_param, }
svc = SVC(kernel=K_param, probability=True)
clf = GridSearchCV(svc, param_grid)
clf = clf.fit(X_train_pca, y_train)
```
On voit bien les 3 paramètres caractèristiques d'une SVM apparaitre dans ce code. La librairie sklearn nous premet ensuite de créer un objet SVM et de l'adapter à notre dataset de test.
En regardant la documentation, on peux voir que les paramètres `G_param` et `C_param` seront séléctionnées par cross-validation. Il est possible de les afficher grâce à `clf.best_params_`.

Par la suite à fera varier ces paramètres, et notamment `K_param` afin d'étudier le fonctionnement de SVM.

##### Lineaire
Ce kernel est le plus simple, c'est donc le moins consommateur en ressource.

| Kernel | Faces per person | n_components | test %      | Average accuracy | Time (s) |
| ------ | ---------------- | ------------ | ----------- | ---------------- | -------- |
| linear | 35               | 50           | 18          | 0.651            | 1286     |
| linear | 35               | 100          | 18          | 0.700            | 219      |
| linear | 35               | 150          | 18          | 0.714            | 467      |
| linear | 35               | 200          | 18          | 0.735            | 601      |

Cependant on remarque que lorsque le kernel linéaire travail avec peu de composantes, les performances sont très largement dégradées.

##### Polynomial

| Kernel | Faces per person | n_components | test %      | Average accuracy | Time (s) |
| ------ | ---------------- | ------------ | ----------- | ---------------- | -------- |
| poly   | 35               | 100          | 18          | 0.473            | 265      |
| poly   | 35               | 150          | 18          | 0.378            | 442      |
| poly   | 35               | 200          | 18          | 0.319            | 854       |

Pour une raison inconue, nous n'arrivons pas à obtenir des resultats probant avec ce noyau. Même en jouant sur 2 facteurs optionnels (`coef0` et `degree`), il nous est impossible de dépasser une précision de 0,7.
De plus, on remarque que les performances diminuent avec le nombre de filtres appliqués. On peux conclure que ce kernel n'est pas l'option adaptée pour ce cas d'usage.

##### Radial Basis Function

| Kernel | Faces per person | n_components | test %      | Average accuracy | Time (s) |
| ------ | ---------------- | ------------ | ----------- | ---------------- | -------- |
| rbf    | 35               | 50           | 18          | 0.722            | 102      |
| rbf    | 35               | 100          | 18          | 0.789            | 229      |
| rbf    | 35               | 150          | 18          | 0.751            | 342      |
| rbf    | 35               | 200          | 18          | 0.768            | 506      |

Pour notre cas d'usage on peux voir que c'est le noyau `rbf` qui obtiens les meilleurs performances.

### Test de validation
Pour évaluer la pertinence de notre modèle, on utilise une matrice de confusion grâce à la fonction `confusion_matrix`. A noter que la fonction `classification_report` permet d'extraire des métriques plus compactes que la matrice de confusion, mais que toutes les données sont explicités dans la matrice.

Dans notre cas il est pertinent d'implémenter une courbe ROC et une courbe rappel-précision. 

## Classificateur Nearest Neighbour

### Définition


### Implémentation
```
neigh = KNeighborsClassifier(n_neighbors=10)
clf = neigh.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print("Average accuracy %0.3f" % accuracy_score(y_test, y_pred))
```
Il est possible de paramétrer l'object `KNeighborsClassifier` avec plusieurs options, notamment avec `n_neighbors` ou `weights` et `algorithm`. Nous allons succéssivement tester ces paramètres afin de retenir la meilleure configuration. 

| n_neighbors | weights  | algorithm |  Average accuracy |
| ----------- | -------- | --------- | ----------------- |
| 3           | default  | default   | 0.530             |
| 5           | default  | default   | 0.565             |
| 10          | default  | default   | 0.543             |
| 50          | default  | default   | 0.373             |
| 5           | uniform  | default   | 0.570             |
| 5           | distance | default   | 0.580             |
| 5           | distance | ball_tree | 0.581             |
| 5           | distance | kd_tree   | 0.573             |
| 5           | distance | brute     | 0.584             |

Grâce à ce tableau de comparaison on peux voir que la combinaison de paramètres la plus performante est `KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='ball_tree')`.

## Documentation
+ https://scikit-learn.org/stable/modules/grid_search.html
+ https://en.wikibooks.org/wiki/Support_Vector_Machines
+ https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


