# SRS Projet de reconnaissance facile

Dans le cadre de l'UV SRS à l'IMT Lille Douai (ex Telecom Lille), nous avons eu l'occasion de nous inicier à l'analyse biométrique basée sur l'IA.
Nous allons utiliser le code fourni par la biobliotèque [scikit-learn](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py) et nous allons analyser son fonctionnement afin de l'améliorer.

## TODO
- [x] Structure globale du code 
- [x] Variation de la proportion de test 
- [x] Variation du nombre de composantes
- [x] Variation du kernel SVM
- [ ] Implementation de courbe ROC
- [ ] Implementation de courbe rappel-précision
- [ ] Nearest Neighbour définition
- [x] Implémentation KNN
- [ ] Utilisation Random Forest

## Index
+ **[Analyse de RecoFaciale](#analyse-de-RecoFaciale)**
	- [Recupération des données](#recupération-des-données) 
	- [Prétraitement des données](#prétraitement-des-données)
	- [Classificateur](#classificateur)
		- [Définition de SVM](#définition-de-SVM)
		- [Implémentation](#implémentation)
	- [Test de validation](#test-de-validation)
+ **[Classificateur Nearest Neighbour](#classificateur-nearest-neighbour)**	
	- [Définition](#définition)
	- [Implémentation](#implémentation)


## Analyse de RecoFaciale

### Workflow général

![alt text](https://raw.githubusercontent.com/migonidec/SRS_RecoFaciale/master/images/global_scheme.png)

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

On peux remarquer le nombre de filtre joue fortement sur la précision du model. Cependant, à partir d'un certain nombre les performance baissent fortement. En effet, théoriquement les premières eigenfaces portent les informations principales puis la quantité d'information décrois. On peux donc assimiler les eigenfaces d'indice elevé comme du bruit pour notre analyse.
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
Le principe de cette méthode est l’apprentissage par analogie, celui-ci peut-être comparé à la citation : « Dis moi qui sont tes amis, et je te dirais qui tu es ». 
En outre, le but est de rechercher un ou plusieurs cas déjà similaires et résolus, puisque le nous nous basons sur un ensembles de données connues (training data). Pour cela, il faut classifier et estimer ce cas. Pour cela, un algorithme ayant les principes suivants peut -être utilisé :

**- Paramètre** : le nombre k de voisins

**- Donnée** : un échantillon de m exemples  et leurs classes

**- Entrée** : un enregistrement Y
1. Déterminer les k plus proches exemples de Y en calculant les distances
2. Classification et estimation du résultat 
Sortie : Le résultat anticipé

Le choix de la distance calculer est primordial au bon fonctionnement de la méthode. Mais alors, comment choisir la méthode des calculs de distance?Cela dépend forcément du type d’analyse, puisque nous avons le choix entre la:

**- Distance numérique**
**- Distance entre nominaux**
**- Distance Euclidienne**
**- Diverses**

Dans notre cas, on utilisera la distance euclidienne entre les points. Nous présenterons un visage de test et observerons les plus proches visage de tests. Puis, nous allons comptabiliser la majorité des points afin d’en déduire la classe du visage en question.

Il faut savoir que d’autres normes peuvent être utilisées, avec chacun un intérêts différents. Il nous faudra donc choisir le bon nombre de voisins proches ( K ) afin d’avoir un taux d’erreurs acceptables ou étant dans nos critères. 

Voici un exemple d’un calcul des proches voisins via la méthode d’une distance Euclidienne.
Le but ici, est de prédire si le client va acheter ou non un mac dans une boutique en lui posant 3 questions.

|Nom du client | Age  | Salaire |  Nombre de produits Apple |  Achete un MAC ? |
| ----------- | -------- | --------- | ----------------- |------------------ |
|Monny|35|35K|3|Non| 
|Fahima|22|50K|2|Oui|
|Amine|63|200K|1|Non|
|Fabien|59|170K|1|Non|
|Guillaume|25|40K|4|Oui|
|Valentin|37|50K|2|**?**|

Une fois ces informations obtenues, nous pouvons commencer à calculer les distances eucliennes en fixant notre nombres de voisins à K=3. Voici un exemple pour la distance Euclidienne de Monny.

|Nom du client | Age  | Salaire |  Nombre de produits Apple |  Achete un MAC ? | Distance Euclidienne|
| ----------- | -------- | --------- | ----------------- |------------------ |---------------- |
|Monny|35|35K|3|Non| √[(35-37)²+(35-50)²+(3-2)²]=15,16|
|Valentin|37|50K|2|**?**||

En répétant l'exercice pour chacun d'entre eux, les proches plus proches voisins sont donc Monny, Fahima, Guillaume. Leurs résultats pour la réponse achète un MAC sont : Non, Oui, Oui. Nous disposons donc de deux oui, un non. La majorité est oui, Valentin sera donc placé dans la catégorie d’acheteur de MAC.

|Nom du client | Age  | Salaire |  Nombre de produits Apple |  Achete un MAC ? |
| ----------- | -------- | --------- | ----------------- |------------------ |
|Monny|35|35K|3|Non| 
|Fahima|22|50K|2|Oui|
|Amine|63|200K|1|Non|
|Fabien|59|170K|1|Non|
|Guillaume|25|40K|4|Oui|
|Valentin|37|50K|2|**OUI**|

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


## Classificateur Random Forest

### Définition
Afin de bien comprendre l'utilisation de l'algorithme Random Forest, il est nécéssaire de connaitre la notion d'arbre binaire de décision ([doc](#https://perso.univ-rennes1.fr/valerie.monbet/doc/cours/IntroDM/Chapitre6.pdf). En effet, l'algorithme Random Forest utlise un multitude d'arbres binaires de décision pour fonctionner.

Ainsi chaque arbre binaire va permettre d'analyser une composante de notre input. En utilisant l'exemple fourni dans l'explication de l'algorithme Nearest Neighbour, on peux assimiler chaque arbre de décision à une colonne du tableau. 
L'input va donc être soumise à chaque arbre de décision de la forêt, chaque arbre va pouvoir donner une conclusion sur la caractèristique recherchée. En faisant la moyenne des décisions des arbres, l'algorithme va sortir un prédiction sur l'input. 

De cette explication, on peux extraire 2 paramètres majeurs de ce classificateur :
* Le nombre d'arbre de décision utilisé : `n_estimators`
* La taille maximal des arbres de décision : `max_depth`

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

On peux voir que la combinaison la plus pertinente dans notre cas d'usage est le suivant : `RandomForestClassifier(n_estimators=100, max_depth=15)`.
On remarque que les performances optimales de l'algorithme sont visibles lorsque `n_estimators = n_components`. De plus, la limitation de la profondeur des arbres permet d'éviter la prise de décision parasites.

## Documentation
+ https://scikit-learn.org/stable/modules/grid_search.html
+ https://en.wikibooks.org/wiki/Support_Vector_Machines
+ https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


