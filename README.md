# SRS Projet de reconnaissance facile

Dans le cadre de l'UV SRS à l'IMT Lille Douai (ex Telecom Lille), nous avons eu l'occasion de nous inicier à l'analyse biométrique basée sur l'IA.
Nous allons utiliser le code fourni par la biobliotèque [scikit-learn](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py) et nous allons analyser son fonctionnement afin de l'améliorer.

## Index
1. **[Analyse du fonctionnement](#analyse-du-fonctionnement)**
1.1 [Recupération des données](#recupération-des-données)
1.2 [Prétraitement des données](#Prétraitement-des-données)
1.3 [Classificateur](#Classificateur)

## Définition de SVM
Les machines à vecteurs de support sont des techniques d'apprentissage supervisé qui permettent de classifier des données de grandes dimensions. Les SVMs sont donc adaptés pour traiter des données tel que les visages.
Le fonctionnement des SVM est défini en fonction de 2 paramètres : un noyau (*kernel*) et une marge maximale (*soft margin*).
* Kernel. Cet élément permet de résoudre des problèmes linéairement non séparables. Il semble qu'il n'existe pas de méthode deterministe pour choisir un kernel, cependant il est possible de determiner le bon kernel de manière empirique (cross-validation). 
Enfin, le kernel peut être configuré en utlisant un paramètre &gamma;. Ce paramètre sera configuré de manière empirique, cependant il est de fournir un ensemble de valeurs croissantes exponentiellement ([doc](https://en.wikibooks.org/wiki/Support_Vector_Machines))
* Soft Margin. De la même manière que &gamma;, il est conseillé de fournir un ensembles de valeurs croissantes exponentiellement.



## Analyse du fonctionnement

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

#### Support Vector Machines (SVM)
Voici le code remanié du code origial, cela pour plus de clareté
```
K_param = 'rbf'
G_param = np.logspace(0, 1, 100)
C_param = np.logspace(0, 1, 100)

param_grid = {'C': C_param, 'gamma': G_param, }
svc = SVC(kernel=K_param, probability=True)
clf = GridSearchCV(svc, param_grid)
clf = clf.fit(X_train_pca, y_train)
```
On voit bien les 3 paramètres caractèristiques d'une SVM apparaitre dans ce code. La librairie sklearn nous premet ensuite de créer un objet SVM et de l'adapter à notre dataset de test.
En regardant la documentation, on peux voir que les paramètres `G_param` et `C_param` seront séléctionnées par cross-validation. Il est possible de les afficher grâce à `clf.best_params_`.

Par la suite à fera varier ces paramètres, et notamment `K_param` afin d'étudier le fonctionnement de SVM.


## Documentation
https://en.wikibooks.org/wiki/Support_Vector_Machines
https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


