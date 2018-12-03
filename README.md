# SRS Projet de reconnaissance facile

Dans le cadre de l'UV SRS à l'IMT Lille Douai (ex Telecom Lille), nous avons eu l'occasion de nous inicier à l'analyse biométrique basée sur l'IA.
Nous allons utiliser le code fourni par la biobliotèque [scikit-learn](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py) et nous allons analyser son fonctionnement afin de l'améliorer.


## Analyse du fonctionnement

### Recupération des données
La première étape du programme est la récupération des données. Ces données proviennent de la base données [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).
Lors de l'importation des données, il est possible de jouer sur 2 paramètres principaux : `min_faces_per_person` et `resize`. 
#### Nombre de visages par classes
```
lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.9, data_home=".")
```

| Faces per person| # of personns | n_components |  Average accuracy | Time (s) |
| --------------- | ------------- | -------------| ----------------- | -------- |
| 10              | 158           | 100          | 0.480             | 4553     |
| 35              | 24            | 100          | 0.743             | 717      |
| 60              | 8             | 100          | 0.828             | 203      |

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
On va premièrement séparer notre ensemble en 2 : une sample de d'entraiement et un sample de test. Ce dernier permettra de valider les performance de notre entrainement. 
On définit ensuite un nombre de composantes (eigenfaces) à extraire de nos classes pour pouvoir entrainer notre modèle.



