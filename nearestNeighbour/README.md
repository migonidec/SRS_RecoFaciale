# SRS Projet de reconnaissance facile

## Index
+ **[Classificateur Nearest Neighbour](#classificateur-nearest-neighbour)**	
	- [Définition](#définition)
	- [Implémentation](#implémentation)

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
Il est possible de paramétrer l'objet `KNeighborsClassifier` avec plusieurs options, notamment avec `n_neighbors` ou `weights` et `algorithm`. Nous allons succéssivement tester ces paramètres afin de retenir la meilleure configuration. 

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

Grâce à ce tableau de comparaison on peut voir que la combinaison de paramètres la plus performante est `KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='ball_tree')`.

