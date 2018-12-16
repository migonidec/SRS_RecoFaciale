# SRS Projet de reconnaissance facile

Dans le cadre de l'UV SRS à l'IMT Lille Douai (ex-Telecom Lille), nous avons eu l'occasion de nous initier à l'analyse biométrique basée sur l'IA.
Nous allons utiliser le code fourni par la bibliothèque [scikit-learn](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py) et nous allons analyser son fonctionnement afin de l'améliorer.

## TODO
- [x] Structure globale du code 
- [x] Variation de la proportion de test 
- [x] Variation du nombre de composantes
- [x] Variation du kernel SVM
- [ ] Implementation de courbe ROC
- [ ] Implementation de courbe rappel-précision
- [x] K Nearest Neighbour 
- [x] Random Forest

## Index
+ **[Analyse de RecoFaciale](recoFaciale)**
+ **[Classificateur Nearest Neighbour](nearestNeighbour)**	
+ **[Classificateur Random Forest](randomForest)**	

## Documentation
+ https://scikit-learn.org/stable/modules/grid_search.html
+ https://en.wikibooks.org/wiki/Support_Vector_Machines
+ https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
