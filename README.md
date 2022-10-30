# Homework 2 - Alexio BEC

## Installation

En plus des bibliothèques habituelles en Machine Learning, nous avons choisi d'utiliser la bibliothèque plotly pour afficher les graphes, ce qui permet de les manipuler (par exemple tourner pour des graphes 3D) puis de les enregistrer en format png très facilement.

Pour installer la bibliothèque il faut lancer la commande **pip install plotly**

## Objectifs

A partir de données météo, prédire la température une semaine après la dernière mesure. Pour cela deux approches ont été effectuées, un modèle composé de RNN et un de CNN, pour ces deux modèles, j'ai choisi de prendre comme référence la semaine pour la forme des modèles. 

## Données

Le jeu de données de 2019 sert d'entraînement et celles de 2020 de jeu de test.

Le RNN prend en entrée des vecteurs de 7 jours et ressort les 7 jours suivants, les données d'entrainement sont donc des 7-uplets pour les inputs et les labels.

Le CNN prend en entrée les données en entier et à un noyau de 7 pour agréger les données à l'échelle de la semaine.

## Présentation des modèles

### RNN
Le modèle à base de RNN est composé d'une couche de RNN qui prend des vecteurs de taille 7 et ressort des représentations de taille _nhid_, paramètre réglable. On applique la fonction d'activation ReLU au vecteur de représentation puis on le donne à un MLP linéaire à une couche qui crée un vecteur de taille 7.

### CNN
Le modèle à base de RNN est composé d'une couche de CNN avec un noyau de taille 7 et un décalage de 1 (pour être dans les mêmes conditions que le RNN). On applique ensuite la fonction d'activation Sigmoïde au résultat, on prend ensuite le dernier élément pour prédire la semaine suivante.

J'ai essayé d'ajouter une couche linéaire, mais comme la taille de la sortie du CNN dépend de la taille de l'entrée, je ne peux pas mettre le MLP en dehors de la fonction forward et je ne suis pas sûr qu'il applique le gradient dans ce cas, la constatation que j'ai faite est que l'ajout de MLP réduit la précision.

## Choix des paramètres

Les deux modèles ont le même optimiseur, Adam.

Pour décider des paramètres des deux modèles, j'ai principalement regardé le graphe des loss d'entrainement et de test en fonction de l'époque.

### RNN

![rnn1](https://github.com/alexiobec/homework_deep_learning/blob/main/img/RNN1.png?raw=true)

J'ai d'abord choisi une couche cachée de taille 5 avec un learning rate de 0.01. On peut voir que les deux loss convergent bien mais sont très proches durant tout l'entraînement. Le modèle n'a pas assez de paramètres, j'ai donc ensuite augmenté la taille de la couche cachée à 100.

![rnn2](https://github.com/alexiobec/homework_deep_learning/blob/main/img/RNN2.png?raw=true)

Avec une couche cachée de taille 100 et un learning rate de 0.01 la convergence ne se fait qu'après 90 époques et la loss sur les données de test explose. Le modèle a cette fois trop de paramètres, j'ai décidé de réduire la couche cachée à la taille 50.

![rnn3](https://github.com/alexiobec/homework_deep_learning/blob/main/img/RNN3.png?raw=true)

Avec une couche cachée de taille 50, la convergence se fait très vite, de l'ordre de 10 époques, mais il reste un peu d'instabilité, pour régler cela, j'ai réduit le learning rate de moitié, à 0.005.

Il y a cependant une anomalie, la loss sur le jeu de test est tout le temps inférieure à celle du jeu d'entraînement. 

![rnn4](https://github.com/alexiobec/homework_deep_learning/blob/main/img/RNN4.png?raw=true)

Avec ce nouveau learning rate, la convergence est un peu plus lente mais il n'y a plus d'instabilité.

![rnn5](https://github.com/alexiobec/homework_deep_learning/blob/main/img/RNN5.png?raw=true)

Cependant, une couche cachée de taille 50 est peut-être pas nécessaire, j'ai décidé de la réduire à 30 et de garder 20 époques.

#### Paramètres du modèle RNN final :
  * Taille de la couche cachée = 30
  * Learning rate = 0.005
  * Nombre d'époques = 20

### CNN

![cnn1](https://github.com/alexiobec/homework_deep_learning/blob/main/img/CNN1.png?raw=true)

On peut d'abord observer que les deux courbes convergent bien. Cependant, elles se stabilisent à partir de 70 époques, pour augmenter la vitesse de convergence, j'ai augmenté le learning rate à 0.05.

On peut aussi observer la même anomalie la loss sur le jeu de test est inférieure à celle du jeu d'entraînement avant 85 époques.

![cnn2](https://github.com/alexiobec/homework_deep_learning/blob/main/img/CNN2.png?raw=true)

La convergence est beaucoup plus rapide mais les deux courbes fluctuent beaucoup plus, spécifiquement la loss du test. Pour remédier à cela, j'ai baissé le learning rate au double de ce qu'il était au départ, 0.02.

![cnn3](https://github.com/alexiobec/homework_deep_learning/blob/main/img/CNN3.png?raw=true)

La convergence est environ deux fois plus rapide qu'avec le learning rate à 0.01, les oscillations sur la loss d'entraînement ont disparu. En vue de ces courbes, j'ai choisi 35 époques pour le modèle final.

#### Paramètres du modèle CNN final :
  * Learning rate = 0.02
  * Nombre d'époques = 35

## Comparaison

Pour comparer les deux modèles, j'ai prédit la température du 07/04/2020 en prenant en donnée les températures allant du 01/01/2020 au 31/03/2020. La température attendue est de 21°C.

J'ai créé un sous dataset evaluation issu du jeu de test en ne prenant que les 3 premiers mois.

Les valeurs que j'obtenais sur les deux modèles variant beaucoup en fonction de l'exécution, je prends la moyenne et l'écart-type sur 100 exécutions des deux modèles. Voici leurs graphes :

![rnntot](https://github.com/alexiobec/homework_deep_learning/blob/main/img/rnntot.png?raw=true)

![cnntot](https://github.com/alexiobec/homework_deep_learning/blob/main/img/cnntot.png?raw=true)

* RNN :
  - moyenne : 18.09
  - écart-type : 5.13

* CNN :
  - moyenne : 16.80
  - écart-type : 0.74

On peut constater que le RNN est plus proche en moyenne de la bonne valeur mais a un très grand écart-type, presque 7 fois supérieur à celui du CNN. Le CNN est plus loin de la bonne valeur, cependant ses valeurs fluctuent beaucoup moins.

## Limites

Les limites de cette étude, sont le faible jeu de données, les modèles que j'ai gardé les plus simples possibles, avec pour chacun qu'une seule couche de RNN ou de CNN, les paramètres que j'ai testés indépendamment, j'en fixait un avant d'en modifier un autre.

## Conclusion

Les deux modèles sont très différents, aucun des deux n'est précis par la moyenne et l'écart-type. Cependant on peut imaginer que si on moyenne sur de très nombreux essais avec le modèle RNN, la moyenne des essais devrait se rapprocher de la valeur réelle, alors qu'avec le modèle CNN, elle ne pourrait pas être atteinte. Cependant, il faudrait améliorer les deux modèles pour pouvoir les utiliser effectivement. 
