# Reconstruction d'écoulements

## Table des Matières

- [Description](#description)
- [Fichiers](#fichiers)
- [Utilisation](#utilisation)

## Description

Ce dossier montre deux exemples d'utilisation du code pour tenter de reconstruire un écoulement avec des données manquantes. Un premier exemple en utilisant les PINNs et un deuxième sans les utiliser. Il montre bien que les PINNs permettent de faire des choses que le machine learning classique ne peut pas faire.

Voilà un exemple du nombre de points que l'on prend pour un pas de temps :

![Scatter Plot des Données](./results/1_reconstruction_avec_pinns/points_plot.png)

### Reconstruction avec les PINNs :

![Vidéo de la reconstruction](./results/1_reconstruction_avec_pinns/velocity_norm.gif)


### Reconstruction sans les PINNs :
![Vidéo de la reconstruction](./results/2_reconstruction_sans_pinns/velocity_norm.gif)


## Fichiers

- `main` : Il permet de lancer le fichier, on peut rentrer les hyperparamètres de nos simulations dans ce fichier.
- `model.py` : Il contient la construction des différentes architectures de réseaux de neurones ainsi que la fonction qui permet de calculer les résidus de l'équation différentielle
- `geometry.py` : Il contient la classe `RectangleWithoutCylinder` qui permet d'avoir des points de l'espace (x,y,t) afin de tester les résidus de l'EDP sur ces valeurs
- `constants.py` : Pour stocker des constantes dont on pourrait se servir
- `onyxia-init.sh`: Le fichier qu'on met dans onyxia pour préinstaller les différents modules
- `run.py` : Le fichier avec une classe qui permet de lancer les simulations. Il permet de faire le lien entre tous les fichiers.
- `utils.py` : Pour charger les données sur lesquelles on va entraîner notre modèle, permet aussi de charger le modèle.
- `train.py` : C'est ici que l'on va faire l'optimisation et nos différentes epochs
- `results` : Chaque dossier représente une simulation différente, dans chaque dossier on retrouve le modèle aux différentes epochs ainsi que les plots qui ont été réalisés sur le modèle entraîné
- `data`: dossier à créer pour mettre les données des modèles dans le domaine ainsi que sur les bords des domaines (me contacter ou demander à John Redford pour y avoir accès.)
- `results_Cl` : dossier ou l'on a les résultats attendus de portance, pour comparer les résultats de nos modèles. On obtient ces résultats avec le code de John Redford
- `plot_....ipynb` : afin de plot une certaine valeur. On retrouvera le plot dans le dossier results
 

## Utilisation

Il faut charger les données dans un dossier data dans ce fichier (cf utilisation d'onyxia dans le rapport)

Ensuite régler les hyperparamètres comme on en a envie

Et enfin on peut lancer dans le terminal avec la commande python main.py
