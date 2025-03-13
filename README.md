# Github lié au stage de Hugo Pavy à RTE sur les PINNs

## Table des Matières

- [Description](#description)
- [Fonctionnalités](#Fonctionnalités)

## Description

### Contexte du stage 

Ce document est le compte rendu d’un travail de 6 mois dans le cadre de mon premier stage de césure avant ma dernière année du cycle ingénieur civil des Mines de Paris. Il a été réalisé à RTE dans le pôle de gestion des actifs de la R\&D. Les sujets d’études du pôle gestion des actifs sont principalement de prédire des défaillances d’actifs (câbles, pylônes…), d’optimiser leur remplacement, et de concevoir des méthodes pour détecter les défaillances. 

Le pôle de gestion des actifs développe un modèle qui cherche à prédire la fatigue des câbles. Dans ce modèle il est question de réaliser un grand nombre de fois des simulations d'écoulements autour d'un cylindre. Au vu du nombre très important de simulations, il serait avantageux de pouvoir accélérer grandement la vitesse pour simuler un écoulement. C'est là qu'intervient ce sujet de stage : on peut chercher à utiliser l'apprentissage profond pour pouvoir produire des simulations presque instantanément.

Durant ce stage j'ai travaillé sur les méthodes classiques de deep learning (DL) utilisées dans le contexte des Physics Informed Neuron Networks (PINNS). J'ai essayé et appris beaucoup de différentes architectures. Vous trouverez en annexe le meilleur contenu que j'ai trouvé sur internet pour me former sur les différentes notions.

Ce stage est la suite d'un premier stage de 3 mois à RTE réalisé par Issame Maghraoui, élève à Polytechnique. Je me suis appuyé sur ses travaux afin de réaliser mon travail.


### Différentes parties

Durant ce stage je me suis donc intéressé à la technologie des PINNs. Dans un premier temps l'objectif a été à partir de données partielles d'un écoulement, d'essayer de le reconstruire, de faire donc une sorte d'interpolation. Ensuite j'ai essayé de prédire des écoulements de câbles en mouvement. J'ai aussi exploré plusieurs architectures : MLP classiques, PIKANs, Deep Neural Operator et GNN.

J'ai eu la chance durant ce stage de présenter mon travail à la conférence DTE AICOMAS. C'était une expérience très enrichissante pour moi autant de montrer mon travail que de voir celui des autres et le fonctionnement de la recherche. Merci à Fikri, John et Eric pour cette oppurtunité.

Si vous souhaitez lire mon rapport de stage ou pour toute autre demande, merci de me contacter par mail ou via linkedin

**mail**: [hugo.pavy@etu.minesparis.psl.eu](mailto:hugo.pavy@etu.minesparis.psl.eu)

**linkedin**: [Hugo Pavy](https://www.linkedin.com/in/hugo-pavy/)

## Fonctionnalités

Chaque dossier correspond au code d'une partie de mon rapport, il peut être pris de manière indépendante. Il y a beaucoup de similarité dans chaque dossier, j'aurais pu faire un module mieux construit mais je n'ai pas pris le temps. Une fois que l'on a compris le fonctionnement du code d'un dossier je pense donc que c'est très simple de se mettre dans un autre.

Je lance mes codes sur la plateforme onyxia pour avoir accès à des GPU (cf le rapport)

Globalement les codes ont tous à peu près la même structure de fichiers

## Exemples de reconstructions d'écoulements

### Reconstruction avec les PINNs :

![Vidéo de la reconstruction](./reconstruction_ecoulement/results/1_reconstruction_avec_pinns/velocity_norm.gif)


### Reconstruction sans les PINNs :
![Vidéo de la reconstruction](./reconstruction_ecoulement/results/2_reconstruction_sans_pinns/velocity_norm.gif)