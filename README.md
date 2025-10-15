# GitHub Repository for Hugo Pavy’s Internship at RTE on PINNs

**version française plus bas**

---

## Table of Contents
- [Description](#description)
- [Features](#features)

## Description

### Internship Context
This document summarizes six months of work completed during my first gap-year internship, before my final year at Mines Paris. The internship took place at RTE, within the Asset Management R&D department, which focuses on predicting asset failures (cables, pylons, etc.), optimizing their replacement, and designing failure detection methods.
The department is developing a model to predict cable fatigue, requiring a large number of flow simulations around a cylinder. Given the high computational cost, the goal of this internship was to leverage deep learning to produce near-instantaneous simulations.
During this internship, I worked on classical deep learning methods in the context of Physics-Informed Neural Networks (PINNs). I explored and learned various architectures. The appendix includes the best online resources I found to train on these concepts.
This internship builds upon the work of Issame Maghraoui, a Polytechnique student, who completed a 3-month internship at RTE prior to my arrival.

### Key Focus Areas
My work focused on PINNs. Initially, the goal was to reconstruct a flow from partial data, essentially performing interpolation. I then aimed to predict flows around moving cables. I also explored several architectures: classical MLPs, PIKANs, Deep Neural Operators, and GNNs.
I had the opportunity to present my work at the DTE AICOMAS conference, a highly rewarding experience both for showcasing my research and learning from others. Special thanks to Fikri, John, and Eric for this opportunity.
For any inquiries (reading my internship report, questions, etc.), feel free to contact me via email or LinkedIn:
**Email**: [hugo.pavy@etu.minesparis.psl.eu](mailto:hugo.pavy@etu.minesparis.psl.eu)
**LinkedIn**: [Hugo Pavy](https://www.linkedin.com/in/hugo-pavy/)

## Features
Each folder corresponds to a section of my report and can be used independently. While the codes share many similarities, I did not have time to create a more modular structure. Once you understand the structure of one folder, navigating the others is straightforward.
The codes are run on the Onyxia platform to access GPUs (see the report for details).
Overall, the codes share a similar file structure.

## Flow Reconstruction Examples
### Reconstruction with PINNs:
![Flow Reconstruction Video](./reconstruction_ecoulement/results/1_reconstruction_avec_pinns/velocity_norm.gif)
### Reconstruction without PINNs:
![Flow Reconstruction Video](./reconstruction_ecoulement/results/2_reconstruction_sans_pinns/velocity_norm.gif)


# Github lié au stage de Hugo Pavy à RTE sur les PINNs


## Table des Matières
- [Description](#description)
- [Fonctionnalités](#fonctionnalités)

## Description

### Contexte du stage
Ce document est le compte rendu d’un travail de 6 mois réalisé dans le cadre de mon premier stage de césure avant ma dernière année du cycle ingénieur civil des Mines de Paris. Il a été mené au sein de RTE, dans le pôle de gestion des actifs de la R&D. Les sujets d’études de ce pôle portent principalement sur la prédiction des défaillances d’actifs (câbles, pylônes, etc.), l’optimisation de leur remplacement, et la conception de méthodes pour détecter les défaillances.
Le pôle développe un modèle visant à prédire la fatigue des câbles. Ce modèle nécessite de réaliser un grand nombre de simulations d’écoulements autour d’un cylindre. Étant donné le volume très important de simulations, il serait avantageux d’accélérer significativement leur vitesse d’exécution. C’est dans ce contexte que s’inscrit ce stage : utiliser l’apprentissage profond pour produire des simulations quasi instantanées.
Durant ce stage, j’ai travaillé sur les méthodes classiques de deep learning (DL) appliquées aux Physics-Informed Neural Networks (PINNs). J’ai exploré et appris à maîtriser plusieurs architectures. Vous trouverez en annexe les meilleures ressources que j’ai identifiées pour me former sur ces notions.
Ce stage fait suite à un premier stage de 3 mois réalisé par Issame Maghraoui, élève à Polytechnique, dont les travaux ont servi de base à mon propre travail.

### Différentes parties
Mon travail s’est concentré sur les PINNs. Dans un premier temps, l’objectif était de reconstruire un écoulement à partir de données partielles, c’est-à-dire d’effectuer une interpolation. Ensuite, j’ai cherché à prédire des écoulements autour de câbles en mouvement. J’ai également exploré plusieurs architectures : MLP classiques, PIKANs, Deep Neural Operator et GNN.
J’ai eu l’opportunité de présenter mes résultats à la conférence DTE AICOMAS, une expérience très enrichissante tant pour partager mon travail que pour découvrir celui des autres et le fonctionnement de la recherche. Je remercie Fikri, John et Eric pour cette opportunité.
Pour toute demande (lecture du rapport de stage, questions, etc.), n’hésitez pas à me contacter par email ou via LinkedIn :
**Email** : [hugo.pavy@etu.minesparis.psl.eu](mailto:hugo.pavy@etu.minesparis.psl.eu)
**LinkedIn** : [Hugo Pavy](https://www.linkedin.com/in/hugo-pavy/)

## Fonctionnalités
Chaque dossier correspond à une partie distincte de mon rapport et peut être utilisé de manière indépendante. Bien que les codes présentent de nombreuses similitudes, je n’ai pas eu le temps de les modulariser davantage. Une fois la structure d’un dossier comprise, il est facile de naviguer dans les autres.
Les codes sont exécutés sur la plateforme Onyxia pour accéder à des GPU (voir le rapport pour plus de détails).
Globalement, les codes partagent une structure de fichiers similaire.

## Exemples de reconstructions d’écoulements
### Reconstruction avec les PINNs :
![Vidéo de la reconstruction](./reconstruction_ecoulement/results/1_reconstruction_avec_pinns/velocity_norm.gif)
### Reconstruction sans les PINNs :
![Vidéo de la reconstruction](./reconstruction_ecoulement/results/2_reconstruction_sans_pinns/velocity_norm.gif)

---


