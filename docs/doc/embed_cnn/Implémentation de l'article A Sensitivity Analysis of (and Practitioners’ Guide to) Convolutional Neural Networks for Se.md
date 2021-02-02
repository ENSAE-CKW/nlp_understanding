# Implémentation de l'article *A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification* de Zhang et Wallace [(2012)](https://arxiv.org/pdf/1510.03820.pdf)

Le but de ce document est de présenter l'article de Zhang et al. puis d'appliquer ses différentes conclusions à trois jeux de données.

[TOC]

## Article de Zhang 

L'article de Zhang et al propose une analyse de sensibilité des réseaux de neurones convolutionnels pour la classification binaire de phrases.  Ces réseaux se basent sur une unique couche convolutionnelle suivi d'une opération de *pooling* ainsi que d'une couche complètement connectée . 

<img src="C:\Users\khale\OneDrive\Documents\Ensae\Projet stat\Papier\modele.png" alt="modele"  />

Pour se faire, les auteurs ont fait varier différents paramètres du modèle : 

- le type d'embedding (ou plongement lexical) : One-hot encoding, Word2Vec et Glove
- la largeur des filtres de la couche convolutionnelle (il s'agit de filtre unidimensionnel)
- le nombre de types de filtres différents 
- la fonction de pooling 
- la fonction d'activation en sortie de la couche complètement connectée
- le paramètre de *dropout* (probabilité "d'éteindre" un neurone)

Cependant les paramètres suivants restent fixes durant l'expérimentation :

- L'algorithme d'optimisation ADADELTA (qui ne nécessite pas d'initialisation d'un taux d'apprentissage)
- Des minibatchs de taille 50
- La fonction d'entropie croisée pour la fonction objective (problème de classification)

L'entrée du réseau est une phrase représentée sous la forme d'une matrice de dimension $s \times E$ où $s$ désigne le nombre de mots dans la phrase et $E$ la dimension de l'*embedding*. Le nombre de mots $s$ est fixé pour l'ensemble des phrases : si une phrase contient plus de $s$ mots alors seuls les $s$ premiers seront préservés et si une phrase contient moins de $s$ mots alors cette phrase sera complétée par un mot *neutre* appelé padding.

On distinguera dans la suite : 

- *le type de filtre* : il s'agit de la largeur du filtre (dans l'illustration, on retrouve des filtres de largeur **2**, de largeur **3** et de largeur **4**.)
- *le nombre de type de filtre* : il s'agit du nombre de largeurs de filtres différentes utilisées dans la couche convolutionnelle (dans l'illustration, on retrouve trois types de filtre : **2, 3 et 4**)
- *le nombre de filtres utilisés pour chaque type* : il s'agit du nombre de *features maps*. (dans l'illustration, pour chaque type de filtres, deux filtres sont calculés). 

Afin de déterminer les paramètres optimaux, Zhang et al proposent une méthodologie consistant à :

- Choisir entre Word2Vec ou Glove pour de la classification de phrases (ne pas choisir le OneHotEncoding)

- Considérer le modèle avec un seul type de filtre et choisir la largeur du filtre en le faisant varier dans un intervalle "raisonnable" (les auteurs proposent entre 1 et 10) : on note $f^*$, la taille obtenue (dans l'exemple de l'article, la largeur optimale lorsqu'on a un filtre est $f^* = 7$). Puis tester des modèles avec deux, trois, quatre types de filtres de largeur proche de $f^*$ (par exemple, $(6,7,8,9)$). Pour chaque largeur, 400 filtres sont calculés. 

- Une fois le nombre de type de filtres différents et leur largeur choisis : on cherche le nombre de filtres à créer pour chaque type. Les auteurs proposent 

- Faire varier la régularisation : le dropout ainsi que la pénalisation $L_2$.

  ## Application au jeu [allocine_review](https://huggingface.co/datasets/allocine)

  ### Présentation du jeu de données

  Le jeu *allocine_review* est un jeu de données contenant des avis sur des films issus du site [allocine](https://www.allocine.fr/). Les avis ont été écrits entre 2006 et 2020. Pour chaque avis, l'utilisateur indique une note comprise entre 0.5 et 5 sur 5 : si la note est inférieure ou égale à 2 alors l'avis est considéré comme négatif sinon il est considéré comme positif.

  Le jeu est déjà découpé en trois sous-jeux : le jeu d'entrainement (160 000 avis), le jeu de validation (20 000 avis) et le jeu de test (20 000 avis).

  La taille 

  

  

