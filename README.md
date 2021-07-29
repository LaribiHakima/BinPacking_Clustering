# Techniques d'optimisation combinatoire pour la résolution du problème de Bin Packing et du Clustering

## Problème du Bin Packing
Le Bin Packing est un problème d'optimisation combinatoire, ici, il a été question d'implémenter :
- Les méthodes exactes : Branch & Bound avec parcours en parfondeur et en largeur.
- Les méthodes approchées :
  - Les Heuristiques Spécifiques : Best Fit, Best Fit Decreasing, First Fit, First Fit Decreasing, Worst Fit, Next Fit
  - Les Méta-heuristiques : Algorithme Génitique, Recuit Simulé

Les performances des algorithmes implémentés ont été testées sur des instances de types : faciles, normales, difficiles, aléatoires.

## Problème du Clustering
Le clustering (ou partitionnement des données) est une méthode de classification non supervisée qui rassemble un ensemble d’algorithmes d’apprentissage dont le but est de regrouper entre elles des données ayant des caractéristiques similaires.
Ici, il a été question d'adapter l'algorithme du Recuit Simulé, initialement conçu pour résoudre des problèmes d'optimisation combinatoire, pour convenir au problème du Clustering. Les perfomrances de la solution implémentée ont été testées sur deux datasets : Iris et Heart.

## Interface Graphique
L'ensemble des algorithmes implémentés et des tests réalisés ont été assemblés dans l'interface graphique conçue avec le Framework Streamlit.  
Exécuter avec la commande suivante sur console : **streamlit run main.py**
