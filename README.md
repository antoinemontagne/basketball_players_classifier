## Projet de Classification des Joueurs de Basket

Ce projet vise à classifier les joueurs de basket dans 2 catégories :

- **0:** Leur carrière durera moins de 5 ans
- **1:** Leur carrière durera plus de 5 ans

Chaque joueur est décrit par 19 caractéristiques ainsi que son prénom.

Le projet est organisé en plusieurs dossiers :

- **classifiers:** Contient un fichier Jupyter comprenant différents tests et une analyse approfondie du problème.
- **data:** Les données des joueurs sont stockées ici.
- **flask-api:** Une API Flask en Python permettant d'interroger le classifieur.
- **model:** Les poids du classifieur et de l'optimiseur sont sauvegardés ici.
- **webpage:** Fichiers HTML pour interroger le classifieur.

Plusieurs classifieurs ont été testés dans le Jupyter, celui utilisé avec l'API est un Perceptron Multi-Couches (MLP) stocké dans le dossier classifiers, sous le nom `mlp_classifier.py`. Il peut être réentraîné en modifiant les variables globales dans le fichier et en lançant la commande :
```bash
python classifiers/mlp_classifier.py
```

Ce MLP est implémenté en PyTorch et évalué de deux manières : classiquement ou en validation croisée en utilisant les fonctions du fichier `core_functions.py`.

Pour lancer l'API Flask, exécutez la commande :
```bash
python flask_api/flask_inference.py
```

Ensuite, ouvrez une nouvelle fenêtre de terminal et exécutez :
```bash
python -m http.server 8000
```

Enfin, accédez à l'URL suivante dans votre navigateur :

[http://localhost:8000/webpage/predict_future_talent.html](http://localhost:8000/webpage/predict_future_talent.html)

Dans le premier champ de la page, saisissez les caractéristiques d'un joueur au format JSON. Un exemple est fourni dans le fichier `default_player.txt` dans le dossier `webpage`.

Deux champs seront ensuite affichés :

- **Prédiction :** 0 ou 1 selon le classifieur.
- **Probabilité :** La probabilité de sortie du classifieur avant d'appliquer le seuil de 0.5.



