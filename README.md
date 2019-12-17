# MachineLearning





### Description du projet

Ce projet a pour but de classifier des cellules infectées ou non par le malaria (paludisme).

Il comprend trois principaux fichiers sources:

	* program.py     : programme qui crée un modèle simple avec des convolutions, l'entraîne et l'évalue. Il produit deux fichiers pour la sauvegarde du model et de ses poids; basicModel.h5 et basicWeights.h5.

	* testResNet50.py: programme qui utilise la technologie ResNet50 (qui n'est pour l'instant pas fiable car résultats trop médiocres). Il produit deux fichiers pour la sauvegarde du model et de ses poids; ResNet50Model.h5 et ResNet50Weights.h5.

	* predict.py     : programme qui permet d'effectuer des prédictions. Il faudra préciser en ligne de commande le type de model à employer (le modèle simple ou celui utilisant ResNet50).

	Pour l'exécuter, il faut procéder comme suit:
		$ python3 predict.py [ResNet50|basicModel]
		Indiquer soit ResNet50 ou basicModel. Avec cette ligne de commande, le programme prend tout le dataset prévu pour les prédictions ('TestPredictions') et renvoie les prédictions faites sur chacune des images et renvoie les pourcentages finaux de la prédiction pour chacun des répertoires présents dans le dossier 'TestPredictions'.

	Ou

		$ python3 predict.py [ResNet50|basicModel] cheminVersImage1 cheminVersImage2 ....

		Avec cette dernière, les prédictions sont faites uniquement sur les images citées en arguments en ligne de commande.
				
				
### Datas:
Le dataset a été pris à partir de ce lien:

https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria/data

### Inspirations et liens:

	* https://keras.io/
	* https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
	* https://www.kaggle.com/
	* ...	

