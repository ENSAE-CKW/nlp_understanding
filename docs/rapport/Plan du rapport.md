# Plan du rapport

## Introduction : 

La Société Générale Assurances collecte et reçoit une grosse volumétrie de données textuelles à travers différents canaux et notamment par mail. Par exemple, lorqu'un client ou un prospect souhaite se renseigner ou se plaindre auprès d'un service, comment récupérer ce mail et l'envoyer vers le bon destinataire. Initialement, un opérateur humain était là pour lire tous ces mails et les renvoyer. Mais cela est à la fois coûteux en main d'oeuvre et aliénant pour l'individu.
C'est pourquoi il a été développé au sein datalab de la Société Générale Assurances des modèles de classification textuelle pour assigner à chaque mail reçu, un service vers lequel il devait être retourné.


Mais depuis quelques années, les utilisations des réseaux de neurones se multiplient. En effet, ces modèles permettent de répondre à un large spectre de problématiques : classification d'images (AlexNet [2012]) , traitement du langage (Word2Vec, Bert [2018]). 

Malgré des performances remarquables, ces méthodes souffrent d'un manque d'interprétabilité. Il est délicat d'obtenir des éléments expliquant les prédictions d'un modèle constituant généralement un frein pour les équipes métiers. Cette problématique est connue en *machine learning*.
Le but de ce projet est d'utiliser ces différentes méthodes sur une sélection de modèles variés et de comparer les résultats à *des données réelles* issues du département xxxx. 

Dans un premier temps, nous expliciterons les modèles utiliser




## Plan
- les modèles qui nous allons étudier (pour classifier):
    - Architecture CNN
        - Présentation rapide (intérêt vers le CNN)
        - Architecture global : CONV, POOL and FC
        - Filtre, padding, et stride ?
    - Architecture RNN, LSTM et (ajout d'un CNN) **-En une page ?-**
        - Présentation du RNN (intérêt versus le FC), du vanilla RNN, bidimensional RNN
        - Petit mot rapide pour évoquer les différences sur l'entrainement par exemple : BPTT (backpropagation throught time)
        - Limite de ces modèles naïfs (explosing and vanishing gradient) et quelques solutions 
        -  LSTM : input gate, forget gate et output gate.
        -  Modèle CNN-LSTM et LSTM CNN
        -  Bidimensional and Word2Vec
    - Architecture BERT et CNN
        - Présentation des transformers et intérêt
        - L'Attention est tout ce dont vous avez besoin
        - Principal avantage
- les méthodes d'interprétabilité
    - Grad-Cam
    - Couches d’attention
    - Lime / Shap / Deeplift
- Comparaison / Benchmark


Bibliographie : 

- Deep Learning, *Ian Goodfellow and Yoshua Bengio and Aaron Courville*