| Modèle      | Accuracy | Loss     |
| :---        |    :----:   |          ---: |
| Chapitre 1 : Linear     | 35 %       | 1.82   |
| Chapitre 2 : Permute_Linear   | 38.944 %        | 1.78      |
| Chapitre 2 : Convolutions   | 47.478 %        | 1.47      |
| Chapitre 2 : Strided Convolutions   | 35.255 %        | 1.81      |
| Chapitre 2 : My_Conv   | 66.07 %        | 0.98     |
| Chapitre 2 : Bottleneck   | 23.655 %       | 2.00      |
| Chapitre 2 : Inverted_Bottleneck   | 23.962 %       | 2.47      |

On notera qu'il est évident que les modèles employant le bottleneck ne sont aucunement optimaux étant donné leur piètre performance, ou bien qu'ils ne correspondent pas au problème de classification donné.

Entrainement sur 10 epochs et des batchs de 128 pour la plupart, utilise CUDA pour accélerer les calculs.

Pour tester un des modèles, à la racine : 

```console
python3 file.py
```
