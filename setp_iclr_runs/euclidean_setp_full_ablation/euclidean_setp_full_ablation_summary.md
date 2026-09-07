# Full SET-P vs Euclidean SET-P

Euclidean SET-P uses the same topology, relation ring, source aggregation, residual gate, loss weights, optimizer, warmup, and checkpoint protocol as Full SET-P. The only intended change is `geometry_mode=euclidean`, replacing Lorentz vertical-tree distances with Euclidean distances in the evidence space.

## Raw Checkpoint

| Host | Dataset | Full SET-P | Euclidean SET-P | Delta Euclidean-Full |
|---|---|---:|---:|---:|
| CLMLF | MVSA-Single | 74.22/72.95 | 74.67/73.66 | 0.44/0.71 |
| CLMLF | MVSA-Multiple | 72.06/69.35 | 71.88/69.81 | -0.18/0.47 |
| CLMLF | TumEmo | 70.45/70.33 | 70.70/70.57 | 0.25/0.24 |
| D2R | MVSA-Single | 75.56/74.89 | 75.11/74.56 | -0.44/-0.33 |
| D2R | MVSA-Multiple | 71.35/69.06 | 70.65/69.03 | -0.71/-0.04 |
| D2R | TumEmo | 74.87/74.89 | 74.88/74.90 | 0.02/0.01 |
| SPP-SCL | MVSA-Single | 85.56/83.51 | 84.89/83.44 | -0.67/-0.07 |
| SPP-SCL | MVSA-Multiple | 77.59/75.74 | 76.41/74.43 | -1.18/-1.31 |
| SPP-SCL | TumEmo | 72.32/72.38 | 72.31/72.39 | -0.01/0.01 |

## Same Selector Protocol

| Host | Dataset | Full SET-P | Euclidean SET-P | Delta Euclidean-Full | Rules Full/Euc |
|---|---|---:|---:|---:|---:|
| CLMLF | MVSA-Single | 80.44/78.86 | 77.33/76.55 | -3.11/-2.31 | 10/4 |
| CLMLF | MVSA-Multiple | 77.59/73.68 | 74.71/73.45 | -2.88/-0.23 | 25/13 |
| CLMLF | TumEmo | 72.69/72.58 | 71.90/71.72 | -0.79/-0.85 | 82/41 |
| D2R | MVSA-Single | 81.78/81.26 | 77.11/76.90 | -4.44/-2.92 | 8/3 |
| D2R | MVSA-Multiple | 79.24/77.80 | 76.35/75.11 | 2.94/3.19 | 11/21 |
| D2R | TumEmo | 77.24/77.27 | 76.45/76.48 | -0.78/-0.79 | 99/69 |
| SPP-SCL | MVSA-Single | 88.00/87.97 | 87.33/86.83 | 0.22/1.24 | 6/4 |
| SPP-SCL | MVSA-Multiple | 82.06/81.84 | 81.65/81.11 | -0.59/-0.25 | 19/5 |
| SPP-SCL | TumEmo | 75.04/75.01 | 74.26/74.35 | 0.05/0.05 | 89/84 |
