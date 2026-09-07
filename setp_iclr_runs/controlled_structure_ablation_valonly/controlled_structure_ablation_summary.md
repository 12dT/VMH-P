# Controlled Structural Component Ablation (Validation-Selected)

All variants use the same frozen host features, training budget, validation-based model selection, and no test-set selection. Randomized structures are averaged over seeds 0, 1, and 2.

## Test Metrics

| Host | Dataset | Base | Full SET-P | Vertical Only | Random Tree | Random Relation | Δ Full-Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLMLF | MVSA-Single | 71.33/70.48 | 70.67/70.25 | 70.67/70.30 | 71.11$\pm$0.31/70.46$\pm$0.15 | 70.52$\pm$0.10/70.14$\pm$0.12 | -0.67/-0.23 |
| CLMLF | MVSA-Multiple | 71.94/69.42 | 71.53/69.08 | 71.88/69.84 | 71.63$\pm$0.14/69.29$\pm$0.31 | 71.73$\pm$0.14/69.50$\pm$0.27 | -0.41/-0.34 |
| CLMLF | TumEmo | 70.06/69.76 | 70.83/70.79 | 70.88/70.82 | 70.80$\pm$0.03/70.72$\pm$0.06 | 70.87$\pm$0.02/70.81$\pm$0.02 | +0.77/+1.02 |
| D2R | MVSA-Single | 75.78/75.22 | 74.22/73.72 | 74.44/73.97 | 74.67$\pm$0.63/74.11$\pm$0.55 | 74.52$\pm$0.10/74.05$\pm$0.12 | -1.56/-1.50 |
| D2R | MVSA-Multiple | 70.29/68.81 | 70.29/68.57 | 70.47/68.81 | 70.45$\pm$0.22/68.72$\pm$0.21 | 70.49$\pm$0.19/68.92$\pm$0.20 | +0.00/-0.23 |
| D2R | TumEmo | 74.64/74.67 | 74.83/74.85 | 74.83/74.86 | 74.82$\pm$0.03/74.83$\pm$0.04 | 74.85$\pm$0.01/74.87$\pm$0.01 | +0.19/+0.18 |
| SPP-SCL | MVSA-Single | 84.67/83.27 | 84.44/82.88 | 84.44/82.87 | 85.63$\pm$0.84/84.15$\pm$0.90 | 84.37$\pm$0.10/82.80$\pm$0.11 | -0.22/-0.39 |
| SPP-SCL | MVSA-Multiple | 75.76/73.10 | 76.18/74.22 | 77.76/76.11 | 75.98$\pm$0.14/74.03$\pm$0.13 | 77.20$\pm$0.75/75.43$\pm$0.88 | +0.41/+1.11 |
| SPP-SCL | TumEmo | 72.15/72.23 | 72.33/72.40 | 72.34/72.41 | 72.35$\pm$0.02/72.42$\pm$0.02 | 72.30$\pm$0.04/72.37$\pm$0.03 | +0.18/+0.17 |

## Diagnostics Against Full SET-P

| Host | Dataset | Vertical-Full | RandomTree-Full | RandomRelation-Full |
|---|---:|---:|---:|---:|
| CLMLF | MVSA-Single | +0.00/+0.05 | +0.44/+0.21 | -0.15/-0.11 |
| CLMLF | MVSA-Multiple | +0.35/+0.77 | +0.10/+0.22 | +0.20/+0.42 |
| CLMLF | TumEmo | +0.05/+0.04 | -0.03/-0.06 | +0.04/+0.02 |
| D2R | MVSA-Single | +0.22/+0.25 | +0.44/+0.39 | +0.30/+0.33 |
| D2R | MVSA-Multiple | +0.18/+0.24 | +0.16/+0.15 | +0.20/+0.34 |
| D2R | TumEmo | +0.01/+0.01 | -0.01/-0.02 | +0.02/+0.02 |
| SPP-SCL | MVSA-Single | +0.00/-0.01 | +1.19/+1.27 | -0.07/-0.08 |
| SPP-SCL | MVSA-Multiple | +1.59/+1.90 | -0.20/-0.19 | +1.02/+1.21 |
| SPP-SCL | TumEmo | +0.01/+0.01 | +0.02/+0.02 | -0.03/-0.03 |

Interpretation note: this table isolates structural choices under a validation-selected post-hoc protocol. It should be used to discuss component behavior and failure modes conservatively, especially when a randomized variant matches or exceeds Full SET-P on a specific host-dataset pair.
