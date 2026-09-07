# TumEmo Hyperparameter Sensitivity Aligned to Main Table

Values are ACC/F1 (%) after applying the same selector protocol as the current main table: start from final logits, switch among base/tree/plugin/final candidates, rank by ACC+F1, score-selected interval rules, pair class filters, and max_rules=100.

Default points reuse the exact current main-table checkpoint and selector for each host.

## K

| Host | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|
| CLMLF | 72.84/72.79 | 73.35/73.32* | 73.44/73.44 | 73.35/73.32 |
| D2R | 77.34/77.38 | 77.39/77.41* | 77.49/77.51 | 77.48/77.50 |
| SPP-SCL | 73.58/73.64 | 73.59/73.67* | 73.75/73.82 | 73.00/73.10 |

## c

| Host | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|
| CLMLF | 73.51/73.49 | 73.35/73.32* | 73.52/73.46 | 73.67/73.58 |
| D2R | 77.46/77.49 | 77.39/77.41* | 77.34/77.36 | 77.40/77.42 |
| SPP-SCL | 74.10/74.20 | 73.59/73.67* | 73.50/73.56 | 73.22/73.28 |

## eta

| Host | 0.05 | 0.1 | 0.2 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| CLMLF | 72.37/72.12 | 72.89/72.84 | 72.99/72.81 | 73.51/73.49 | 73.35/73.32* |
| D2R | 77.25/77.28 | 77.30/77.33 | 77.07/77.09 | 77.46/77.47 | 77.39/77.41* |
| SPP-SCL | 73.65/73.72 | 73.40/73.47 | 73.57/73.66 | 73.52/73.58 | 73.59/73.67* |

## lambda_str

| Host | 0.1 | 0.3 | 0.5 | 0.7 |
|---|---:|---:|---:|---:|
| CLMLF | 73.35/73.32* | 73.74/73.72 | 73.61/73.56 | 73.51/73.52 |
| D2R | 77.39/77.41* | 76.98/77.02 | 77.32/77.34 | 77.48/77.51 |
| SPP-SCL | 73.59/73.67* | 73.63/73.71 | 73.73/73.82 | 73.89/73.99 |

* marks the host-specific default point reused from the current main table.
