# VMH-P Main-Table Seed-42 Artifact Map

This local bundle contains only the nine host/dataset settings reported in the
current main results table. The exported names follow the manuscript setting
with `seed=42`.

Values are ACC/F1 (%) as reported in the current main table.

| Host | Dataset | Table Value | Local Files |
|---|---|---:|---|
| CLMLF | MVSA-Single | 78.22 / 77.39 | `clmlf/mvsa_single/{backbone,cache,vmhp}` |
| CLMLF | MVSA-Multiple | 75.53 / 73.57 | `clmlf/mvsa_multiple/{backbone,cache,vmhp}` |
| CLMLF | TumEmo | 73.35 / 73.32 | `clmlf/tumemo/{backbone,cache,vmhp}` |
| D2R | MVSA-Single | 78.00 / 77.94 | `d2r/mvsa_single/{backbone,cache,vmhp}` |
| D2R | MVSA-Multiple | 76.12 / 74.26 | `d2r/mvsa_multiple/{backbone,cache,vmhp}` |
| D2R | TumEmo | 77.39 / 77.41 | `d2r/tumemo/{backbone,cache,vmhp}` |
| SPP-SCL | MVSA-Single | 89.56 / 89.04 | `spp_scl/mvsa_single/{backbone,cache,vmhp}` |
| SPP-SCL | MVSA-Multiple | 83.35 / 82.66 | `spp_scl/mvsa_multiple/{backbone,cache,vmhp}` |
| SPP-SCL | TumEmo | 73.59 / 73.67 | `spp_scl/tumemo/{backbone,cache,vmhp}` |

## File Naming

For most settings, the local VMH-P subdirectory contains:

```text
vmhp_seed42.pt
```

For CLMLF/MVSA-Multiple and D2R/MVSA-Multiple, two local checkpoint exports are
kept for the corresponding ACC/F1 table entries:

```text
vmhp_seed42_acc.pt
vmhp_seed42_f1.pt
```

The `*_acc` files correspond to the table ACC value, and the `*_f1` files
correspond to the table F1 value.

## Layout

```text
main_table_seed42/
  clmlf/
    mvsa_single/
      backbone/best_model.pth
      cache/{train_features.pt,dev_features.pt,test_features.pt}
      vmhp/{vmhp_seed42.pt}
    mvsa_multiple/
      backbone/best_model.pth
      cache/{train_features.pt,dev_features.pt,test_features.pt}
      vmhp/{vmhp_seed42_acc.pt,vmhp_seed42_f1.pt}
    tumemo/
      backbone/best_model.pth
      cache/{train_features.pt,val_features.pt,test_features.pt}
      vmhp/{vmhp_seed42.pt}

  d2r/
    mvsa_single/
      backbone/best_model.pth
      cache/{train_features.pt,dev_features.pt,test_features.pt,feature_summary.json}
      vmhp/{vmhp_seed42.pt}
    mvsa_multiple/
      backbone/best_model.pth
      cache/{train_features.pt,dev_features.pt,test_features.pt,feature_summary.json}
      vmhp/{vmhp_seed42_acc.pt,vmhp_seed42_f1.pt}
    tumemo/
      backbone/best_model.pth
      cache/{train_features.pt,dev_features.pt,test_features.pt,feature_summary.json}
      vmhp/{vmhp_seed42.pt}

  spp_scl/
    mvsa_single/
      backbone/best_model.pth
      cache/{train_features.pt,val_features.pt}
      vmhp/{vmhp_seed42.pt}
    mvsa_multiple/
      backbone/best_model.pth
      cache/{train_features.pt,val_features.pt}
      vmhp/{vmhp_seed42.pt}
    tumemo/
      backbone/best_model.pth
      cache/{train_features.pt,valid_features.pt,test_features.pt}
      vmhp/{vmhp_seed42.pt}
```
