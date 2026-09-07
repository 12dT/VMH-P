# Efficiency Host Summary with GFLOPs

GFLOPs are counted for the VMH-P plug-in head only and exclude frozen host inference.

| Host | Params (M) | Plugin Size (MB) | GFLOPs | Extra Time (ms/sample) |
|---|---:|---:|---:|---:|
| D2R | 0.1521--0.1530 | 0.60 | 1.60e-4--1.69e-4 | 0.0120--0.0178 |
| CLMLF | 0.1521--0.4746 | 0.60--1.83 | 1.63e-4--5.43e-4 | 0.0118--0.0206 |
| SPP-SCL | 0.0128--0.0139 | 0.07 | 2.35e-5--3.04e-5 | 0.0077--0.0171 |
