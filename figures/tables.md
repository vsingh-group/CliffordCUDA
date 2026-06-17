### Forward speed (µs)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch size = 1** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 15.5 ± 0.5 | 15.7 ± 0.8 | 16.1 ± 0.3 | 19.6 ± 2.3 | 31.8 ± 0.2 | 55.8 ± 0.2 |
| &nbsp;&nbsp;Wedge product | 12.4 ± 0.0 | 13.7 ± 0.2 | 15.1 ± 0.2 | 18.2 ± 0.2 | 33.5 ± 6.4 | 58.7 ± 0.6 |
| &nbsp;&nbsp;Inner product | 12.1 ± 0.1 | 12.7 ± 0.3 | 13.8 ± 0.2 | 16.2 ± 0.2 | 34.2 ± 0.1 | 71.2 ± 0.4 |
| &nbsp;&nbsp;Left contraction | 12.2 ± 0.3 | 13.3 ± 0.5 | 14.9 ± 0.1 | 18.6 ± 0.1 | 33.7 ± 0.6 | 65.9 ± 0.3 |
| &nbsp;&nbsp;Right contraction | 12.2 ± 0.2 | 13.1 ± 0.2 | 14.9 ± 0.2 | 18.5 ± 0.2 | 33.9 ± 0.4 | 69.4 ± 0.2 |
| &nbsp;&nbsp;Regressive product | 23.3 ± 1.1 | 23.1 ± 0.6 | 23.6 ± 0.6 | 23.7 ± 0.7 | 44.6 ± 1.5 | 69.3 ± 0.4 |
| &nbsp;&nbsp;Rotor application | 24.5 ± 0.3 | 28.1 ± 0.3 | 33.2 ± 0.3 | 43.1 ± 1.1 | 47.0 ± 0.6 | 52.6 ± 0.6 |
| **Batch size = 16** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 16.0 ± 0.6 | 17.1 ± 0.5 | 20.7 ± 0.4 | 32.2 ± 0.7 | 74.1 ± 0.6 | 242.1 ± 4.2 |
| &nbsp;&nbsp;Wedge product | 12.9 ± 0.3 | 14.3 ± 0.1 | 17.4 ± 0.2 | 24.4 ± 0.3 | 42.0 ± 0.2 | 89.5 ± 0.2 |
| &nbsp;&nbsp;Inner product | 12.9 ± 0.2 | 14.6 ± 0.2 | 17.0 ± 0.2 | 22.4 ± 0.2 | 36.2 ± 0.1 | 83.0 ± 0.2 |
| &nbsp;&nbsp;Left contraction | 12.9 ± 0.2 | 14.1 ± 1.7 | 16.4 ± 0.2 | 21.9 ± 0.1 | 31.2 ± 0.1 | 65.4 ± 0.1 |
| &nbsp;&nbsp;Right contraction | 12.7 ± 0.2 | 13.9 ± 0.1 | 17.5 ± 0.2 | 20.4 ± 4.1 | 49.4 ± 0.1 | 111.0 ± 0.7 |
| &nbsp;&nbsp;Regressive product | 23.9 ± 0.7 | 23.7 ± 0.5 | 26.8 ± 0.3 | 37.7 ± 0.6 | 116.3 ± 0.3 | 163.7 ± 0.4 |
| &nbsp;&nbsp;Rotor application | 24.8 ± 0.5 | 28.2 ± 0.5 | 33.9 ± 0.8 | 44.0 ± 0.6 | 48.3 ± 0.3 | 100.6 ± 0.6 |
| **Batch size = 64** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 17.3 ± 0.4 | 21.0 ± 0.8 | 34.3 ± 0.3 | 79.8 ± 0.4 | 254.9 ± 1.4 | 970.7 ± 9.4 |
| &nbsp;&nbsp;Wedge product | 14.5 ± 0.1 | 18.0 ± 0.1 | 24.1 ± 0.1 | 41.3 ± 0.3 | 90.6 ± 0.3 | 245.5 ± 3.2 |
| &nbsp;&nbsp;Inner product | 14.7 ± 0.1 | 18.2 ± 0.2 | 24.4 ± 0.2 | 44.1 ± 0.2 | 104.8 ± 0.3 | 286.9 ± 6.5 |
| &nbsp;&nbsp;Left contraction | 14.2 ± 0.2 | 17.2 ± 0.2 | 22.0 ± 0.2 | 36.8 ± 0.2 | 79.5 ± 0.1 | 228.3 ± 5.2 |
| &nbsp;&nbsp;Right contraction | 14.1 ± 0.3 | 18.7 ± 0.4 | 21.6 ± 0.1 | 36.6 ± 0.2 | 81.2 ± 0.2 | 225.9 ± 4.5 |
| &nbsp;&nbsp;Regressive product | 23.9 ± 4.1 | 28.3 ± 0.4 | 38.1 ± 0.3 | 72.0 ± 0.5 | 163.8 ± 0.3 | 275.6 ± 2.6 |
| &nbsp;&nbsp;Rotor application | 24.8 ± 0.3 | 28.9 ± 0.5 | 34.6 ± 0.5 | 45.0 ± 1.2 | 49.3 ± 0.8 | 102.9 ± 0.2 |

### Backward speed (µs)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch size = 1** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 72.5 ± 0.7 | 72.0 ± 0.8 | 72.5 ± 0.4 | 72.9 ± 0.7 | 69.7 ± 0.7 | 132.5 ± 0.8 |
| &nbsp;&nbsp;Wedge product | 80.7 ± 0.9 | 80.0 ± 0.5 | 82.5 ± 0.9 | 53.6 ± 0.7 | 78.9 ± 0.5 | 164.5 ± 0.7 |
| &nbsp;&nbsp;Inner product | 78.6 ± 0.7 | 78.2 ± 0.6 | 78.5 ± 0.7 | 78.9 ± 0.7 | 92.2 ± 0.9 | 157.8 ± 125.1 |
| &nbsp;&nbsp;Left contraction | 78.1 ± 4.5 | 78.1 ± 0.7 | 79.2 ± 0.5 | 78.7 ± 0.4 | 76.5 ± 0.3 | 148.3 ± 0.6 |
| &nbsp;&nbsp;Right contraction | 78.3 ± 0.8 | 78.1 ± 0.8 | 86.4 ± 1.3 | 78.0 ± 0.7 | 76.4 ± 0.4 | 137.8 ± 0.4 |
| &nbsp;&nbsp;Regressive product | 87.2 ± 0.8 | 89.3 ± 0.9 | 89.2 ± 0.7 | 89.2 ± 1.1 | 98.2 ± 0.6 | 149.0 ± 0.8 |
| **Batch size = 16** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 72.0 ± 0.9 | 71.5 ± 0.6 | 72.1 ± 0.2 | 81.0 ± 0.5 | 153.9 ± 1.1 | 514.5 ± 9.8 |
| &nbsp;&nbsp;Wedge product | 80.7 ± 0.9 | 81.8 ± 0.9 | 82.9 ± 0.6 | 67.4 ± 0.4 | 132.1 ± 0.5 | 362.0 ± 0.7 |
| &nbsp;&nbsp;Inner product | 77.9 ± 3.3 | 78.7 ± 0.8 | 79.0 ± 0.6 | 89.5 ± 0.5 | 173.0 ± 0.8 | 452.5 ± 2.5 |
| &nbsp;&nbsp;Left contraction | 77.2 ± 0.4 | 78.8 ± 0.6 | 78.7 ± 0.7 | 83.3 ± 0.8 | 137.1 ± 0.6 | 381.7 ± 2.7 |
| &nbsp;&nbsp;Right contraction | 77.4 ± 0.8 | 54.0 ± 13.2 | 84.3 ± 1.0 | 82.7 ± 1.1 | 200.2 ± 0.5 | 367.7 ± 0.6 |
| &nbsp;&nbsp;Regressive product | 88.2 ± 0.7 | 89.2 ± 0.6 | 88.5 ± 0.7 | 92.9 ± 0.8 | 254.2 ± 1.0 | 403.9 ± 2.5 |
| **Batch size = 64** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 71.4 ± 1.0 | 72.3 ± 0.5 | 81.4 ± 0.6 | 159.7 ± 0.6 | 528.8 ± 12.7 | 1,983.6 ± 2.8 |
| &nbsp;&nbsp;Wedge product | 79.0 ± 0.5 | 82.1 ± 0.5 | 85.1 ± 3.0 | 137.2 ± 0.4 | 371.9 ± 1.5 | 1,331.2 ± 15.1 |
| &nbsp;&nbsp;Inner product | 78.9 ± 0.7 | 77.7 ± 0.7 | 89.9 ± 0.7 | 178.6 ± 0.8 | 466.2 ± 1.4 | 1,638.8 ± 7.9 |
| &nbsp;&nbsp;Left contraction | 77.1 ± 1.8 | 78.9 ± 0.8 | 84.9 ± 0.5 | 143.7 ± 0.5 | 403.1 ± 1.9 | 1,334.3 ± 10.9 |
| &nbsp;&nbsp;Right contraction | 78.1 ± 1.1 | 53.7 ± 0.6 | 83.2 ± 0.5 | 143.5 ± 0.5 | 401.9 ± 0.7 | 1,330.8 ± 16.1 |
| &nbsp;&nbsp;Regressive product | 89.1 ± 1.1 | 89.1 ± 0.6 | 93.8 ± 0.5 | 166.9 ± 0.6 | 429.3 ± 2.1 | 1,374.7 ± 13.7 |

### Forward memory (MiB)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch size = 1** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 0.00 | 0.01 | 0.04 | 0.14 | 0.52 | 2.05 |
| &nbsp;&nbsp;Wedge product | 0.01 | 0.03 | 0.09 | 0.14 | 0.52 | 2.05 |
| &nbsp;&nbsp;Inner product | 0.00 | 0.01 | 0.04 | 0.14 | 0.52 | 4.28 |
| &nbsp;&nbsp;Left contraction | 0.01 | 0.03 | 0.09 | 0.26 | 1.02 | 4.05 |
| &nbsp;&nbsp;Right contraction | 0.01 | 0.03 | 0.09 | 0.26 | 1.02 | 2.20 |
| &nbsp;&nbsp;Regressive product | 0.01 | 0.02 | 0.08 | 0.28 | 1.05 | 4.11 |
| &nbsp;&nbsp;Rotor application | 0.01 | 0.02 | 0.03 | 0.07 | 0.17 | 0.38 |
| **Batch size = 16** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 0.03 | 0.05 | 0.12 | 0.31 | 0.88 | 2.75 |
| &nbsp;&nbsp;Wedge product | 0.03 | 0.08 | 0.18 | 0.43 | 1.10 | 2.89 |
| &nbsp;&nbsp;Inner product | 0.03 | 0.05 | 0.26 | 0.66 | 1.79 | 4.98 |
| &nbsp;&nbsp;Left contraction | 0.04 | 0.08 | 0.18 | 0.44 | 1.10 | 2.91 |
| &nbsp;&nbsp;Right contraction | 0.04 | 0.08 | 0.18 | 0.44 | 1.10 | 2.91 |
| &nbsp;&nbsp;Regressive product | 0.04 | 0.08 | 0.19 | 0.51 | 1.62 | 3.94 |
| &nbsp;&nbsp;Rotor application | 0.04 | 0.07 | 0.15 | 0.31 | 0.64 | 1.32 |
| **Batch size = 64** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 0.10 | 0.20 | 0.41 | 0.88 | 2.00 | 5.00 |
| &nbsp;&nbsp;Wedge product | 0.10 | 0.22 | 0.46 | 1.00 | 2.22 | 5.14 |
| &nbsp;&nbsp;Inner product | 0.11 | 0.24 | 0.54 | 1.23 | 2.92 | 7.23 |
| &nbsp;&nbsp;Left contraction | 0.11 | 0.22 | 0.46 | 1.00 | 2.23 | 5.16 |
| &nbsp;&nbsp;Right contraction | 0.11 | 0.22 | 0.46 | 1.00 | 2.23 | 5.16 |
| &nbsp;&nbsp;Regressive product | 0.13 | 0.27 | 0.57 | 2.01 | 4.25 | 9.19 |
| &nbsp;&nbsp;Rotor application | 0.13 | 0.26 | 0.53 | 1.06 | 2.14 | 4.32 |

### Backward memory (MiB)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch size = 1** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 0.01 | 0.03 | 0.11 | 0.40 | 1.55 | 6.09 |
| &nbsp;&nbsp;Wedge product | 0.01 | 0.04 | 0.17 | 0.77 | 2.55 | 10.09 |
| &nbsp;&nbsp;Inner product | 0.01 | 0.05 | 0.17 | 0.65 | 2.55 | 10.09 |
| &nbsp;&nbsp;Left contraction | 0.01 | 0.05 | 0.20 | 0.77 | 3.05 | 12.09 |
| &nbsp;&nbsp;Right contraction | 0.01 | 0.05 | 0.20 | 0.77 | 3.05 | 12.09 |
| &nbsp;&nbsp;Regressive product | 0.02 | 0.06 | 0.21 | 0.79 | 3.08 | 12.16 |
| **Batch size = 16** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 0.05 | 0.12 | 0.28 | 0.75 | 2.25 | 7.50 |
| &nbsp;&nbsp;Wedge product | 0.06 | 0.13 | 0.34 | 1.00 | 3.75 | 11.50 |
| &nbsp;&nbsp;Inner product | 0.06 | 0.13 | 0.34 | 1.00 | 3.25 | 11.50 |
| &nbsp;&nbsp;Left contraction | 0.06 | 0.14 | 0.38 | 1.12 | 3.75 | 13.50 |
| &nbsp;&nbsp;Right contraction | 0.06 | 0.14 | 0.38 | 1.12 | 3.75 | 13.50 |
| &nbsp;&nbsp;Regressive product | 0.07 | 0.16 | 0.41 | 1.20 | 3.90 | 13.80 |
| **Batch size = 64** |  |  |  |  |  |  |
| &nbsp;&nbsp;Geometric product | 0.19 | 0.40 | 0.84 | 1.88 | 4.50 | 12.00 |
| &nbsp;&nbsp;Wedge product | 0.20 | 0.41 | 0.94 | 2.25 | 6.00 | 16.00 |
| &nbsp;&nbsp;Inner product | 0.20 | 0.42 | 0.94 | 2.25 | 6.00 | 16.00 |
| &nbsp;&nbsp;Left contraction | 0.20 | 0.42 | 0.94 | 2.25 | 6.00 | 18.00 |
| &nbsp;&nbsp;Right contraction | 0.20 | 0.42 | 0.94 | 2.25 | 6.00 | 18.00 |
| &nbsp;&nbsp;Regressive product | 0.23 | 0.49 | 1.07 | 2.51 | 6.52 | 19.05 |
