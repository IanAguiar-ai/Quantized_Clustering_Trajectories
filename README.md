# Quantized Clustering of Synthetic GNSS Trajectories

This repository demo shows how to use **`cluster_routes`** (from `quantized_clustering_trajectories.py`) to group **80 synthetic trajectories** into route types.

- **Notebook**: `demo_cluster_routes.ipynb`
- **Core code**: `quantized_clustering_trajectories.py`
- **Paper (PDF)**: `SIoT_Article_Similarity_Trajectories.pdf` <link>

## What this demo does

1. **Build 4 base shapes** in lat/lon: (A) east–west line, (B) north–south line, (C) diagonal, (D) S-curve.
2. **Sample noisy trajectories** per type with tiny jitter/offset and irregular sampling.
3. **Cluster** all 80 routes with `cluster_routes` using:
   - `truncation_deg = 0.0001` (≈ 11 m near the equator)
   - `scale = 1000.0`
   - `cut_distance = 0.04`
   - `method = "average"`

Counts per type for this demo: **25 (A)**, **15 (B)**, **5 (C)**, **35 (D)**.

## How the method works (high level)

- **Quantization**: mid-tread rounding directly in latitude/longitude to reduce points and bound spatial error.
- **Symmetric similarity**: for each point in one route, find nearest point in the other (via KDTree), average a bounded inverse-distance score both ways, then take the geometric mean.
- **Clustering**: convert similarity to dissimilarity `D = 1 - S`, run hierarchical clustering, and cut at `cut_distance`.

For full details and motivation, see the included paper and equations. The code in `quantized_clustering_trajectories.py` implements the same pipeline used in the paper.

## How to run locally

1. Open `demo_cluster_routes.ipynb` and run all cells.
2. Tweak parameters like `truncation_deg`, `scale`, `cut_distance` to see the effect on clustering.  
3. Replace the synthetic `routes` list with your own list of `pandas.DataFrame` objects (each with columns `latitude` and `longitude`).

### Minimal example with your own routes

```python
import pandas as pd
from quantized_clustering_trajectories import cluster_routes

# Suppose you already built a list of DataFrames:
# routes = [df0, df1, ..., dfN]  # each with 'latitude' and 'longitude'

groups, medoids = cluster_routes(
    routes=routes,
    truncation_deg=0.0001,
    lat_col="latitude",
    lon_col="longitude",
    cut_distance=0.04,
    method="average",
)

print("Cluster sizes:", [len(v) for v in groups.values()])
print("Representative medoids:", medoids)
```

## Updates

- **1.0.0**
    - First version.

## Citation

If you use or adapt this technique, please cite the paper and reference this repository. See the PDF for the formal description and references.
