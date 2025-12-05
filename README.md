# GoldenF

# 🧠 GO/PAGER Embedding Merger and Clustering Tool

This Python tool merges **Gene Ontology (GO)** description embeddings and **PAGER** graph embeddings, then applies various clustering algorithms to automatically group Gene Sets (GS). The script is configurable through command-line arguments and supports multiple clustering backends, GPU options, and CSV export.

---

## ✨ Features

* 🔄 **Merge embeddings** using:

  * Weighted merge (`alpha`-controlled)
  * Concatenation
* 🔍 **Clustering algorithms supported:**

  * K-Means
  * Agglomerative Clustering
  * Spectral Clustering
  * Louvain Community Detection
  * Girvan–Newman
  * HDBSCAN
* 🧬 Works with GO and Node2Vec/PAGER `.npz` embeddings
* 💾 Optional saving of merged embeddings and predicted labels
* ⚙️ Full control via CLI arguments
* 🧱 Modular design for integration into larger pipelines

---

## 🚀 Installation

Install all dependencies:

```bash
pip install numpy scikit-learn networkx hdbscan python-louvain
```

Ensure your project has:

* GO embeddings `.npz`
* PAGER/Node2Vec embeddings `.npz`
* A GS ID text file `gs_ids.txt`

---

## 🏁 Usage

General command:

```bash
python your_script.py \
    --go_embedding_file path/to/go_embeddings.npz \
    --pager_embedding_file path/to/pager_embeddings.npz \
    --gs_file path/to/gs_ids.txt \
    --merge_method weighted \
    --alpha 0.7 \
    --clustering_algo louvain \
    --num 4 \
    --save_csv
```

---

## 🔧 Command-Line Arguments

| Argument                 | Type  | Default                  | Description                                                                                                 |
| ------------------------ | ----- | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `--go_embedding_file`    | str   | required                 | Path to GO description embeddings (`.npz`)                                                                  |
| `--pager_embedding_file` | str   | required                 | Path to PAGER/Node2Vec embeddings (`.npz`)                                                                  |
| `--gs_file`              | str   | required                 | File containing GS IDs, one per line                                                                        |
| `--merge_method`         | str   | concatenate              | Options: `weighted`, `concatenate`                                                                          |
| `--alpha`                | float | 0.8                      | Weight for weighted merge                                                                                   |
| `--num`                  | int   | 2                        | Number of clusters (for KMeans/Agglomerative/Spectral)                                                      |
| `--clustering_algo`      | str   | agglomerative_clustering | Options: `kmeans`, `agglomerative_clustering`, `spectral_clustering`, `louvain`, `girvan_newman`, `hdbscan` |
| `--cut_dim`              | int   | 512                      | Dimensionality of merged embeddings                                                                         |
| `--gpu`                  | flag  | False                    | Enable GPU if supported by merger                                                                           |
| `--save_csv`             | flag  | False                    | Save results to CSV                                                                                         |
| `--merge_output`         | str   | demo_test.csv            | Output CSV for merged embeddings                                                                            |
| `--labels_output`        | str   | predicted_labels.csv     | Output CSV for predicted labels                                                                             |

---

## 📤 Output Files

If `--save_csv` is used:

### `merged_embeddings.csv`

Contains:

```
gs_id, dim_0, dim_1, ..., dim_n
```

### `predicted_labels.csv`

```
gs_id, pred_label
```

---

## 🧠 How It Works (Overview)

1. **Load GO & PAGER embeddings**
2. **Align vectors for selected GS IDs**
3. **Merge vectors using**:

   * Weighted merge:
     [
     v = (1-\alpha) v_{\text{go}} + \alpha v_{\text{pager}}
     ]
   * OR concatenation
4. **Clean matrix** (handle NaNs, symmetrize if needed)
5. **Apply clustering algorithm**
6. **Return cluster labels + merged embeddings**

---

## 🧪 Supported Clustering Methods

### Feature-based:

* **KMeans**
* **Agglomerative Clustering**
* **HDBSCAN**

### Graph-based (requires affinity matrix):

* **Spectral Clustering**
* **Louvain**
* **Girvan–Newman**

---

## 📚 Example GS IDs File (`gs_ids.txt`)

```
GO:0008150
GO:0009987
GO:0032502
GO:0044699
```

---

## 🛠 Internal Module Dependencies

Your script depends on:

```python
from data_processing.embedding_operation import EmbeddingMerger
```

`EmbeddingMerger` must define:

```python
merge(npz_file1, npz_file2, gs_id_list, merge_method, alpha, output_file, cut_dim)
```

---

## 🧪 Example Output in Terminal

```
[OK] Clustering done with algo='louvain', k=4
     Merged 256 GS IDs; example: ['GO:0008219', 'GO:0032501', 'GO:0009987']
     Labels example: [1, 0, 2, 1, 3, ...]
[Saved] Merged embeddings -> demo_test.csv
[Saved] Predicted labels -> predicted_labels.csv
```

