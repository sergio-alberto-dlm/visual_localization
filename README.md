# 📷 Visual Localization: MegaLoc + VGGT on CroCoDL

This project implements a **cross device visual localization pipeline** leveraging the strong semantic embeddings of **MegaLoc** and the foundational model **VGGT**.  
It is designed for the **CroCoDL dataset**, and provides tools for:

- ✅ Extracting **global embeddings** for map/query images  
- ✅ Performing **image retrieval** (top-K)  
- ✅ Performing **visual localization** leveraging the robustness of VGGT
- ✅ **Visualizing sessions** and trajectories with [rerun.io](https://rerun.io)

---

## 🚀 Installation

This project uses **Git submodules**, so make sure you clone the repository properly.  

### 1. Clone the repository

```bash
git clone git@github.com:sergio-alberto-dlm/visual_localization.git --recursive
cd visual_localization
``` 

💡 If you already cloned the repo without --recursive, run:

```bash
git submodule update --init --recursive
```

### 2. Create conda virtual environment (recommended)

```bash 
conda env create -f environment.yml
conda activate vision
```

## 🧩 Usage
### 1. Extract embeddings

Compute MegaLoc features for the device map dataset. Each subsession will produce one .npy file containing embeddings.

```bash 
python tasks/glob_feats.py \
  --sessions_root /path/to/sessions \
  --out_dir /path/to/output \
  --device \
  --subsession \
  --batch_size 32 \
  --compute_device cuda:0 \
  --normalize \
  --skip_existing
```

Key arguments:

* --sessions_root: Path to sessions containing ios_map/ and ios_query/

* --out_dir: Directory to store embeddings (.npy per subsession)

* --device: Device type could be [ios, hl, spot]

* --subsession: Subsession name you wish to compute 

* --normalize: Apply L2 normalization (recommended)

* --skip_existing: Skip already processed subsessions

### 2. Run retrieval

Retrieve top-K most similar map images for each query image using cosine similarity (with optional FAISS acceleration).

```bash 
python tasks/retrieval.py \
  --sessions_root /path/to/sessions \
  --map_feats_root /path/to/output_embeddings \
  --map_device \
  --query_device \
  --out_jsonl /path/to/results.jsonl \
  --topk 10 \
  --use_faiss --faiss_gpu \
  --normalize \
  --compute_device cuda:0 
```

Key arguments:

* --map_feats_root: Features directory (output of extractor)

* --map_device: Device type for map split 

* --query_device: Device type for query split 

* --topk: Number of nearest neighbors to retrieve (default: 10)

* --use_faiss: Use FAISS for fast similarity search

* --faiss_gpu: Move FAISS index to GPU

Results are saved as a .jsonl file with retrieval information.

### 3. Visualize the dataset

Visualize device map/query sessions with rerun.io:

```bash 
python utils/vis_crocodl.py \
  --scene /path/to/scene \
  --split \
  --device \
  --subsession \
  --keyframes-only \
  --spawn \
  --vis-retrieve-queries \
  --queries-pose
```

Key arguments:

* --scene: Path to a single capture/scene containing map,query subfolders per device

* --split: Split to visualize

* --device: Device type

* --subsession: Subsession to visualize. Use 'all' to iterate all

* --keyframes_only: (query only) visualize only pruned keyframes

* --spawn: Spawn the Rerun viewer automatically

* --vis-retrieve-queries: If provided, visualize retrieve queries (poses as pinhole cameras)

* --queries-pose: Path to trajectories.txt of retrieve queries

## 🙏 Acknowledgments

- [**CroCoDL benchmark**](https://github.com/cvg/crocodl-benchmark/tree/main) – dataset and tools for large-scale visual localization  
- [**FAISS**](https://github.com/facebookresearch/faiss) – efficient similarity search library  
- [**MegaLoc**](https://github.com/gmberton/megaloc) -- embeddings extractor
