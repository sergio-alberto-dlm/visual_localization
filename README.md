# 📷 Image Retrieval with DINOv3 + FAISS on CroCoDL

This project implements an **image retrieval pipeline** leveraging the strong semantic embeddings of **DINOv3** (ViT-B/16) and efficient similarity search with **FAISS**.  
It is designed for the **CroCoDL dataset**, and provides tools for:

- ✅ Extracting **global embeddings** for map/query images  
- ✅ Performing **image retrieval** (top-K nearest neighbors)  
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

Compute DINOv3 (ViT-B/16) features for the iOS map dataset. Each subsession will produce one .npy file containing embeddings.

```bash 
python tasks/extract_embeddings.py \
  --sessions_root /path/to/sessions \
  --out_dir /path/to/output \
  --repo_dir submodules/dinov3 \
  --weights /path/to/dinov3_vitb16.pth \
  --batch_size 32 \
  --device cuda:0 \
  --normalize
```

Key arguments:

* --sessions_root: Path to sessions containing ios_map/ and ios_query/

* --out_dir: Directory to store embeddings (.npy per subsession)

* --normalize: Apply L2 normalization (recommended)

* --skip_existing: Skip already processed subsessions

### 2. Run retrieval

Retrieve top-K most similar map images for each query image using cosine similarity (with optional FAISS acceleration).

```bash 
python tasks/retrieval.py \
  --sessions_root /path/to/sessions \
  --map_feats_root /path/to/output_embeddings \
  --out_jsonl /path/to/results.jsonl \
  --repo_dir submodules/dinov3 \
  --weights /path/to/dinov3_vitb16.pth \
  --topk 10 \
  --use_faiss --faiss_gpu \
  --normalize
```

Key arguments:

* --map_feats_root: Features directory (output of extractor)

* --topk: Number of nearest neighbors to retrieve (default: 10)

* --use_faiss: Use FAISS for fast similarity search

* --faiss_gpu: Move FAISS index to GPU

Results are saved as a .jsonl file with retrieval information.

### 3. Visualize the dataset

Visualize iOS map/query sessions with rerun.io:

```bash 
python tasks/visualize.py \
  --scene /path/to/capture/scene \
  --split map \
  --subsession all \
  --every 5 \
  --spawn
```

Key arguments:

* --scene: Path to a single capture scene directory

* --split: Either map or query

* --subsession: Subsession name (or all for all subsessions)

* --every: Log every Nth frame (for faster visualization)

* --spawn: Automatically launch Rerun viewer

## 🙏 Acknowledgments

- [**CroCoDL benchmark**](https://github.com/cvg/crocodl-benchmark/tree/main) – dataset and tools for large-scale visual localization  
- [**DINOv3**](https://github.com/facebookresearch/dinov3) – strong semantic embeddings with ViT  
- [**FAISS**](https://github.com/facebookresearch/faiss) – efficient similarity search library  
