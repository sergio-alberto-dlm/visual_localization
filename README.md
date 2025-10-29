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

### 4. Retrieval Reranking (optional)

Rerank top-k most similar images using LightGlue + SuperPoint number of inliers. SuperPoint finds local features and LightGlue matches them per image pairs (query and each of the top-k retrieved images), then, RANSAC over geometry check is used to count inliers and outliers.

First, compute the local features.

#### 4.1 Extract local features

```bash
python tasks/local_feats.py \
  --dataset_path <path_to_dataset> \
  --output_path <path_to_output_directory> \
  --building "<building/location>"
```

arguments:
  * --dataset_path: path to dataset root. i.e., containing directory of both SUCCULLENT and HYDRO.

  * --output_path: Path to the directory to store all local features.

  * --building: Scene. Admits HYDRO or SUCCULENT

#### 4.2 Reranking with number of inliers.

Once the local features are computed, you may rerank each retrieval output with the following command:

```bash
python tasks/rerank_retrieval.py \
  --feats_dir <local_features_directory> \
  --output_dir <dir_for_reranked_jsons> \
  --input_json <topk_json_to_process>
```

Arguments:

* --feats_dir: Path to directory where all local features live. Same as <output_path> of step 4.1.

* --output_dir: Path to directory where all reranked json will be dumped.

* --input_json: Path of json with top-k retrieval data to rerank.

### 5. Pose Estimation plus VGGT

With top-k image retrieval data, either directly from step 2 (MegaLoc) or step 4 (reranking), run the wollowing command to compute the poses of each image query. You must run it for each topk json.

```bash
python tasks/pose_estimation.py \
  --dataset_path <path_to_dataset> \
  --output_dir <dir_for_pose_jsons> \
  --input_json <path_to_topk_json>
```

Arguments:

* --dataset_path: path to dataset root. i.e., containing directory of both SUCCULLENT and HYDRO.

* --output_dir: Path to directory where all pose estimation jsons will be dumped.

* --input_json: Path of individual json with topk images to process.


## 🙏 Acknowledgments

- [**CroCoDL benchmark**](https://github.com/cvg/crocodl-benchmark/tree/main) – dataset and tools for large-scale visual localization  
- [**FAISS**](https://github.com/facebookresearch/faiss) – efficient similarity search library  
- [**MegaLoc**](https://github.com/gmberton/megaloc) -- embeddings extractor
- [**LightGlue**](https://github.com/cvg/LightGlue) -- Local features extractor and matching.
- [**VGGT**](https://vgg-t.github.io/) -- Foundation model for 3D reconstruction.

We acknowledge Jean Bernard Hayet for the infraestrcuture support.

