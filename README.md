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