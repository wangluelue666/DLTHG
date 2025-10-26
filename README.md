
# DLTHG: Dynamic Learning on Temporal HyperGraphs

##  Project Overview
This project provides an implementation of **scholar influence prediction** based on temporal hypergraph representation learning.  
The pipeline includes:  
1. Data preprocessing & feature extraction  
2. Hypergraph construction  
3. Structural / sequential / embedding feature extraction  
4. Dual-LSTM predictor training  



##  Installation

### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate dlthg_env
````

### Option 2: Pip

```bash
pip install -r requirements.txt
```

>  Make sure your **PyTorch** version matches `1.13.1` with **CUDA 11.6**, otherwise `dhg==0.9.4` may fail to install.

---

##  Usage

### Step 1: Data & Feature Preparation

```bash
python run_prepare.py --dataset acm --start_year 2000 --end_year 2014 --window 3
```

### Step 2: Model Training

```bash
python run_train.py --dataset acm --target hindex --gpu_id 0
```

---


