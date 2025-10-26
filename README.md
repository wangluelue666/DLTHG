
# DLTHG: Dynamic Learning on Temporal HyperGraphs

## 📖 Project Overview
This project provides an implementation of **scholar influence prediction** based on temporal hypergraph representation learning.  
The pipeline includes:  
1. Data preprocessing & feature extraction  
2. Hypergraph construction  
3. Structural / sequential / embedding feature extraction  
4. Dual-LSTM predictor training  



## ⚙️ Installation

### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate dlthg_env
````

### Option 2: Pip

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure your **PyTorch** version matches `1.13.1` with **CUDA 11.6**, otherwise `dhg==0.9.4` may fail to install.

---

## 🚀 Usage

### Step 1: Data & Feature Preparation

```bash
python run_prepare.py --dataset acm --start_year 2000 --end_year 2014 --window 3
```

### Step 2: Model Training

```bash
python run_train.py --dataset acm --target hindex --gpu_id 0
```

---

## 📂 Project Structure

```text
DLTHG/
├── config/                 # Path and hyperparameter configs
├── data/                   # Raw and processed data
├── features/               # Extracted features
├── models/                 # Model implementations (HGNN-SI, Dual-LSTM, etc.)
├── run_prepare.py          # Preparation script
├── run_train.py            # Training script
├── environment.yml         # Conda environment file
├── requirements.txt        # Pip requirements
└── README.md               # Project documentation
```

---

## 📊 Results

(Coming soon: experimental results and evaluation metrics)

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgements

* [PyTorch](https://pytorch.org/)
* [DHG: Deep Hypergraph Library](https://github.com/iMoonLab/Deep-Hypergraph)

```

这样一整段全是 **Markdown**，直接复制保存成 `README.md` 就能用了。  

要不要我再帮你在 **Results** 那一节加一个 **表格模板**（RMSE / MAE / R² 三列），方便你以后直接填实验结果？
```
