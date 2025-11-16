MeltBoost: Machine Learning Model for Molecular Melting Point Prediction

MeltBoost is a machine-learning framework for predicting molecular melting points using a combination of classical descriptors and a quantum-mechanical descriptor.
The project currently uses an XGBoost regression model, with a focus on interpretability, reproducibility, and extensibility for future model upgrades.

🔍 Project Overview

Predicting melting points remains a difficult problem in computational chemistry due to the combined influence of intermolecular forces, packing, symmetry, and electronic structure. MeltBoost approaches the problem through:

A curated set of 2D + 3D molecular descriptors

A quantum descriptor derived from ab initio calculations (unique aspect of this project)

A well-tuned XGBoost model for accurate regression

A modular pipeline for dataset preparation, descriptor computation, training, and evaluation

This repository aims to provide a baseline model that balances computational cost and predictive performance while leaving room for future acceleration through model stacking or surrogate ML models.

✨ Key Features
📌 Quantum Descriptor Integration (Unique Contribution)

This project includes a quantum-mechanical descriptor (e.g., HOMO/LUMO energies, dipole moment, electronic spatial extent, etc.).
This descriptor adds physics-grounded signal beyond typical cheminformatics features — improving interpretability and performance.

📌 Classical ML with XGBoost

Fast training

Handles nonlinear interactions well

Naturally suited for tabular chemical data

Feature importance visualization included

📌 Extensible Descriptor Pipeline

All descriptors are generated via a pluggable architecture, making it easy to:

Add new QM properties

Swap in ML-predicted properties in future versions

Add force-field or molecular mechanics descriptors

📌 Future Direction: ML-Accelerated Quantum Descriptors

One long-term goal of this project is to replace expensive quantum calculations with machine-learned surrogate models, reducing computation time by orders of magnitude while keeping quantum-like accuracy.

📁 Repository Structure
MeltBoost/
│
├── data/                   # Raw and processed datasets
├── descriptors/            # Descriptor computation modules
│   ├── quantum.py          # Quantum descriptor generation
│   ├── classical.py        # RDKit-based descriptors
│   └── utils.py
│
├── model/
│   └── xgboost_model.py    # Training and inference code
│
├── notebooks/
│   └── exploration.ipynb   # EDA, descriptor analysis
│
├── results/                # Outputs, metrics, feature importance
│
└── README.md

🚀 How to Use
1. Install dependencies
pip install -r requirements.txt

2. Generate descriptors
python descriptors/quantum.py
python descriptors/classical.py

3. Train the model
python model/xgboost_model.py --train

4. Run predictions
python model/xgboost_model.py --predict input.smi

📊 Performance Summary

(Tailor this once you have results.)

RMSE: X.XX

MAE: X.XX

Best Features: Quantum descriptor, molecular volume, aromaticity index, etc.

🧭 Roadmap
Short term

Add cross-validation and uncertainty estimates

Improve quantum descriptor normalization

Release pre-computed descriptors for benchmarking

Mid term

Integrate graph-based features (2D/3D GNN embeddings)

Add explainability tools (SHAP, permutation importance)

Long term (most important)

Train surrogate ML models to approximate quantum descriptors

Goal: ~100x speedup vs ab initio calculations

Integrate surrogates seamlessly into the pipeline

Explore more advanced ML models for melting point prediction
(e.g., GNNs, transformers, stacking ensembles)

📄 License

MIT License (or your preferred one)

🙌 Contributions

PRs, feature suggestions, and benchmarks are welcome!
