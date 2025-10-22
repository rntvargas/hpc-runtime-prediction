# Machine Learning-Based Runtime Prediction and Energy Optimization for HPC Job Scheduling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Overview

This repository contains the complete implementation and reproducible results for the research paper:

**"Machine Learning-Based Runtime Prediction and Energy Optimization for HPC Job Scheduling Using the NREL Eagle Supercomputer Dataset"**

**Authors:**  
- Renato Quispe Vargas (Universidad Nacional del Altiplano, Peru) - 72535253@est.unap.edu.pe  
- Briggitte Jhosselyn Vilca Chambilla (Universidad Nacional del Altiplano, Peru) - 71639757@unap.edu.pe  
- Fred Torres Cruz (Universidad Nacional del Altiplano, Peru) - ftorres@unap.edu.pe 

##  Abstract

High-Performance Computing (HPC) systems face significant challenges in job scheduling due to inaccurate runtime predictions, leading to resource underutilization and high energy consumption. This paper presents a comprehensive machine learning approach for predicting job runtime and optimizing energy usage on the NREL Eagle supercomputer using a real-world dataset of over 11 million jobs spanning November 2018 to February 2023.

**Key Findings:**
- Random Forest achieves MAE of 1.91 hours with R² = 0.450
- 91.5% weighted average energy savings potential identified
- Time limit (user request) is the most important predictor (56.8%)
- Model performs excellently on short jobs (<1h, MAE=1.07h)

##  Repository Structure

```
hpc-runtime-prediction/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation information
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   └── generate_figures.py      # Main analysis script
│
├── figures/                     # Generated figures (publication quality)
│   ├── scatter.png              # Actual vs Predicted runtime
│   ├── importance.png           # Feature importances
│   ├── energy_distribution.png  # Energy savings distribution
│   └── error_analysis.png       # Error analysis by runtime range
│
├── data/                        # Dataset information
│   └── README.md                # Dataset download instructions
│
└── paper/                       # Research paper
    └── paper.pdf                # Published paper (add when available)
```

##  Dataset

This research uses the **NREL Eagle Supercomputer Dataset** containing 11,030,377 HPC jobs from November 2018 to February 2023.

### Download Instructions

1. **Visit:** [https://data.openei.org/submissions/5860](https://data.openei.org/submissions/5860)
2. **Download:** `eagle_data.parquet` (253 MB, recommended) or `eagle_data.csv.bz2` (115 MB)
3. **Place:** In the project root directory

**Dataset Citation:**
```
National Renewable Energy Laboratory (NREL). (2023). 
NREL Eagle Supercomputer Jobs Dataset. 
OpenEI. https://data.openei.org/submissions/5860
```

### Dataset Statistics (After Preprocessing)
- **Total jobs analyzed:** 213,362 (stratified sample)
- **Median runtime:** 10.5 minutes
- **Mean utilization:** 18.01% (indicating significant over-provisioning)
- **Jobs with <60% utilization:** 90.8%

##  Installation & Usage

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- ~300MB free disk space (for dataset)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hpc-runtime-prediction.git
cd hpc-runtime-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the Eagle dataset (see instructions above)
# Place eagle_data.csv.bz2 or eagle_data.parquet in project root
```

### Running the Analysis

```bash
# Generate all figures and results
python src/generate_figures.py
```

**Output:**
- 4 publication-quality figures saved in `figures/`
- Performance metrics printed to console
- Model training and evaluation results

**Expected Runtime:** ~5-10 minutes (depending on hardware)

##  Results Summary

### Model Performance (Table II from paper)

| Model | MAE (hours) | RMSE (hours) | R² Score |
|-------|-------------|--------------|----------|
| Mean Predictor (baseline) | 3.19 | 5.19 | -0.000 |
| User Estimate (0.4×limit) | 10.75 | 22.30 | -17.438 |
| **Random Forest** | **1.91** | **3.85** | **0.450** |
| **Gradient Boosting** | **1.95** | **3.84** | **0.455** |

### Feature Importance

| Feature | Importance | Percentage |
|---------|------------|------------|
| **Time Limit (user request)** | 0.568 | **56.8%** |
| Partition | 0.168 | 16.8% |
| Memory per CPU | 0.133 | 13.3% |
| CPUs per Task | 0.113 | 11.3% |
| Number of Tasks | 0.018 | 1.8% |

### Energy Optimization Results

- **Weighted energy savings:** 91.5%
- **Jobs with >40% savings:** 95.7%
- **Jobs with 15-40% savings:** 3.7%
- **Jobs with <15% savings:** 0.6%

### Error Analysis by Runtime Range

| Runtime Range | Median Error | MAE | Sample Size |
|---------------|--------------|-----|-------------|
| Short (<1h) | 0.14h | 1.07h | 29,633 jobs |
| Medium (1-10h) | -0.39h | 1.95h | 9,597 jobs |
| Long (>10h) | -8.40h | 9.10h | 3,443 jobs |

**Key Insight:** Model performs excellently for short jobs but underestimates long-running jobs.

##  Figures

All figures are publication-quality (300 DPI) and saved in the `figures/` directory:

1. **scatter.png** - Scatter plot showing actual vs predicted runtime with trend line
2. **importance.png** - Horizontal bar chart of feature importances
3. **energy_distribution.png** - Histogram of energy savings distribution
4. **error_analysis.png** - Box plots of prediction errors by runtime range

## 🔧 Code Structure

### Main Script: `src/generate_figures.py`

The script performs the following steps:

1. **Data Loading** - Loads NREL Eagle dataset (11M+ jobs)
2. **Preprocessing** - Cleans data, converts units, derives features
3. **Feature Engineering** - Creates ntasks, cpus_per_task, mem_per_cpu
4. **Model Training** - Trains Random Forest and Gradient Boosting models
5. **Evaluation** - Calculates MAE, RMSE, R² metrics
6. **Visualization** - Generates 4 publication-quality figures
7. **Energy Analysis** - Computes potential energy savings

**Key Functions:**
- `parse_wallclock()` - Parses time limit format (HH:MM:SS or D-HH:MM:SS)
- `parse_memory()` - Parses memory format (4G, 4000M, etc.)
- Feature importance analysis
- Energy optimization calculations with 10% safety margin

##  Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@article{quispe2025hpc,
  title={Machine Learning-Based Runtime Prediction and Energy Optimization for HPC Job Scheduling Using the NREL Eagle Supercomputer Dataset},
  author={Quispe Vargas, Renato and Torres Cruz, Fred and Vilca Chambilla, Briggitte Jhosselyn},
  journal={[Journal Name]},
  year={2025},
  volume={XX},
  pages={XX--XX},
  publisher={[Publisher]},
  doi={10.XXXX/xxxxxx}
}
```

**For the dataset:**
```bibtex
@misc{nrel2023eagle,
  title={NREL Eagle Supercomputer Jobs Dataset},
  author={{National Renewable Energy Laboratory}},
  year={2023},
  publisher={OpenEI},
  url={https://data.openei.org/submissions/5860}
}
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Summary:** You are free to use, modify, and distribute this code for academic and commercial purposes with attribution.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Areas for improvement:**
- Additional ML models (XGBoost, Neural Networks)
- Hyperparameter optimization
- Feature engineering improvements
- Support for other HPC datasets

##  Contact

**Renato Quispe Vargas**  
Universidad Nacional del Altiplano  
Email: 72535253@est.unap.edu.pe

**Briggitte Jhosselyn Vilca Chambilla**  
Universidad Nacional del Altiplano  
Email: 71639757@unap.edu.pe

**Fred Torres Cruz**  
Universidad Nacional del Altiplano  
Email: ftorres@unap.edu.pe

##  Acknowledgments

- National Renewable Energy Laboratory (NREL) for providing the Eagle dataset
- Universidad Nacional del Altiplano, Puno, Peru
- Open source community for excellent ML libraries (scikit-learn, pandas, matplotlib)

##  Related Publications

[Add links to related work or preprints here]

##  Links

- **Paper:** [DOI Link] (add when published)
- **Dataset:** [https://data.openei.org/submissions/5860](https://data.openei.org/submissions/5860)
- **Conference/Journal:** [Add link]

---

**Keywords:** HPC, Job Scheduling, Machine Learning, Random Forest, Energy Optimization, Runtime Prediction, Eagle Supercomputer, NREL

