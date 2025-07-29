# 🏁 F1 Race Winner Prediction System

> **Multi-Season ML Model with 86.18% ROC AUC**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ROC AUC](https://img.shields.io/badge/ROC_AUC-86.18%25-brightgreen.svg)](#performance)

## 🚀 **Overview**
Advanced machine learning system that predicts F1 race winners using multi-season training (2023+2024) to forecast 2025 races. Features temporal validation, 57 engineered features, and incremental learning capabilities.

## 📊 **Performance**
- **86.18% ROC AUC** with temporal validation
- **1,177 race records** from 2023-2025 seasons
- **57 engineered features** with cross-season validation
- **Live 2025 predictions** for ongoing championship

## 🏆 **Key Features**
✅ **Live Race Predictions** - Forecast 2025 F1 race winners  
✅ **Multi-Season Learning** - Learns from dominance + competition patterns  
✅ **Temporal Validation** - Proper past→future methodology  
✅ **Production Ready** - Complete deployment package  
```
## 🚀 **Quick Start**
```git clone https://github.com/ArhaanAli04/f1-race-predictor.git ```
```cd f1-race-predictor```
```pip install -r requirements.txt```

Run notebooks in order: `01_data_exploration.ipynb` → `02_feature_engineering.ipynb` → `03_model_development.ipynb`

```

## 🎯 **Technical Approach**
- **Training**: 2023+2024 F1 seasons (918 records)
- **Testing**: 2025 season (259 records through July)
- **Algorithm**: Tuned Random Forest with hyperparameter optimization
- **Features**: Driver performance, team strength, championship context

## 📈 **Results**
| Model | ROC AUC | Strengths |
|-------|---------|-----------|
| **Tuned Random Forest** | **86.18%** | **Best overall** |
| Random Forest | 88.32% | Cross-validation |
| Logistic Regression | 86.26% | Interpretable |
| XGBoost | 80.60% | Feature insights |

## 🔮 **Applications**
- Fantasy F1 winner predictions
- Sports betting analysis  
- Media race previews
- Team strategy insights

## 🛠️ **Technologies**
Python • Scikit-learn • Pandas • XGBoost • Jupyter

## 📄 **License**
MIT License

## 📞 **Contact**
- Email: arhaan.ali2004@gmail.com

---
*F1 Race Winner Prediction with Advanced Machine Learning*
