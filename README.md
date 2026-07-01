# SDN-Intrusion-Detection-DeepLearning
Deep learning-based intrusion detection system for Software-Defined Networks (SDN) using LSTM, CNN, and Hybrid CNN-LSTM models.
# Deep Learning-Based Intrusion Detection for Software Defined Networks (SDN)

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</p>

---

## Overview

Software Defined Networking (SDN) has transformed modern computer networks by separating the control plane from the data plane, making networks more flexible, programmable, and easier to manage. However, this centralized architecture also introduces new security challenges, making effective Intrusion Detection Systems (IDS) an essential component of SDN environments.

This project investigates the effectiveness of deep learning techniques for detecting malicious network traffic in SDN environments. Three neural network architectures are implemented and evaluated:

* Long Short-Term Memory (LSTM)
* Convolutional Neural Network (CNN)
* Hybrid CNN-LSTM

The models are trained using a balanced dataset generated with ADASYN oversampling and evaluated using several classification metrics, including Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.

The hybrid CNN-LSTM model achieved the highest overall performance among the evaluated architectures.

---

## Project Objectives

The main objectives of this project are:

* Develop an intrusion detection system for Software Defined Networks.
* Compare different deep learning architectures.
* Investigate the effectiveness of CNN feature extraction.
* Evaluate sequential learning using LSTM.
* Compare model performance using multiple evaluation metrics.
* Identify the most suitable architecture for SDN intrusion detection.

---

## Key Features

* Deep Learning-based Intrusion Detection System
* Binary Classification (Normal vs Attack)
* Three Deep Learning Architectures
* Data Balancing using ADASYN
* Standardized Feature Scaling
* Performance Evaluation with Multiple Metrics
* Confusion Matrix Visualization
* ROC-AUC Analysis
* Training History Visualization
* Reproducible Experiments

---

## Dataset

The dataset used in this project was created by combining traffic collected from multiple SDN environments.

### Data Sources

* Normal SDN Traffic
* Metasploitable Attack Traffic
* Open vSwitch (OVS) Traffic

The final dataset was converted into a binary classification problem consisting of:

| Class  | Description                |
| ------ | -------------------------- |
| Normal | Legitimate Network Traffic |
| Attack | Malicious Network Traffic  |

Categorical attributes were removed during preprocessing, and only numerical features were retained for model training.

---

## Data Preprocessing Pipeline

The preprocessing workflow consists of the following stages:

```
Raw Dataset
      │
      ▼
Merge Multiple Datasets
      │
      ▼
Remove Non-Numerical Features
      │
      ▼
Binary Label Encoding
      │
      ▼
Feature Scaling
(StandardScaler)
      │
      ▼
Class Balancing
(ADASYN)
      │
      ▼
Model Training
      │
      ▼
Performance Evaluation
```

---

## Repository Structure

```
SDN-Intrusion-Detection-DeepLearning
│
├── README.md
├── requirements.txt
│
├── data
│   └── dataset_information.txt
│
└── notebooks
    ├── info.ipynb
    ├── lstm.ipynb
    ├── cnn.ipynb
    └── lstm+cnn.ipynb
```

---

## Workflow

The overall workflow implemented in this repository is illustrated below.

```
               Dataset Collection
                      │
                      ▼
             Data Preprocessing
                      │
                      ▼
              Feature Standardization
                      │
                      ▼
             Class Balancing (ADASYN)
                      │
                      ▼
          Deep Learning Model Training
          ┌────────┬────────┬────────┐
          │        │        │
          ▼        ▼        ▼
        LSTM      CNN   CNN-LSTM
          │        │        │
          └────────┴────────┘
                   │
                   ▼
         Performance Evaluation
                   │
                   ▼
      Accuracy • Precision • Recall
      F1-score • ROC-AUC • Confusion Matrix
```
## Project Highlights

This project demonstrates the application of deep learning techniques for intrusion detection in Software Defined Networks (SDN). Several neural network architectures were implemented and experimentally compared using the same preprocessing pipeline and evaluation metrics.

### Highlights

* Deep Learning-based Intrusion Detection System (IDS)
* Binary Classification (Normal vs Attack)
* Comparison of Three Deep Learning Architectures
* Data Balancing with ADASYN
* Standardized Feature Scaling
* Comprehensive Performance Evaluation
* Hybrid CNN-LSTM Achieved the Best Performance
* Fully Implemented in Python using TensorFlow/Keras

---

## Performance Summary

| Model           |   Accuracy | Overall Rank |
| --------------- | ---------: | -----------: |
| Hybrid CNN-LSTM | **96.84%** |           1 |
| LSTM            | **95.44%** |           2 |
| CNN             | **95.28%** |           3 |

The experimental results indicate that combining convolutional feature extraction with sequential learning improves intrusion detection performance compared with individual CNN or LSTM models.

---

## Future Improvements

Several enhancements can be explored in future work:

* Multi-class intrusion detection
* Cross-validation for more robust evaluation
* Hyperparameter optimization
* Transformer-based architectures
* Explainable AI (XAI) techniques
* Real-time SDN traffic monitoring
* Deployment using Flask or FastAPI
* Integration with SDN controllers

---

## Reproducibility

To reproduce the experiments:

1. Install the required dependencies.
2. Open the notebooks in the recommended order.
3. Execute each notebook from top to bottom.
4. Compare the evaluation metrics generated for each model.

Random seeds were fixed during training to improve reproducibility.

---

## Repository Structure

```text
SDN-Intrusion-Detection-DeepLearning/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── dataset_information.txt
│
└── notebooks/
    ├── info.ipynb
    ├── lstm.ipynb
    ├── cnn.ipynb
    └── lstm+cnn.ipynb
```

---

## Author

**Hodeis**

GitHub:
https://github.com/hodeis99

---

## Citation

If you find this repository useful for your research or academic work, please consider citing it.

```bibtex
@misc{SDN_IDS_DeepLearning,
  author = {Hodeis},
  title = {Deep Learning-Based Intrusion Detection for Software Defined Networks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/hodeis99/SDN-Intrusion-Detection-DeepLearning}
}
```

---

## License

This project is released under the MIT License.

You are free to use, modify, and distribute this project for academic and research purposes.

---

## Acknowledgements

This project was developed to investigate the effectiveness of deep learning architectures for Software Defined Network intrusion detection.

Special thanks to the developers and maintainers of the following open-source projects:

* TensorFlow
* Keras
* Scikit-Learn
* Imbalanced-Learn
* NumPy
* Pandas
* Matplotlib

Their outstanding work made this project possible.

---

## Final Remarks

Deep learning has become an important approach for modern intrusion detection systems due to its ability to automatically learn complex traffic patterns from network data.

This repository provides a comparative implementation of three widely used neural network architectures for SDN intrusion detection. The experimental results demonstrate that the Hybrid CNN-LSTM architecture achieved the best overall performance, highlighting the benefits of combining convolutional feature extraction with sequential learning.

It is hoped that this repository will serve as a useful reference for students, researchers, and practitioners interested in cybersecurity, software-defined networking, and deep learning applications.

---

**If you found this project useful, consider giving it a star on GitHub.**
