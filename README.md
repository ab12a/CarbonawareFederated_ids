# Carbon-Aware Federated Learning for Energy-Efficient Intrusion Detection in Distributed IoT Networks

## Project Overview

This repository contains the implementation developed for the MSc Computer Science research project:


**Carbon-Aware Federated Learning for Energy-Efficient Intrusion Detection in Distributed IoT Networks**

The project investigates whether carbon-aware scheduling can reduce unnecessary computational activity during federated learning while maintaining effective intrusion detection performance.

The proposed framework combines a Random Forest-based Intrusion Detection System (IDS) with a simulated Federated Learning (FL) environment. A carbon-aware scheduler monitors simulated carbon intensity values and allows training to proceed only when the carbon intensity remains below a predefined threshold.

---

## Features

- Centralized baseline Random Forest IDS
- Carbon-aware Federated Learning simulation
- Five simulated federated clients
- Threshold-based carbon-aware scheduling
- Automatic generation of experimental results and visualisations
- Performance evaluation using standard IDS metrics

---

## Dataset

This project uses the **NSL-KDD Intrusion Detection Dataset**.

Class labels:

- **0** – Normal network traffic
- **1** – Malicious network traffic (Attack)

The repository includes:

```text
data/
├── KDDTrain+.txt
└── KDDTest+.txt
```

Original dataset source:

https://www.unb.ca/cic/datasets/nsl.html

---

## Project Files

### `base_ids.py`

Implements the centralized Random Forest Intrusion Detection System used to establish the baseline performance.

### `final_federated_carbon_sim.py`

Implements the Carbon-Aware Federated Learning framework by:

- partitioning the dataset across five clients
- performing local Random Forest training
- simulating global model aggregation
- applying carbon-aware scheduling
- generating experimental results
- automatically creating visualisations

---

## Generated Results

Running the federated simulation automatically creates:

```text
results/
├── federated_results.csv
├── carbon_intensity_chart.png
└── accuracy_bar_chart.png
```

The baseline IDS additionally generates:

```text
results/
└── baseline_results.txt
```

---

## Evaluation Metrics

The framework evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-Score

---

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Required libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

---

## Running the Project

Run the baseline IDS:

```bash
python base_ids.py
```

Run the Carbon-Aware Federated Learning simulation:

```bash
python final_federated_carbon_sim.py
```

---

## Repository Structure

```text
CarbonawareFederated_ids/
│
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── results/
│   ├── baseline_results.txt
│   ├── federated_results.csv
│   ├── carbon_intensity_chart.png
│   └── accuracy_bar_chart.png
│
├── base_ids.py
├── final_federated_carbon_sim.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Project

This repository accompanies the MSc Computer Science project:


**Carbon-Aware Federated Learning for Energy-Efficient Intrusion Detection in Distributed IoT Networks**

The implementation demonstrates that carbon-aware scheduling can reduce unnecessary federated learning activity while maintaining intrusion detection accuracy of approximately **77–78%** in a simulated distributed IoT environment.
