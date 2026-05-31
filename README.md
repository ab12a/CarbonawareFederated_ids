# Carbon-Aware Federated Learning for Energy-Efficient Intrusion Detection in Distributed IoT Networks

## Project Overview

This project investigates whether carbon-aware scheduling can reduce the computational cost of federated learning-based intrusion detection systems while maintaining acceptable detection accuracy.

The proposed framework combines Federated Learning (FL) with a Random Forest intrusion detection model. Training rounds are executed only when simulated carbon intensity values remain below a predefined threshold.

## Dataset

NSL-KDD Intrusion Detection Dataset

- Normal traffic = 0
- Attack traffic = 1

## Project Components

### Baseline IDS
A centralized Random Forest intrusion detection system trained on the NSL-KDD dataset.

### Carbon-Aware Federated IDS
A simulated federated learning environment where clients participate in training only during low-carbon periods.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## Results

Experimental results indicate that the carbon-aware scheduling approach can reduce training activity while maintaining detection accuracy between approximately 77% and 79%.

## Repository Structure

```text
Results/
├── accuracy_plot.png
├── baseline_results.txt
└── federated_results.txt

base_ids.py
federated_carbon_sim.py
plot.py
