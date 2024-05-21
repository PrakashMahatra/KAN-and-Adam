
# Kolmogorov-Arnold Networks (KAN) and Adam Optimizer Implementations

This repository contains Python implementations of Kolmogorov-Arnold Networks (KAN) and the Adam optimization algorithm. The KAN implementation includes training and evaluation of the network, and the Adam optimizer is implemented as a standalone class.

## Introduction

### Kolmogorov-Arnold Networks (KAN)

Inspired by the Kolmogorov-Arnold representation theorem, the paper Kolmogorov-Arnold Networks (KANs) provides promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation functions on edges (“weights”). KANs have no linear weights at all – every weight parameter is replaced by a univariate function parametrized as a spline. They showed that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability.

**Advantages of KANs:**

- **Accuracy**: Much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving.
- **Interpretability**: KANs can be intuitively visualized and can easily interact with human users.
- **Efficiency**: KANs possess faster neural scaling laws than MLPs.

 In summary, KANs are promising alternatives to MLPs, opening opportunities for further improving today’s deep learning models which rely heavily on MLPs.

### Adam Optimizer

The Adam (Adaptive Moment Estimation) optimizer is a popular optimization algorithm used in training machine learning models. It combines the advantages of two other extensions of stochastic gradient descent, namely AdaGrad and RMSProp.

## Requirements

- Python 3.6 or higher
- NumPy
- SciPy
- Matplotlib
- Seaborn
  

## Installation

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/PrakashMahatra/KAN-and-Adam.git
cd KAN-and-Adam
