# Linear Regression Python Implementation

Welcome to the **Linear Regression Python Implementation** repository! This repository contains Python implementations of linear regression using various methods. The methods covered are:

1. **Least Squares** (Closed-Form)
2. **Direct and Vectorized Method**
3. **Gradient Descent**
4. **Pseudo-Inverse**
5. **Adaline** (Adaptive Linear Neuron)

## Table of Contents

- [Project Overview](#project-overview)
- [Methods Overview](#methods-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project explores different techniques for implementing linear regression models in Python. The goal is to provide a clear, educational comparison of the various methods that can be used for training linear models. The implemented methods vary from the simplest least-squares method to advanced approaches like gradient descent and Adaline.

## Methods Overview

1. **Least Squares**  
   This method computes the closed-form solution for the linear regression problem. It's based on minimizing the sum of squared differences between the predicted and actual target values.

2. **Direct and Vectorized Method**  
   A more optimized and computationally efficient approach to the least-squares problem by leveraging matrix operations. This implementation speeds up the computation for large datasets.

3. **Gradient Descent**  
   An iterative optimization algorithm used to minimize the cost function by adjusting model parameters (weights) based on the gradient of the cost function.

4. **Pseudo-Inverse**  
   This method uses the Moore-Penrose pseudo-inverse to compute the best-fit line, even when the matrix is not invertible.

5. **Adaline (Adaptive Linear Neuron)**  
   An adaptive model that adjusts weights incrementally as the data is processed. It is based on the same principle as gradient descent but is tailored for online learning scenarios.

## Installation

### Prerequisites

Make sure you have Python 3.x installed on your machine. You'll also need the following Python libraries:

- `numpy` 
- `matplotlib` (for visualization)

You can install them via pip:

```bash
pip install numpy matplotlib
