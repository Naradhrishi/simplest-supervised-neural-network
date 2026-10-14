# simplest-supervised-neural-network
A bare-bones single-neuron perceptron in pure Python. It uses a forward pass linear equation, manual calculus chain rule, and gradient descent to iteratively adjust weight and bias parameters until the network's prediction error hits zero—perfectly demonstrating how AI learns from scratch.

---

## 📐 Network Architecture

<!-- REPLACE THE LINE BELOW WITH YOUR ACTUAL IMAGE LINK AFTER UPLOADING TO GITHUB -->
![Neural Network Diagram](network-diagram.png)

This project simulates a **Single-Layer Perceptron** (the simplest unit of a neural network). It consists of:
* **Input (\(x = 2.0\)):** The feature processed by the network.
* **Weight (\(w\)):** The adjustable slope parameter initialized at `3.0`.
* **Bias (\(b\)):** The adjustable offset parameter initialized at `1.0`.
* **Output (\(\hat{y}\)):** The prediction node calculated as \(\hat{y} = xw + b\).
* **Target (\(y = 10.0\)):** The ground-truth correct answer.

---

## 🧠 How It Works (The 3 Phases)

The script runs a training loop for 40 iterations. Every single iteration goes through three distinct mathematical phases:

### Phase 1: The Forward Pass
The network makes a guess (\(\hat{y}\)) and calculates how far off it is using a half-Mean Squared Error equation:
\[\hat{y} = (x \cdot w) + b\]
\[\text{Error } (E) = \frac{1}{2}(\hat{y} - y)^2\]

### Phase 2: The Manual Calculus Chain Rule
Instead of using automated libraries, the code calculates exactly how the total Error changes relative to \(w\) and \(b\) using the calculus chain rule:
* **Output Error Dependency:** \(\frac{\partial E}{\partial \hat{y}} = \hat{y} - y\)
* **Weight Gradient:** \(\frac{\partial E}{\partial w} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w} = (\hat{y} - y) \cdot x\)
* **Bias Gradient:** \(\frac{\partial E}{\partial b} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial b} = (\hat{y} - y) \cdot 1\)

### Phase 3: Gradient Descent Update
The network adjusts the weight and bias by subtracting a fraction of the calculated gradients, scaled by the `learning_rate` (\(\eta = 0.1\)):
\[w = w - (\eta \cdot \frac{\partial E}{\partial w})\]
\[b = b - (\eta \cdot \frac{\partial E}{\partial b})\]

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.x** (This script runs using standard built-in Python and has **zero external library dependencies**).

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd simplest-supervised-neural-network
   ```
2. Run the script:
   ```bash
   python main.py
   ```
