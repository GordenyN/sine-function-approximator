# Sine Function Approximator with Neural Network (TensorFlow)

This project demonstrates how a simple fully connected neural network can learn to approximate the sine function using TensorFlow 2.

## Model
- Input: Single float (x-value)
- Output: Predicted sine of x
- Architecture:
  - Dense(32, relu)
  - Dense(64, relu)
  - Dense(1, linear)

##  Data
Synthetic sine wave generated using `numpy.linspace`.

##  Requirements
- TensorFlow
- NumPy
- Matplotlib

Install with:

```bash
pip install -r requirements.txt
