# Deep Network Architectures: Overcoming Vanishing Gradients in VGG Models

## Description
This project investigates the challenges of training very deep neural networks, specifically the **VGG38 model**, which suffers from the vanishing gradient problem. The project explores and demonstrates the application of **Batch Normalization (BN)** and **Residual Connections (RC)** to mitigate this issue. Experiments are conducted on the CIFAR100 dataset to evaluate the effectiveness of these techniques in improving training stability and model performance.

---

## Goals

1. **Analyze Vanishing Gradient Problem:** Diagnose the issue in VGG38 by monitoring gradient flow and comparing it with the shallower VGG08.
2. **Apply Batch Normalization:** Test its effect on stabilizing activations and improving convergence.
3. **Apply Residual Learning:** Introduce skip connections to simplify optimization and enhance training depth.
4. **Combine BN and RC:** Achieve optimal performance by leveraging the benefits of both methods.

---

## Experiments and Results

- **Models:** VGG08, VGG38, VGG38 with Batch Normalization (BN), Residual Connections (RC), and a combination of both (BN + RC).
- **Dataset:** CIFAR100 (60,000 32x32 images across 100 classes).

