# PriSEG: Honest-Majority Maliciously Secure Framework for Private Neural Network Segmentation

PriSEG is a maliciously secure framework for efficient 3-party computation (3PC) protocols, specifically optimized for privacy-preserving neural network segmentation. This work is built upon the **Falcon** architecture, utilizing replicated secret sharing (RSS) to achieve high performance in the honest-majority setting.

### Requirements
---
The framework is designed to run on most Linux distributions. It has been primarily developed and tested on **Ubuntu 16.04, 18.04, and 20.04**.

**Required Packages:**
* **`g++`**: High-performance C++ compiler (supporting C++11 or later).
* **`make`**: Build automation tool.
* **`libssl-dev`**: OpenSSL development libraries for secure communication and cryptographic primitives.

Install these dependencies using your package manager:
```bash
sudo apt-get update
sudo apt-get install g++ make libssl-dev
