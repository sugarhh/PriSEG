# PriSEG： Requirements & Project Structure

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
