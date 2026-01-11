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

### Repository Structure
 — Source code: Contains 3PC protocol implementations and secure layer definitions.lib_eigen/
 — Eigen library: Used for optimized and fast matrix multiplications.files/
— Configuration: Shared keys, IP addresses, and data files.files/preload/
— Pre-trained models: Contains data for pretrained networks from SecureML.util/
— Utilities: Dependencies for AES randomness and cryptographic primitives.scripts/
— Python scripts: Tools to generate trained models and pre-process data for accuracy testing.god script
— Makes remote runs and coordination simpler.

### Building the Code
```bash
cd PriSEG
make all -j$(nproc)

### Running the Code
To run the code, choose one of the following options:
make terminal: Runs the 3PC code on localhost with output from $P_0$ printed to standard output.
make file: Runs the 3PC code on localhost with output from $P_0$ printed to output/3PC.txt.
make command: Enables running a specific network, dataset, and adversarial model specified through the makefile.

