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
```

### Repository Structure
* `files/`    - Shared keys, IP addresses and data files.
* `files/preload`    - Contains data for pretrained network from SecureML. The other networks can be generated using `scripts` and functions in `secondary.cpp`
* `lib_eigen/`    - [Eigen library](http://eigen.tuxfamily.org/) for faster matrix multiplication.
* `src/`    - Source code.
* `util/` - Dependencies for AES randomness.
* `scripts/` - Contains python code to generate trained models for accuracy testing over a batch.
* The `god` script makes remote runs simpler (as well as the `makefile`)

### Building the Code
```bash
cd PriSEG
make all -j$(nproc)
```

### Running the Code

To run the code, simply choose one of the following options: 

* `make`: Prints all the possible makefile options.
* `make terminal`: Runs the 3PC code on localhost with output from $P_0$ printed to standard output.
* `make file`: : Runs the 3PC code on localhost with output from $P_0$ printed to a file (in `output/3PC.txt`)
* `make command`: Enables running a specific network, dataset, adversarial model, and run type (localhost/LAN/WAN) specified through the `makefile`. This takes precedence over choices in the `src/main.cpp` file.

### Work in Progress

The documentation is currently being refined. Please check back soon or contact the maintainers for specific queries.
