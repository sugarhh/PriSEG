# PriSEG： Requirements & Project Structure

### Requirements
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
* `./files/`    - Shared keys, IP addresses and data files.
* `./files/preload`    - Contains data for pretrained network from SecureML. The other networks can be generated using `scripts` and functions in `secondary.cpp`
* `./lib_eigen/`    - [Eigen library](http://eigen.tuxfamily.org/) for faster matrix multiplication.
* `./src/`    - Source code.
* `./util/` - Dependencies for AES randomness.

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


## Model Parameters & Data Preparation

PriSEG performs inference using **3-party additive secret sharing**. Before running the protocols, both model parameters and input datasets must be pre-processed into three shares.

### Model Weights (Parameters)
For U2-Net, the parameters are derived from the official `u2net.pth` weights. The plaintext model and pre-trained weights can be obtained from the official [U-2-Net Repository](https://github.com/NathanUA/U-2-Net).
* **Processing:** The weights are converted to a numerical format and split into three additive secret shares.
* **Storage Path:** These shares must be placed in: `files/preload/[NETWORK_NAME]/all_canshu/`.

### Dataset & Image Preprocessing
Input images for segmentation tasks (e.g., SOD datasets) are prepared as follows:
* **Resize:** All images are resized to 320 $\times$ 320.
* **Conversion:** Images are converted to numerical matrices and then split into three additive secret shares.
* **Storage Path:** The resulting input shares are stored in: `files/preload/[NETWORK_NAME]/all_input/`.


## Documentation Status

This documentation provides a comprehensive overview of the essential components required for understanding and reproducing our work. We will continue to provide more detailed instructions in future updates to further assist the research community.
