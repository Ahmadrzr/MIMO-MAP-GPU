# Maximum A Posteriori MIMO GPU Detector (MIMO-MAP-GPU)

MIMO-MAP-GPU is an open source C++/CUDA implementaion of MIMO MAP Detector.

# Features
	- Multithread C++ MIMO MAP Detector method (MAP-GPU)
	- Multithread GPU MIMO MAP Detector method (MAP-CPU)
	- Supports up to 8x8 MIMO 
	- Supports up to 256 QAM
	- Supports input Log Likelihood Ratio (LLR) for each bit
	- Returns output LLR for each transmitted bit
	- Supports Nvidia's GPUs
	- Benchmarking code included

# System Model
	- MAP-GPU and MAP-CPU solve the following MIMO problem
		- y = Hs + n where s is Ntx-sample transmit symbol vector, H is Nrx by Ntx MIMO channel matrix, n is Nrx-sample additive Guassian noise vector, and y is Nrx-sample received signal vector
		- Calculates output LLR for each bit using equation (12) of "J. C. Hedstrom, C. H. Yuen, R. R. Chen and B. Farhang-Boroujeny, "Achieving Near MAP Performance With an Excited Markov Chain Monte Carlo MIMO Detector," in IEEE Transactions on Wireless Communications, vol. 16, no. 12, pp. 7718-7732, Dec. 2017."


# Requirements
	- NVIDIA Graphics Card
	- NVIDIA Graphics driver 390
	- CUDA 9.1
	- OpenMP

# Installation on Ubuntu 18.04
	- NVIDIA driver 390
		- In software & updates, select the restricted and multiverse repositories
		- In the Additional Drivers tab in software & updates select the NVIDIA proprietary driver (390 for CUDA 9)
	- CUDA 9.1
		- sudo apt update
		- sudo apt install nvidia-cuda-toolkit
	- OpenMP
		- sudo apt install libomp-dev

# Build Instruction
	- Makefile

# Test
	- Build the project by make command
	- run test_map_det.out
		- The test code runs MAP-CPU and MAP-GPU detectors and compare their processing times. 
		- MAP-CPU and MAP-GPU run 20 MIMO blocks each with 4 transmit antennas, 5 receive antennas, and 64-QAM modulation
		- test_map_det accepts following parameters: test_map_det.out blocks Ntx Nrx Mc number_of_threads 
		- Sample output is:
			=============================================================
			Device 0: "GeForce GTX 760"
			  CUDA Driver Version / Runtime Version          9.1 / 9.1
			  CUDA Capability Major/Minor version number:    3.0
			  Total amount of global memory:                 1992 MBytes (2088632320 bytes)
			  GPU Max Clock rate:                            1150 MHz (1.15 GHz)
			  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
			  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
			  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
			  Total amount of constant memory:               65536 bytes
			  Total amount of shared memory per block:       49152 bytes
			  Total number of registers available per block: 65536
			  Warp size:                                     32
			  Maximum number of threads per multiprocessor:  2048
			  Maximum number of threads per block:           1024
			  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
			  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
			  Maximum memory pitch:                          2147483647 bytes
			  Texture alignment:                             512 bytes
			  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
			  Run time limit on kernels:                     Yes
			  Integrated GPU sharing Host Memory:            No
			  Support host page-locked memory mapping:       Yes
			  Alignment requirement for Surfaces:            Yes
			  Device has ECC support:                        Disabled
			  Device supports Unified Addressing (UVA):      Yes
			  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
			=============================================================
			Configuration
			blocks = 128
			 Ntx   = 4
			 Nrx   = 4
			  Mc   = 64 
			=============================================================
			GPU MAP DET started ... Done
				time = 7441 msec
			CPU MAP DET started ... Done
				time = 146220 msec
			Speed up = 19.7   
			=============================================================
			Total number of bits  = 3072
			Mean absolute error   = 0.00302
			=============================================================

# Target Platforms
	- Linux

# Copyright
	Copyright 2018 Ahmad RezazadehReyhani

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

# Acknowledgment
	Grateful appreciation to the Farhang Wireless Inc. for their support and generously funding the implementation of this library.
  (http://farhangwireless.com)

