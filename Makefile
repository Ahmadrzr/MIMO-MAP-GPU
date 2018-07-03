#	Copyright:
#		Copyright 2018 Ahmad RezazadehReyhani
#
#		Licensed under the Apache License, Version 2.0 (the "License");
#		you may not use this file except in compliance with the License.
#		You may obtain a copy of the License at
#
#		   http://www.apache.org/licenses/LICENSE-2.0
#
#		Unless required by applicable law or agreed to in writing, software
#		distributed under the License is distributed on an "AS IS" BASIS,
#		WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#		See the License for the specific language governing permissions and
#		limitations under the License.
#
#   Acknowledgment:
#		Grateful appreciation to the Farhang Wireless Inc. for their support and generously funding the implementation of this library.
#   	http://farhangwireless.com/

NVCCFLAGS := -O3 -Xcompiler -fopenmp -Xcompiler -march=native -maxrregcount 32 --use_fast_math -gencode arch=compute_30,code=compute_30 -D_FORCE_INLINES -Xptxas=-v

all: test_map_det.out

test_map_det.out: test_det.cpp map_gpu.cu map_cpu.cpp
	nvcc $(NVCCFLAGS) test_det.cpp map_gpu.cu map_cpu.cpp -o test_map_det.out

clean:
	rm test_map_det.out

