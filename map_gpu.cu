//   Copyright:
//       Copyright 2018 Ahmad RezazadehReyhani
//
//       Licensed under the Apache License, Version 2.0 (the "License");
//       you may not use this file except in compliance with the License.
//       You may obtain a copy of the License at
//
//          http://WWW.apache.org/licenses/LICENSE-2.0
//
//       Unless required by applicable law or agreed to in writing, software
//       distributed under the License is distributed on an "AS IS" BASIS,
//       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//       See the License for the specific language governing permissions and
//       limitations under the License.
//
//   Acknowledgment:
//       Grateful appreciation to the Farhang Wireless Inc. for their support and generously funding the implementation of this library.
//       http://farhangwireless.com/

#include "map_det.h"

// Initializes the device and reads its parameters
__host__ void InitDevice()
{
	cudaDeviceProp deviceProp;
	int driverVersion = 0, runtimeVersion = 0;
	cudaSetDevice(0);
	cudaDeviceReset();
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device %d: \"%s\"\n", 0, deviceProp.name);

	// Console log
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

	char msg[256];
	sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
	printf("%s", msg);
	//printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
	//      deviceProp.multiProcessorCount,
	//      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
	//      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
	printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
	printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
	       deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
	       deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
	printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
	       deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
	printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
	       deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


	printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
	       deviceProp.maxThreadsDim[0],
	       deviceProp.maxThreadsDim[1],
	       deviceProp.maxThreadsDim[2]);
	printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
	       deviceProp.maxGridSize[0],
	       deviceProp.maxGridSize[1],
	       deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
	printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
	printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
	printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
	printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
	printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
	printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
	printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
	printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
	printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
}

// MAX-LOG method on device processor
__device__ GpuReal maxlogG(GpuReal a, GpuReal b, char maxlog_type)
{
	if (maxlog_type == 0)
		return (GpuReal)fmaxf((double)a, (double)b);
	else
	{
		double c = fabsf(a - b);
		if (c > 100)
			return (GpuReal)fmaxf((double)a, (double)b);
		else
			return (GpuReal)fmaxf((double)a, (double)b) + logf(1 + expf(-c));
	}
}

// MAX-LOG method on host processor
__host__ GpuReal maxlogH(GpuReal a, GpuReal b, char maxlog_type)
{
	if (maxlog_type == 0)
		return (GpuReal)fmaxf((double)a, (double)b);
	else
	{
		double c = fabsf(a - b);
		if (c > 100)
			return (GpuReal)fmaxf((double)a, (double)b);
		else
			return (GpuReal)fmaxf((double)a, (double)b) + logf(1 + expf(-c));
	}
}

// The kernel code that runs on GPU device
__global__ void MapKernel(int NumTx, int NumRx, int Mc,
                          const char* __restrict__ Dev_PAM,
                          const GpuComplex* __restrict__ Dev_H, const  GpuComplex* __restrict__ Dev_y, const GpuReal* __restrict__ Dev_La,
                          GpuReal* Dev_MaxP, GpuReal* Dev_MaxN, char maxlog_type, const int block)
{
	const int BitsPerThread = (int)NumTx * log2((float)Mc) - (int)log2((float)gridDim.x * blockDim.x);

	const unsigned char Mcb  = (char)log2((float)Mc);
	const char Maskh = sqrt((GpuReal)Mc) - 1;

	const int NumTxMcbh2 = NumTx * Mcb;

	GpuComplex YHS[8];
	__shared__ char BitsPrev[1024 * 8];
	__shared__ GpuReal SumLa[1024];
	__shared__ GpuReal Temp[1024];


	char BitsCurr;

	const int  blockNumRxNumTx = block * NumRx * NumTx;
	const int  blockNumTxMcbh2 = block * NumTx * Mcb;

	SumLa[threadIdx.x] = 0;

	for (int ix = 0; ix < NumRx; ix++)
		YHS[ix] = Dev_y[block * NumRx + ix];

	for (uint64_t work = 0; work < (0x1 << BitsPerThread); work++)
	{
		uint64_t CurrentPoint  = ((blockDim.x * blockIdx.x + threadIdx.x) << BitsPerThread) + work;
		for (int ix = 0; ix < NumTx; ix++)
		{
			BitsCurr     = CurrentPoint & (Mc - 1);
			CurrentPoint = CurrentPoint >> (Mcb);
			if (BitsCurr != BitsPrev[ix * blockDim.x + threadIdx.x] || work == 0)
			{
				if (work == 0)
					for (int ir = 0; ir < NumRx; ir++)
						YHS[ir] = GpuAdd(YHS[ir], GpuMul(Dev_H[blockNumRxNumTx + ir * NumTx + NumTx - 1 - ix], GpuMakeComplex(-Dev_PAM[(BitsCurr >> (Mcb >> 1)) & Maskh], -Dev_PAM[BitsCurr & Maskh])));

				else
					for (int ir = 0; ir < NumRx; ir++)
						YHS[ir] = GpuAdd(YHS[ir], GpuMul(Dev_H[blockNumRxNumTx + ir * NumTx + NumTx - 1 - ix], GpuMakeComplex(Dev_PAM[(BitsPrev[ix * blockDim.x + threadIdx.x] >> (Mcb >> 1)) & Maskh] - Dev_PAM[(BitsCurr >> (Mcb >> 1)) & Maskh], Dev_PAM[BitsPrev[ix * blockDim.x + threadIdx.x] & Maskh] - Dev_PAM[BitsCurr & Maskh])));


				if (work == 0)
					for (int ir = 0; ir < Mcb; ir++)
						SumLa[threadIdx.x] = SumLa[threadIdx.x] + (2 * ((BitsCurr   >> ir) & 0x1) - 1) * Dev_La[blockNumTxMcbh2 + NumTxMcbh2 - 1 - (ix * Mcb + ir)];
				else
					for (int ir = 0; ir < Mcb; ir++)
						SumLa[threadIdx.x] = SumLa[threadIdx.x] + ((2 * ((BitsCurr   >> ir) & 0x1) - 1) - (2 * ((BitsPrev[ix * blockDim.x + threadIdx.x] >> ir) & 0x1) - 1)) * Dev_La[blockNumTxMcbh2 + NumTxMcbh2 - 1 - (ix * Mcb + ir)];
				BitsPrev[ix * blockDim.x + threadIdx.x] = BitsCurr;
			}
		}

		Temp[threadIdx.x] = SumLa[threadIdx.x];
		for (int ix = 0; ix < NumRx; ix++)
			Temp[threadIdx.x] -= YHS[ix].x * YHS[ix].x + YHS[ix].y * YHS[ix].y;

		for (int ix = 0; ix < NumTx; ix++)
		{
			for (int ir = 0; ir < Mcb; ir++)
			{
				if ((BitsPrev[ix * blockDim.x + threadIdx.x] >> ir) & 0x1 == 1)
					Dev_MaxP[(blockNumTxMcbh2 + ix * Mcb + ir)*gridDim.x * blockDim.x + (blockDim.x * blockIdx.x + threadIdx.x)] = maxlogG(
					            Dev_MaxP[(blockNumTxMcbh2 + ix * Mcb + ir) * gridDim.x * blockDim.x + (blockDim.x * blockIdx.x + threadIdx.x)], Temp[threadIdx.x] - Dev_La[blockNumTxMcbh2 + NumTxMcbh2 - 1 - (ix * Mcb + ir)], maxlog_type);
				else
					Dev_MaxN[(blockNumTxMcbh2 + ix * Mcb + ir)*gridDim.x * blockDim.x + (blockDim.x * blockIdx.x + threadIdx.x)] = maxlogG(
					            Dev_MaxN[(blockNumTxMcbh2 + ix * Mcb + ir) * gridDim.x * blockDim.x + (blockDim.x * blockIdx.x + threadIdx.x)], Temp[threadIdx.x] + Dev_La[blockNumTxMcbh2 + NumTxMcbh2 - 1 - (ix * Mcb + ir)], maxlog_type);
			}
		}
	}

}

// method that transfers input data to GPU, calls GPU kernel code, and returns output LLRs
__host__ void map_det_gpu(Mdc* y, Mdc* H, Md* La, Md* Le, int Mc, double sig2, enum Maxlog_type maxlog_type, int nthreads) {
	// y 			received signal vector
	// H    		MIMO channel matrix
	// La   		input LLR vector
	// Le   		output LLR vector
	// Mc   		Constellation size
	// Sig2 		Noise variance
	// maxlog_type  MAX-LOG or LOG-MAP selector
	// nthreads     Number of threads

	// Dev_... 		variables on device memory
	// Host_...	    variables on system memory

	if (nthreads == 0) {
		nthreads = omp_get_num_procs() / 2;
	}

	int tblock = y->dim[0];
	int NumRx = H->dim[1];
	int NumTx = H->dim[2];
	int Mcb   = (int)log2((GpuReal)Mc);
	int Mcbh  = Mcb / 2;

	cudaError Res = cudaSuccess;

	char* Dev_PAM;
	switch (Mcb)
	{
	case 2 :
		Res = cudaMalloc((void**)&Dev_PAM, 2 * sizeof(char));
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_PAM, %lu bytes\n", 2 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		Res = cudaMemcpy(Dev_PAM, PAM_2, 2 * sizeof(char), cudaMemcpyHostToDevice);
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_PAM, %lu bytes\n", 2 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		break;
	case 4 :
		Res = cudaMalloc((void**)&Dev_PAM, 4 * sizeof(char));
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_PAM, %lu bytes\n", 4 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		Res = cudaMemcpy(Dev_PAM, PAM_4, 4 * sizeof(char), cudaMemcpyHostToDevice);
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_PAM, %lu bytes\n", 4 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		break;
	case 6 :
		Res = cudaMalloc((void**)&Dev_PAM, 8 * sizeof(char));
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_PAM, %lu bytes\n", 8 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		Res = cudaMemcpy(Dev_PAM, PAM_6, 8 * sizeof(char), cudaMemcpyHostToDevice);
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_PAM, %lu bytes\n", 8 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		break;
	case 8 :
		Res = cudaMalloc((void**)&Dev_PAM, 16 * sizeof(char));
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_PAM, %lu bytes\n", 16 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		Res = cudaMemcpy(Dev_PAM, PAM_8, 16 * sizeof(char), cudaMemcpyHostToDevice);
		if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_PAM, %lu bytes\n", 16 * sizeof(char)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		break;
	default:
	{printf("Invalid Mcb value\n"); exit(1);}
	}

	GpuReal E_M = 1; // Due to integer PAM structure
	GpuReal Sig = sqrt(sig2);

	GpuComplex* Host_H;
	GpuComplex* Dev_H;

	Host_H = (GpuComplex*)malloc(tblock * NumRx * NumTx * sizeof(GpuComplex));
	if (Host_H == NULL) {printf("Memory allocation error, malloc, Host_H, %lu bytes\n", tblock * NumRx * NumTx * sizeof(GpuComplex)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	int ix;
	#pragma omp parallel for num_threads(nthreads) private(ix)
	for (ix = 0; ix < tblock * NumRx * NumTx; ix++)
		Host_H[ix] = GpuMakeComplex((GpuReal)creal(H->data[ix]) / E_M / Sig, (GpuReal)cimag(H->data[ix]) / E_M / Sig); // apply normalization to H instead of s

	Res = cudaMalloc((void**)&Dev_H, tblock * NumRx * NumTx * sizeof(GpuComplex));
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_H, %lu bytes\n", tblock * NumRx * NumTx * sizeof(GpuComplex)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	Res = cudaMemcpy(Dev_H, Host_H, tblock * NumRx * NumTx * sizeof(GpuComplex), cudaMemcpyHostToDevice);
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_H, %lu bytes\n", tblock * NumRx * NumTx * sizeof(GpuComplex)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	free(Host_H);

	GpuComplex* Host_y;
	GpuComplex* Dev_y;

	Host_y = (GpuComplex*)malloc(tblock * NumRx * sizeof(GpuComplex));
	if (Host_y == NULL) {printf("Memory allocation error, malloc, Host_y, %lu bytes\n", tblock * NumRx * sizeof(GpuComplex)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	#pragma omp parallel for num_threads(nthreads) private(ix)
	for (ix = 0; ix < tblock * NumRx; ix++)
		Host_y[ix] = GpuMakeComplex((GpuReal)creal(y->data[ix]) / Sig, (GpuReal)cimag(y->data[ix]) / Sig);

	Res = cudaMalloc((void**)&Dev_y, tblock * NumRx * sizeof(GpuComplex));
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_y, %lu bytes\n", tblock * NumRx * sizeof(GpuComplex)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	Res = cudaMemcpy(Dev_y, Host_y, tblock * NumRx * sizeof(GpuComplex), cudaMemcpyHostToDevice);
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_y, %lu bytes\n", tblock * NumRx * sizeof(GpuComplex)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	free(Host_y);

	GpuReal* Dev_La;
	GpuReal* Host_La;
	Res = cudaMalloc((void**)&Dev_La, tblock * NumTx * Mcb * sizeof(GpuReal));
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMalloc, Dev_La, %lu bytes\n", tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	Host_La = (GpuReal*)malloc(tblock * NumTx * Mcb * sizeof(GpuReal));
	if (Host_La == NULL) {printf("Memory allocation error, malloc, Host_La, %lu bytes\n", tblock * NumTx * Mcb * sizeof(GpuReal)); exit(1);}
	#pragma omp parallel for num_threads(nthreads) private(ix)
	for (ix = 0; ix < tblock * NumTx * Mcb; ix++)
		Host_La[ix] = (GpuReal)La->data[ix] / 2.0;
	Res = cudaMemcpy(Dev_La, Host_La, tblock * NumTx * Mcb * sizeof(GpuReal), cudaMemcpyHostToDevice);
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_La, %lu bytes\n", tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}

	int G = 0;
	int T = 0;

	if (NumTx * Mcb > 14)
	{
		G = 32;
		T = 1024;
	}
	else if (NumTx * Mcb > 9)
	{
		G = 1;
		T = 1024;
	}
	else
	{
		G = 1;
		T = 1;
	}

	int retry;
	GpuReal* Dev_MaxP;
	GpuReal* Dev_MaxN;

	GpuReal* Host_MaxP = NULL;
	GpuReal* Host_MaxN = NULL;
	retry = 0;
	while (Host_MaxP == NULL)
	{
		Host_MaxP = (GpuReal*)malloc((long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
		if (Host_MaxP != NULL) break;
		retry++;
		if (retry > 50) {printf("Memory allocation error, malloc, Host_MaxP, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		usleep(100000);
	}
	retry = 0;
	while (Host_MaxN == NULL)
	{
		Host_MaxN = (GpuReal*)malloc((long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
		if (Host_MaxN != NULL) break;
		retry++;
		if (retry > 50) {printf("Memory allocation error, malloc, Host_MaxN, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		usleep(100000);
	}

	memset(Host_MaxP, 0, G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
	memset(Host_MaxN, 0, G * T * tblock * NumTx * Mcb * sizeof(GpuReal));

	Res = cudaErrorMemoryAllocation;
	retry = 0;
	while (Res != cudaSuccess)
	{
		Res = cudaMalloc((void**)&Dev_MaxP, (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
		if (Res == cudaSuccess) break;
		retry++;
		if (retry > 50) {printf("Memory allocation error, cudaMalloc, Dev_MaxP, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		usleep(100000);
	}
	Res = cudaErrorMemoryAllocation;
	retry = 0;
	while (Res != cudaSuccess)
	{
		Res = cudaMalloc((void**)&Dev_MaxN, (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
		if (Res == cudaSuccess) break;
		retry++;
		if (retry > 50) {printf("Memory allocation error, cudaMalloc, Dev_MaxN, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
		usleep(100000);
	}

	Res = cudaMemset(Dev_MaxP, -2, (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemset, Dev_MaxP, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	Res = cudaMemset(Dev_MaxN, -2, (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal));
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemset, Dev_MaxN, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}

	for (ix = 0; ix < tblock; ix++)
	{
		MapKernel <<< G, T>>>(NumTx, NumRx, Mc, Dev_PAM, Dev_H, Dev_y, Dev_La, Dev_MaxP, Dev_MaxN, (char)maxlog_type, ix);
	}

	Res = cudaGetLastError();
	if (Res != cudaSuccess)
	{
		printf("Kernel run failed: %s\n", cudaGetErrorString(Res));
		exit(1);
	}

	Res = cudaMemcpy(Host_MaxP, Dev_MaxP, (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal), cudaMemcpyDeviceToHost);
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_MaxP, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}
	Res = cudaMemcpy(Host_MaxN, Dev_MaxN, (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal), cudaMemcpyDeviceToHost);
	if (Res != cudaSuccess) {printf("Memory allocation error, cudaMemcpy, Dev_MaxN, %lu bytes\n", (long int)G * T * tblock * NumTx * Mcb * sizeof(GpuReal)); printf("%s\n", cudaGetErrorString(Res)); exit(1);}

	int block, ir, ID;
	#pragma omp parallel for num_threads(nthreads) private(block,ix,ir,ID)
	for (block = 0; block < tblock; block++)
	{
		for (ix = 0; ix < NumTx; ix++)
		{
			for (ir = 0; ir < Mcbh * 2; ir++)
			{
				for (ID = 1; ID < G * T; ID++)
				{
					Host_MaxP[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir)*G * T + 0] = maxlogH(
					            Host_MaxP[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir) * G * T + 0],
					            Host_MaxP[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir) * G * T + ID], maxlog_type);
					Host_MaxN[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir)*G * T + 0] = maxlogH(
					            Host_MaxN[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir) * G * T + 0],
					            Host_MaxN[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir) * G * T + ID], maxlog_type);
				}
				Le->data[block * NumTx * Mcbh * 2 + NumTx * Mcbh * 2 - 1 - (ix * Mcbh * 2 + ir)] = Host_MaxP[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir) * G * T] - Host_MaxN[(block * NumTx * Mcbh * 2 + ix * Mcbh * 2 + ir) * G * T];
			}
		}

	}

	cudaFree(Dev_PAM);
	cudaFree(Dev_H);
	cudaFree(Dev_y);
	cudaFree(Dev_La);
	cudaFree(Dev_MaxP);
	cudaFree(Dev_MaxN);
	free(Host_MaxP);
	free(Host_MaxN);
	free(Host_La);

	Res = cudaGetLastError();
	if (Res != cudaSuccess)
	{
		printf("Exit Error check: %s\n", cudaGetErrorString(Res));
		exit(1);
	}
}
