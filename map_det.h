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

#include <cuComplex.h>
#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <stdint.h>

#define SINGLE_PERCISION

#ifdef SINGLE_PERCISION
#define GpuReal float
#define GpuComplex cuFloatComplex
#define GpuAdd cuCaddf
#define GpuMul cuCmulf
#define GpuMakeComplex make_cuFloatComplex
#else
#define GpuReal double
#define GpuComplex cuDoubleComplex
#define GpuAdd cuCadd
#define GpuMul cuCmul
#define GpuMakeComplex make_cuDoubleComplex
#endif

typedef std::complex<double> dcomplex;
#define creal    std::real
#define cimag    std::imag

// structures for array of double/complex/integer/long integer vectors
typedef struct Md {
	int ndim;
	int dim[4];
	double* data;
} Md;
typedef struct Mdc {
	int ndim;
	int dim[4];
	dcomplex* data;
} Mdc;
typedef struct Mi {
	int ndim;
	int dim[4];
	int* data;
} Mi;
typedef struct Ml {
	int ndim;
	int dim[4];
	long* data;
} Ml;

enum Maxlog_type {
	maxlog_max,
	maxlog_lut,
	maxlog_logmap
};

__host__ void InitDevice();
__host__ void map_det_gpu(Mdc* y, Mdc* H, Md* La, Md* Le, int Mc, double sig2, enum Maxlog_type maxlog_type, int nthreads);
__host__ void map_det_cpu(Mdc* y, Mdc* H, Md* La, Md* Le, int Mc, double sig2, enum Maxlog_type maxlog_type, int nthreads);
__host__ GpuReal maxlogH(GpuReal a, GpuReal b, char maxlog_type);

// PAM constellation
__const__ char PAM_2 [] = { -1, 1};
__const__ char PAM_4 [] = { -3, -1, 3, 1};
__const__ char PAM_6 [] = { -7, -5, -1, -3, 7, 5, 1, 3};
__const__ char PAM_8 [] = { -15, -13, -9, -11, -1, -3, -7, -5, 15, 13, 9, 11, 1, 3, 7, 5};

static const char *int2pam_lut [] = {NULL, PAM_2, PAM_4, PAM_6, PAM_8};

// define consule colors
#define CLR_WARN "\x1b[34m"
#define CLR_PASS "\x1b[32m"
#define CLR_BOLD "\x1b[1m"
#define CLR_RESET "\x1b[0m"

using namespace std::complex_literals;