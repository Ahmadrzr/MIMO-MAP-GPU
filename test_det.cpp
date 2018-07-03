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

double rnd()
{
    return (2 * (rand() % 100) - 100.0) / 9.87654321;
}

// Runs MAP-CPU and MAP-GPU methods and compares the output LLRs and the processing time
int main(int argc, char* argv[]) {    
    int nblocks  = 20;
    int Ntx = 4;
    int Nrx = 5;
    int Mc  = 64;
    int nthreads = 0;

    if (argc == 1) {
        ;  // pass
    } else if (argc == 5) {
        nblocks  = atoi(argv[1]);
        Ntx = atoi(argv[2]);
        Nrx = atoi(argv[3]);
        Mc  = atoi(argv[4]);
        nthreads = 0;
    } else if (argc == 6) {
        nblocks  = atoi(argv[1]);
        Ntx = atoi(argv[2]);
        Nrx = atoi(argv[3]);
        Mc  = atoi(argv[4]);
        nthreads = atoi(argv[5]);
    } else {
        for (int i = 0; i < argc; i++) {
            printf("argv[%d] = %s = %d\n", i, argv[i], atoi(argv[i]));
        }
        printf("ERROR: don't know how to handle this argument list.\n");
        printf("    Defaults:  nblocks=%d Ntx=%d Nrx=%d Mc=%d nthreads=%d\n", nblocks, Ntx, Nrx, Mc, nthreads);
        printf("    Possible arguments are:\n");
        printf("        <none>\n");
        printf("        nblocks Ntx Nrx Mc\n");
        printf("        nblocks Ntx Nrx Mc nthreads\n");
        exit(1);
    }

    int Mcb = (int)log2((double)Mc);

    srand(clock());

    Mdc y;
    y.ndim = 2;
    y.dim[0] = nblocks;
    y.dim[1] = Nrx;
    y.data = new dcomplex[y.dim[0]*y.dim[1]];
    for (int ix = 0; ix < y.dim[0]*y.dim[1]; ix++)
        y.data[ix] = rnd() + rnd() * 1i;

    Mdc H;
    H.ndim = 3;
    H.dim[0] = nblocks;
    H.dim[1] = Nrx;
    H.dim[2] = Ntx;
    H.data = new dcomplex[H.dim[0]*H.dim[1]*H.dim[2]];
    for (int ix = 0; ix < H.dim[0]*H.dim[1]*H.dim[2]; ix++)
        H.data[ix] = rnd() + rnd() * 1i;

    Md La;
    La.ndim = 1;
    La.dim[0] = nblocks * Ntx * Mcb;
    La.data = new double[La.dim[0]];
    for (int ix = 0; ix < La.dim[0]; ix++)
        La.data[ix]  = rnd();

    Md LeG;
    LeG.ndim = 1;
    LeG.dim[0] = nblocks * Ntx * Mcb;
    LeG.data = new double[LeG.dim[0]];

    Md LeC;
    LeC.ndim = 1;
    LeC.dim[0] = nblocks * Ntx * Mcb;
    LeC.data = new double[LeC.dim[0]];

    printf("%s=============================================================%s\n", CLR_WARN, CLR_RESET);
    InitDevice();
    printf("%s=============================================================%s\n", CLR_WARN, CLR_RESET);
    printf("Configuration\n");
    printf("blocks = %d\n Ntx   = %d\n Nrx   = %d\n  Mc   = %d \n", nblocks, Ntx, Nrx, Mc);
    printf("%s=============================================================%s\n", CLR_WARN, CLR_RESET);
    struct timespec start, end;

    printf("GPU MAP DET started ..."); fflush(NULL);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    map_det_gpu(&y, &H, &La, &LeG, Mc, 1.0, maxlog_logmap, nthreads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double GPU = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf(" Done\n\ttime = %4.0f msec\n", GPU);

    printf("CPU MAP DET started ..."); fflush(NULL);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    map_det_cpu(&y, &H, &La, &LeC, Mc, 1.0, maxlog_logmap, nthreads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double CPU = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf(" Done\n\ttime = %4.0f msec\n", CPU);

    printf("Speed up = %s%s%.1f%s   \n", CLR_PASS, CLR_BOLD, CPU / GPU, CLR_RESET);
    printf("%s=============================================================%s\n", CLR_WARN, CLR_RESET);

    double Error  = 0;
    for (int ix = 0; ix < nblocks * Ntx * Mcb; ix++)
    {
        Error += fabs(LeC.data[ix] - LeG.data[ix]);
    }

    printf("Total number of bits  = %d\n", nblocks * Ntx * Mcb);
    printf("Mean absolute error   = %1.5f\n", Error / (nblocks * Ntx * Mcb));
    printf("%s=============================================================%s\n", CLR_WARN, CLR_RESET);
    return 0;
}
