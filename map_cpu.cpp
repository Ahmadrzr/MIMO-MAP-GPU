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

int bin2int(int* bits, int n) {
    int integer = 0;
    for (int i = 0; i < n; i++) {
        integer += bits[i] << (n - i - 1);
    }
    return integer;
}

void int2bin(int integer, int* bits, int n) {
    for (int i = 0; i < n; i++) {
        bits[i] = integer & 0x1;
        integer = integer >> 1;
    }
}

bool permutation_list_increment(int* plist, int base, int width) {
    //inteneded as an efficient way to go through permutations
    //use this in a while loop rather than doing a for loop of permutations and created a base_repr list
    for (int i = width - 1; i >= 0; i--) {
        if (plist[i] < (base - 1)) {
            plist[i] += 1;
            return true; //successful increment
        } else {
            plist[i] = 0;
        }
    }
    return false; //done, no more incrementing
}

// MAP-CPU multithread C++ code
void map_det_cpu(Mdc* y, Mdc* H, Md* La, Md* Le, int Mc, double sig2, enum Maxlog_type maxlog_type, int nthreads) {
    // y            received signal vector
    // H            MIMO channel matrix
    // La           input LLR vector
    // Le           output LLR vector
    // Mc           Constellation size
    // Sig2         Noise variance
    // maxlog_type  MAX-LOG or LOG-MAP selector
    // nthreads     Number of threads
    double E_M = 2.0 / 3.0 * (Mc - 1);
    int N_rx = y->dim[1];
    int N_tx = H->dim[2];

    int Mcb = (int) log2((float)Mc);
    int Mcbh = (int) Mcb / 2;
    int PMcbh = 1 << Mcbh; //permutations in half constellation bits
    int Tbits = Mcb * N_tx; //number of bits in a time slice of symbols in s

    //precompute permutation lists
    int d_half_perms [PMcbh];
    int bs_perms [PMcbh * Mcbh];
    int perms_binary_half_word [Mcbh];
    for (int p = 0; p < PMcbh; p++) {
        int2bin(p, perms_binary_half_word, Mcbh);
        int decimal = bin2int(perms_binary_half_word, Mcbh);
        d_half_perms[p] = (int)int2pam_lut[Mcbh][decimal];
        for (int j = 0; j < Mcbh; j++) {
            int ind = p * Mcbh + j;
            bs_perms[ind] = (perms_binary_half_word[j] == 1) ? 1 : -1;
        }
    }

    //precalculated H*s
    Mdc Hs_pre;
    Hs_pre.ndim = 3;
    Hs_pre.dim[0] = PMcbh;
    Hs_pre.dim[1] = 2 * N_tx;
    Hs_pre.dim[2] = N_rx;

    //precalculated prior
    Md Labs_pre;
    Labs_pre.ndim = 2;
    Labs_pre.dim[0] = PMcbh;
    Labs_pre.dim[1] = 2 * N_tx;

    if (nthreads == 0) {
        nthreads = omp_get_num_procs() / 2;
    }

    #pragma omp parallel \
    num_threads(nthreads) \
    firstprivate(Hs_pre,Labs_pre)
    #pragma omp for
    for (int t = 0; t < y->dim[0]; t++) {
        double* La_t = &La->data[t * Tbits];

        Hs_pre.data = new dcomplex[PMcbh * 2 * N_rx * N_tx];
        Labs_pre.data = new double[PMcbh * 2 * N_tx];

        double maxp [Tbits];
        double maxn [Tbits];
        int bs [Tbits]; //temporary bit sign
        int perm_list [2 * N_tx];
        int prev_perm_list [2 * N_tx];
        dcomplex yHs [N_rx];
        double dist, prior, maxtmp_full, maxtmp;

        //calculate using symobols/channel at time slice 't'

        for (int i = 0; i < Tbits; i++) {
            //init to very low value
            maxp[i] = -1000000;
            maxn[i] = -1000000;
        }
        //precompute all half constellation permutations
        for (int p = 0; p < PMcbh; p++) {
            for (int i = 0; i < N_tx; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < N_rx; k++) {
                        dcomplex tmp = H->data[t * H->dim[1] * H->dim[2] + k * H->dim[2] + i] * (dcomplex)d_half_perms[p];
                        if (j == 1) {
                            tmp = tmp * 1i; //apply if d_half_perms is real or imag
                        }
                        Hs_pre.data[p * Hs_pre.dim[1]*Hs_pre.dim[2] + (j + 2 * i)*Hs_pre.dim[2] + k] = tmp;
                    }
                    Labs_pre.data[p * Labs_pre.dim[1] + (j + 2 * i)] = 0;
                    for (int k = 0; k < Mcbh; k++) {
                        Labs_pre.data[p * Labs_pre.dim[1] + (j + 2 * i)] += La_t[i * Mcb + j * Mcbh + k] * bs_perms[p * Mcbh + k] / 2.0;
                    }
                }
            }
        }

        for (int i = 0; i < 2 * N_tx; i++) perm_list[i] = 0;
        for (int i = 0; i < N_rx; i++) {
            yHs[i] = y->data[t * N_rx + i];
        }
        prior = 0;
        for (int i = 0; i < 2 * N_tx; i++) {
            //accumulate pre-computed values
            prior += Labs_pre.data[perm_list[i] * Labs_pre.dim[1] + i];
            for (int j = 0; j < Mcbh; j++) {
                bs[i * Mcbh + j] = bs_perms[perm_list[i] * Mcbh + j];
            }
            for (int j = 0; j < N_rx; j++) {
                yHs[j] = yHs[j] - Hs_pre.data[perm_list[i] * Hs_pre.dim[1] * Hs_pre.dim[2] + i * Hs_pre.dim[2] + j];
            }
            prev_perm_list[i] = perm_list[i]; //update
        }

        int p = -1;
        do {
            p++;
            for (int i = 0; i < 2 * N_tx; i++) {
                //update changes to yHs,prior,bs
                if (perm_list[i] != prev_perm_list[i]) {
                    //remove old
                    prior = prior - Labs_pre.data[prev_perm_list[i] * Labs_pre.dim[1] + i];
                    //add new
                    prior = prior + Labs_pre.data[perm_list[i] * Labs_pre.dim[1] + i];
                    for (int j = 0; j < Mcbh; j++) {
                        //remove old
                        //add new
                        bs[i * Mcbh + j] = bs_perms[perm_list[i] * Mcbh + j];
                    }
                    for (int j = 0; j < N_rx; j++) {
                        //remove old
                        yHs[j] = yHs[j] + Hs_pre.data[prev_perm_list[i] * Hs_pre.dim[1] * Hs_pre.dim[2] + i * Hs_pre.dim[2] + j];
                        //add new
                        yHs[j] = yHs[j] - Hs_pre.data[perm_list[i] * Hs_pre.dim[1] * Hs_pre.dim[2] + i * Hs_pre.dim[2] + j];
                    }
                }
                prev_perm_list[i] = perm_list[i];
            }
            dist = 0;
            for (int i = 0; i < N_rx; i++) {
                dist += creal(yHs[i] * conj(yHs[i]));
            }
            maxtmp_full = -1 / sig2 * dist + prior;

            for (int k = 0; k < Tbits; k++) {
                maxtmp = maxtmp_full - La_t[k] * bs[k] / 2.0;
                if (bs[k] == 1) {
                    maxp[k] = maxlogH(maxp[k], maxtmp, (char)maxlog_type);
                } else {
                    maxn[k] = maxlogH(maxn[k], maxtmp, (char)maxlog_type);
                }
            }
        } while (permutation_list_increment(perm_list, PMcbh, 2 * N_tx));

        for (int k = 0; k < Tbits; k++) {
            Le->data[t * Tbits + k] = maxp[k] - maxn[k];
        }
        free(Hs_pre.data);
        free(Labs_pre.data);
    }
}