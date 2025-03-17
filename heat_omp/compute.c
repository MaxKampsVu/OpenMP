#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <float.h>
#include <omp.h>  

#include "compute.h"
#include "fail.h"
#include "input.h"
#include "annotation.h"

/*
// Rowblockwise distribution
void do_compute(const struct parameters *p, struct results *r) {
    size_t N = p->N, M = p->M;
    int num_threads = p->nthreads;
    omp_set_num_threads(num_threads); 

    double *current = malloc(N * M * sizeof(double));
    double *next = malloc(N * M * sizeof(double));

    // Halo rows for first and last row
    double *halo_top = malloc(M * sizeof(double));
    double *halo_bottom = malloc(M * sizeof(double));
    
    memcpy(current, p->tinit, N * M * sizeof(double));

    for (size_t j = 0; j < M; j++) {
        halo_top[j] =    current[j];
        halo_bottom[j] = current[(N - 1) * M + j];
    }

    size_t iter;
    double maxdiff, tmin, tmax, tavg;
    
    // Timing variables
    struct timespec ts_start, ts_end;
    double time;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (iter = 0; iter < p->maxiter; iter++) {
        maxdiff = 0.0;
        tmin = INFINITY;
        tmax = -INFINITY;
        tavg = 0.0;

        // OpenMP parallel region
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            
            int rows_per_thread = N / num_threads;
            int start_row = thread_id * rows_per_thread;
            int end_row = (thread_id == num_threads - 1) ? N : start_row + rows_per_thread;

            double local_tmin = INFINITY;
            double local_tmax = -INFINITY;
            double local_tavg = 0.0;

            // Each thread processes a subset of rows
            for (size_t i = start_row; i < end_row; i++) { 
                for (size_t j = 0; j < M; j++) {
                    size_t left = (j == 0) ? M - 1 : j - 1;
                    size_t right = (j == M - 1) ? 0 : j + 1;

                    double conductivity = p->conductivity[i * M + j];
                    double weight_self = conductivity;
                    double remaining_weight = 1.0 - weight_self;
                    double weight_direct = remaining_weight * (sqrt(2) / (sqrt(2) + 1)) / 4;
                    double weight_diag = remaining_weight * (1 / (sqrt(2) + 1)) / 4;
                    
                    double center = current[i * M + j];
                    double left_n = current[i * M + left];
                    double right_n = current[i * M + right];
                    double top_n = (i == 0) ? halo_top[j] : current[(i - 1) * M + j];
                    double bottom_n = (i == N - 1) ? halo_bottom[j] : current[(i + 1) * M + j]; 
                    double top_left_n = (i == 0) ? halo_top[left] : current[(i - 1) * M + left];
                    double top_right_n = (i == 0) ? halo_top[right] : current[(i - 1) * M + right]; ;
                    double bottom_left_n = (i == N - 1) ? halo_bottom[left] : current[(i + 1) * M + left]; 
                    double bottom_right_n = (i == N - 1) ? halo_bottom[right] : current[(i + 1) * M + right]; 
                    
                    double tnew = weight_self * center +
                               weight_direct * (left_n + right_n + top_n + bottom_n) +
                               weight_diag * (top_left_n + top_right_n +
                                              bottom_left_n + bottom_right_n);
                    
                    double diff = fabs(tnew - current[i * M + j]);

                    #pragma omp critical
                    {
                        maxdiff = fmax(maxdiff, diff);
                    }

                    next[i * M + j] = tnew;

                    // Update local statistics
                    local_tmin = fmin(local_tmin, tnew);
                    local_tmax = fmax(local_tmax, tnew);
                    local_tavg += tnew;
                }
            }

            #pragma omp critical
            {
                tmin = fmin(tmin, local_tmin);
                tmax = fmax(tmax, local_tmax);
                tavg += local_tavg;
            }


            #pragma omp barrier
        } 

        double *temp = current;
        current = next;
        next = temp;

        if (((iter + 1) % p->period == 0 || maxdiff < p->threshold) && p->printreports) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            time = (ts_end.tv_sec - ts_start.tv_sec) + 
                   (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e9;
            
            r->niter = iter + 1;
            r->maxdiff = maxdiff;
            r->tmin = tmin;
            r->tmax = tmax;
            r->tavg = tavg;
            r->time = time;

            report_results(p, r);
        }

        if (maxdiff < p->threshold) {
            iter++; 
            break;
        }
    }

    // Final report
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    time = (ts_end.tv_sec - ts_start.tv_sec) + 
           (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e9;

    r->niter = iter;
    r->maxdiff = maxdiff;
    r->tmin = tmin;
    r->tmax = tmax;
    r->tavg = tavg;
    r->time = time;

    report_results(p, r);

    // Free allocated memory
    free(current);
    free(next);
    free(halo_bottom);
    free(halo_top);
}
*/

// Rowise distribution

void do_compute(const struct parameters *p, struct results *r) {
    size_t N = p->N, M = p->M;
    int num_threads = p->nthreads;
    omp_set_num_threads(num_threads); 

    double *current = malloc(N * M * sizeof(double));
    double *next = malloc(N * M * sizeof(double));

    // Halo rows for first and last row
    double *halo_top = malloc(M * sizeof(double));
    double *halo_bottom = malloc(M * sizeof(double));
    
    memcpy(current, p->tinit, N * M * sizeof(double));

    for (size_t j = 0; j < M; j++) {
        halo_top[j] =    current[j];
        halo_bottom[j] = current[(N - 1) * M + j];
    }

    size_t iter;
    double maxdiff, tmin, tmax, tavg;
    
    // Timing variables
    struct timespec ts_start, ts_end;
    double time;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (iter = 0; iter < p->maxiter; iter++) {
        maxdiff = 0.0;
        tmin = INFINITY;
        tmax = -INFINITY;
        tavg = 0.0;

        // OpenMP parallel region
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

            double local_tmin = INFINITY;
            double local_tmax = -INFINITY;
            double local_tavg = 0.0;

            for (size_t i = 0; i < N; i++) { 
                if(i % thread_id != 0) { // Rows are computed in a cyclic fashion
                    continue;
                }
                for (size_t j = 0; j < M; j++) {
                    size_t left = (j == 0) ? M - 1 : j - 1;
                    size_t right = (j == M - 1) ? 0 : j + 1;

                    double conductivity = p->conductivity[i * M + j];
                    double weight_self = conductivity;
                    double remaining_weight = 1.0 - weight_self;
                    double weight_direct = remaining_weight * (sqrt(2) / (sqrt(2) + 1)) / 4;
                    double weight_diag = remaining_weight * (1 / (sqrt(2) + 1)) / 4;
                    
                    start_roi();
                    double center = current[i * M + j];
                    double left_n = current[i * M + left];
                    double right_n = current[i * M + right];
                    double top_n = (i == 0) ? halo_top[j] : current[(i - 1) * M + j];
                    double bottom_n = (i == N - 1) ? halo_bottom[j] : current[(i + 1) * M + j]; 
                    double top_left_n = (i == 0) ? halo_top[left] : current[(i - 1) * M + left];
                    double top_right_n = (i == 0) ? halo_top[right] : current[(i - 1) * M + right]; ;
                    double bottom_left_n = (i == N - 1) ? halo_bottom[left] : current[(i + 1) * M + left]; 
                    double bottom_right_n = (i == N - 1) ? halo_bottom[right] : current[(i + 1) * M + right];
                    end_roi(); 
                    
                    double tnew = weight_self * center +
                               weight_direct * (left_n + right_n + top_n + bottom_n) +
                               weight_diag * (top_left_n + top_right_n +
                                              bottom_left_n + bottom_right_n);
                    
                    double diff = fabs(tnew - current[i * M + j]);
                    
                    #pragma omp critical
                    {
                        start_roi();
                        maxdiff = fmax(maxdiff, diff);
                        end_roi();
                    }
                    

                    next[i * M + j] = tnew;

                    // Update local statistics
                    local_tmin = fmin(local_tmin, tnew);
                    local_tmax = fmax(local_tmax, tnew);
                    local_tavg += tnew;
                }
            }

            #pragma omp critical
            {
                start_roi();
                tmin = fmin(tmin, local_tmin);
                tmax = fmax(tmax, local_tmax);
                tavg += local_tavg;
                end_roi();
            }


            #pragma omp barrier
        } 

        double *temp = current;
        current = next;
        next = temp;

        if (((iter + 1) % p->period == 0 || maxdiff < p->threshold) && p->printreports) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            time = (ts_end.tv_sec - ts_start.tv_sec) + 
                   (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e9;
            
            r->niter = iter + 1;
            r->maxdiff = maxdiff;
            r->tmin = tmin;
            r->tmax = tmax;
            r->tavg = tavg;
            r->time = time;

            report_results(p, r);
        }

        if (maxdiff < p->threshold) {
            iter++; 
            break;
        }
    }

    // Final report
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    time = (ts_end.tv_sec - ts_start.tv_sec) + 
           (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e9;

    r->niter = iter;
    r->maxdiff = maxdiff;
    r->tmin = tmin;
    r->tmax = tmax;
    r->tavg = tavg;
    r->time = time;

    report_results(p, r);

    // Free allocated memory
    free(current);
    free(next);
    free(halo_bottom);
    free(halo_top);
}