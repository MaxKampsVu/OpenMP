#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <float.h>

#include "compute.h"
#include "fail.h"
#include "input.h"
#include "annotation.h"

void do_compute(const struct parameters *p, struct results *r) {
    size_t N = p->N, M = p->M;
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
    
    // Add timing variables
    struct timespec ts_start, ts_end;
    double time;
    
    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (iter = 0; iter < p->maxiter; iter++) {
        maxdiff = 0.0;
        tmin = INFINITY;
        tmax = -INFINITY;
        tavg = 0.0;

        // Process all rows including boundaries
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                // For boundary rows, just copy the values (fixed boundary conditions)
    
                
                // Compute indices for neighbors with cyclic boundary in M dimension
                size_t left = (j == 0) ? M - 1 : j - 1;
                size_t right = (j == M - 1) ? 0 : j + 1;

                // Compute the weights based on conductivity
                double conductivity = p->conductivity[i * M + j];
                double weight_self = conductivity;
                double remaining_weight = 1.0 - weight_self;
                double weight_direct = remaining_weight * (sqrt(2) / (sqrt(2) + 1)) / 4;
                double weight_diag = remaining_weight * (1 / (sqrt(2) + 1)) / 4;
                double tnew = 0;
                
                // Compute new temperature 
                if (i == 0) { // For cells in the first row
                    tnew = weight_self * current[i * M + j] +
                              weight_direct * (current[i * M + left] + current[i * M + right] +
                                               halo_top[j] + current[(i + 1) * M + j]) +
                              weight_diag * (halo_top[left] + halo_top[right] +
                                             current[(i + 1) * M + left] + current[(i + 1) * M + right]);
                } else if (i == N - 1) { // For cells in the last row
                    tnew = weight_self * current[i * M + j] +
                              weight_direct * (current[i * M + left] + current[i * M + right] +
                                               current[(i - 1) * M + j] + halo_bottom[j]) +
                              weight_diag * (current[(i - 1) * M + left] + current[(i - 1) * M + right] +
                                             halo_bottom[left] + halo_bottom[right]);
                } else { // For cells in middle rows 
                    tnew = weight_self * current[i * M + j] +
                              weight_direct * (current[i * M + left] + current[i * M + right] +
                                               current[(i - 1) * M + j] + current[(i + 1) * M + j]) +
                              weight_diag * (current[(i - 1) * M + left] + current[(i - 1) * M + right] +
                                             current[(i + 1) * M + left] + current[(i + 1) * M + right]);
                }
                
                // Keep track of maxdiff for early termination 
                double diff = fabs(tnew - current[i * M + j]);
                maxdiff = fmax(maxdiff, diff);

                // Set the new value in next Matrix 
                next[i * M + j] = tnew;
            }
        }
        
        // Calculate statistics over the entire grid
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                tmin = fmin(tmin, next[i * M + j]);
                tmax = fmax(tmax, next[i * M + j]);
                tavg += next[i * M + j];
            }
        }
        
        tavg /= (N * M);

        // Swap next and current for next iteration 
        double *temp = current;
        current = next;
        next = temp;

        // Report statistics every period
        if (((iter + 1) % p->period == 0 || maxdiff < p->threshold) && p->printreports) {
            // Stop timing for report
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            time = (ts_end.tv_sec - ts_start.tv_sec) + 
                   (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e9;
            
            // Set up the results structure
            r->niter = iter + 1;
            r->maxdiff = maxdiff;
            r->tmin = tmin;
            r->tmax = tmax;
            r->tavg = tavg;
            r->time = time;
            
            // Report results
            report_results(p, r);
        }
        
        // Check for convergence
        if (maxdiff < p->threshold) {
            iter++; // Count the last iteration
            break;
        }
    }
    
    // Final timing if not already reported
    if (!(iter % p->period == 0 || maxdiff < p->threshold)) {
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        time = (ts_end.tv_sec - ts_start.tv_sec) + 
                (ts_end.tv_nsec - ts_start.tv_nsec) / 1.0e9;
        
        // Set up final results
        r->niter = iter;
        r->maxdiff = maxdiff;
        r->tmin = tmin;
        r->tmax = tmax;
        r->tavg = tavg;
        r->time = time;
        
        // Report final results
        report_results(p, r);
    }

    free(current);
    free(next);
    free(halo_bottom);
    free(halo_top);
}
