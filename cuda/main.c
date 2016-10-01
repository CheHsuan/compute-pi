#include <time.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "montecarlopi.h"

void calculate_error_rate()
{
    double pi = 0.0;
    int i;
    for(i = 1; i <= 100; i++){
        pi = compute_pi_montecarlo_gpu(i*1000000);
        double diff = pi - M_PI > 0 ? pi - M_PI : M_PI - pi;
        double error = diff / M_PI;
        printf("%u,%lf\n", i, error);
    }
}

int main(int argc, char *argv[])
{
    __attribute__((unused)) int N = 400000000;
    __attribute__((unused)) double pi = 0.0;

#if defined(MONTECARLO_GPU)
    pi = compute_pi_montecarlo_gpu(N);
    printf("N = %d , pi = %lf\n", N, pi);
#endif

#if defined(MONTECARLO_CPU)
    pi = compute_pi_montecarlo_cpu(N);
    printf("N = %d , pi = %lf\n", N, pi);
#endif

#if defined(BENCHMARK)
#define CLOCK_ID CLOCK_MONOTONIC_RAW
#define ONE_SEC 1000000000.0

    struct timespec start = {0, 0};
    struct timespec end = {0, 0};

    if (argc < 2) return -1;

    N = atoi(argv[1]);

    // Monte Carlo with cpu
    clock_gettime(CLOCK_ID, &start);
    compute_pi_montecarlo_cpu(N);
    clock_gettime(CLOCK_ID, &end);
    printf("%lf,", (double) (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec)/ONE_SEC);

    // Monte Carlo with gpu
    clock_gettime(CLOCK_ID, &start);
    compute_pi_montecarlo_gpu(N);
    clock_gettime(CLOCK_ID, &end);
    printf("%lf\n", (double) (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec)/ONE_SEC);
#endif

#if defined(ERRORRATE)
    calculate_error_rate();
#endif

    return 0;
}
