#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>
#define ITERATIONSPERTHREAD 4000

__global__ void monte_carlo_kernel( curandState* state, unsigned int seed, int *numbers)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    float x, y;

    curand_init(seed, index, 0, &state[index]);

    for(int i = 0; i < ITERATIONSPERTHREAD; i++) {
        x = curand_uniform (&state[index]);
        y = curand_uniform (&state[index]);
        sum += (x*x + y*y <= 1.0f);
    }
    numbers[index] = sum;
}

double compute_pi_montecarlo_gpu(size_t N){
    int threadsPerBlock = 500;
    int blocksPerGrid = (N / threadsPerBlock) / ITERATIONSPERTHREAD;
    curandState *devStates;
    int *dev_nums;
    int *host_nums = (int *)malloc(sizeof(int) * threadsPerBlock * blocksPerGrid);
    if(host_nums == NULL)
        return 0;

    // malloc memory in gpu
    cudaMalloc((void **)&devStates, sizeof(curandState) * threadsPerBlock * blocksPerGrid);
    cudaMalloc((void **)&dev_nums, sizeof(int) * threadsPerBlock * blocksPerGrid);
    monte_carlo_kernel <<< blocksPerGrid, threadsPerBlock>>> (devStates, time(NULL), dev_nums);
    // copy data from device to host
    cudaMemcpy(host_nums, dev_nums, sizeof(int) * threadsPerBlock * blocksPerGrid, cudaMemcpyDeviceToHost);

    int total_in_quadcircle = 0;
    for(int i = 0; i < threadsPerBlock * blocksPerGrid; i++){
        total_in_quadcircle += host_nums[i];
    }
    double pi = 4 * ((double)total_in_quadcircle / N);

    cudaFree(devStates);
    cudaFree(dev_nums);

    return pi;
}

double compute_pi_montecarlo_cpu(size_t N)
{
    double pi = 0.0;
    size_t sum = 0;
    srand(time(NULL));
    for(size_t i = 0; i < N; i++)
    {
        double x = (double) rand() / RAND_MAX;
        double y = (double) rand() / RAND_MAX;
        if((x * x + y * y) < 1) {
            sum++; 
        }
    }
    pi = 4 * ((double)sum / N);
    return pi;
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

    return 0;
}
