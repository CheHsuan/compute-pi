#include <curand.h>
#include <curand_kernel.h>
extern "C" {
#include "montecarlopi.h"
}

#ifndef ITERATIONSPERTHREAD
#define ITERATIONSPERTHREAD 4000
#endif

extern "C"
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

double compute_pi_montecarlo_gpu(size_t N)
{
    int threadsPerBlock = 1000;
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
