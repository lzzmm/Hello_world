// Compile: nvcc Hello_world.cu -o Hello_world
// Run:     ./Hello_world

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void hello_world(int *global_m, int threadPerBlock) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    global_m[bid * threadPerBlock + tid] = 1;
}

int main() {

    // Test if CUDA available
    int deviceCount = 0, blockCount = 0, maxThreadsPerBlock = 0;

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There is no device suppporting CUDA\n");
    }
    int dev = 0;
    for (; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA.\n");
            else
                printf("there are %d devices supporting CUDA.\n", deviceCount);
        }
        printf("\nDevice %d:\"%s\"\n", dev, deviceProp.name);
        // printf("Major revision number: %d\n", deviceProp.major);
        // printf("Minor revision number: %d\n", deviceProp.minor);
        printf("Compute capability(version): %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total amount of global memory: %lu bytes (%.3f gigabytes)\n", deviceProp.totalGlobalMem, ((float)deviceProp.totalGlobalMem) * 1e-9);

#if CUDART_VERSION >= 2000
        printf("Number of multiprocessors(SM): %d\n", deviceProp.multiProcessorCount);
        printf("Number of cores: %d\n", 8 * deviceProp.multiProcessorCount);
#endif
        printf("Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
        printf("Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("Warp size(in threads): %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum sizes of each dimension of a block: %d x %d x %d \n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[1]);
        printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);
        printf("Texture alignment: %lu bytes\n", deviceProp.textureAlignment);
        printf("Clock rate: %.0f MHz\n", deviceProp.clockRate * 1e-3);
#if CUDART_VERSION >= 2000
        printf("Concurrent copy and execution: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
        blockCount += deviceProp.multiProcessorCount;
        maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    }

    int *host_m, *global_m, sum = 0;

    // allocate host memory
    host_m = (int *)malloc(sizeof(int) * blockCount * maxThreadsPerBlock);

    // Allocate device memory
    cudaMalloc((void **)&global_m, sizeof(int) * blockCount * maxThreadsPerBlock);

    // Execute kernels
    hello_world<<<blockCount, maxThreadsPerBlock>>>(global_m, maxThreadsPerBlock);

    // Transfer output from device memory to host
    cudaMemcpy(host_m, global_m, sizeof(int) * blockCount * maxThreadsPerBlock, cudaMemcpyDeviceToHost);

    for (int i = 0; i < blockCount * maxThreadsPerBlock; i++) {
        sum += host_m[i];
    }

    printf("\nHello_world!\n  ——  from %d CUDA threads in %d GPU(s)\n", sum, deviceCount);

    cudaFree(global_m);
    free(host_m);

    return 0;
}
