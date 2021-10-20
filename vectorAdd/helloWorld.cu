#include <stdio.h>

__global__ void printThreads()
{
    printf("threadId.x : %d, threadId.y : %d, threadId.z : %d \n",
        threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    int nx, ny;
    nx = 16;
    ny = 16;

    dim3 block(8, 8);
    dim3 grid(nx/block.x, ny/block.y);

    printThreads<<<grid, block>>> ();
    cudaDeviceSynchronize();

    return 0;
}