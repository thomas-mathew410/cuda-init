#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define ROW_SIZE 1024
#define COLUMN_SIZE 1024
#define BLK_SIZE 512
#define MIN 1
#define MAX 100

__global__ void setup_kernel(curandState *state, unsigned long long seed){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx < COLUMN_SIZE)
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void init_matrix(int *matrix, int dim, int min, int max, curandState *state)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    int row = idx / dim;
    float myrandf = curand_uniform(&state[row]);
    myrandf *= (max - min+0.999999);
    myrandf += min;
    int myrand = (int)truncf(myrandf);

    matrix[idx] = myrand;
}

int main()
{
    curandState *d_state;
    cudaMalloc(&d_state, COLUMN_SIZE * sizeof(curandState));
    setup_kernel<<<2, BLK_SIZE>>> (d_state, clock());

    int *d_matrix;
    cudaMalloc(&d_matrix, (COLUMN_SIZE * ROW_SIZE) * sizeof(int));

    init_matrix<<<(BLK_SIZE * 4), BLK_SIZE>>> (d_matrix, ROW_SIZE, MIN, MAX, d_state);
}