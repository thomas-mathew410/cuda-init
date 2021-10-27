#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

#define ROW_SIZE 1024
#define COLUMN_SIZE 1024
#define BLK_SIZE 512
#define MIN 1
#define MAX 15

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

__global__ void matrix_transpose(int *matrix, int dim)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    int row = idx / dim;
    int col = idx % dim;

    int temp = 0;
    if(row < col)
    {
        temp = matrix[idx];
        matrix[idx] = matrix[(col * dim) + row];
        matrix[(col * dim) + row] = temp;
    }

}

__global__ void matrix_multiply(int **matrices, int *res, int dim)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    int row = idx / dim;
    int col = idx % dim;

    int cell_value = 0;
    const int row_index = (row * dim);
    const int col_index = (col * dim);
    for(int i = 0; i < dim; i++)
    {
        cell_value += matrices[0][row_index + i] * matrices[1][col_index + i];
    }

    res[(row * dim) + col] = cell_value;
}

int main()
{
    curandState *d_state;
    cudaMalloc(&d_state, COLUMN_SIZE * sizeof(curandState));
    setup_kernel<<<2, BLK_SIZE>>> (d_state, clock());
    int matrix_size = (COLUMN_SIZE * ROW_SIZE);

    int *d_matrix[2], *h_matrix[2];

    for(int i=0; i < 2; i++)
    {
        cudaMalloc(&d_matrix[i], matrix_size * sizeof(int));
        h_matrix[i] = (int *)malloc(matrix_size * sizeof(int));

        init_matrix<<<(BLK_SIZE * 4), BLK_SIZE>>> (d_matrix[i], ROW_SIZE, MIN, MAX, d_state);
        cudaDeviceSynchronize();
        cudaMemcpy(h_matrix[i], d_matrix[i], matrix_size, cudaMemcpyDeviceToHost);
    }

    matrix_transpose<<<(BLK_SIZE * 4), BLK_SIZE>>> (d_matrix[1], ROW_SIZE);
    cudaDeviceSynchronize();

    int *d_matrix_res, *h_matrix_res;
    cudaMalloc(&d_matrix_res, matrix_size * sizeof(int));
    h_matrix_res = (int *)malloc(matrix_size * sizeof(int));

    matrix_multiply<<<(BLK_SIZE * 4), BLK_SIZE>>> (d_matrix, d_matrix_res, ROW_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matrix_res, d_matrix_res, matrix_size, cudaMemcpyDeviceToHost);

    // matrix_transpose<<<(BLK_SIZE * 4), BLK_SIZE>>> (d_matrix[1], ROW_SIZE);
    cudaDeviceSynchronize();

    int *h_matrix_res_serial;
    h_matrix_res_serial = (int *)malloc(matrix_size * sizeof(int));

    // Get transposed matrix
    cudaMemcpy(h_matrix[1], d_matrix[1], matrix_size, cudaMemcpyDeviceToHost);

    for(int k = 0; k < COLUMN_SIZE; k++)
    {
        for(int i = 0; i < COLUMN_SIZE; i++)
        {
            int cellVal = 0;
            for(int j = 0; j < COLUMN_SIZE; j++)
            {
                cellVal += (h_matrix[0][k * COLUMN_SIZE + j] + h_matrix[1][i * COLUMN_SIZE + j]);
            }
            h_matrix_res_serial[k * COLUMN_SIZE + i] = cellVal;
        }
    }

    for(int i = 0; i < matrix_size; i++)
    {
        if(h_matrix_res[i] != h_matrix_res_serial[i])
        {
            printf("Unequal values at row: %d column: %d", i/COLUMN_SIZE, i%COLUMN_SIZE);
            break;
        }
    }

    for(int i=0; i < 2; i++)
    {
        cudaFree(d_matrix[i]);
        free(h_matrix[i]);
    }
    cudaFree(d_matrix_res);
    free(h_matrix_res);
    free(h_matrix_res_serial);
}