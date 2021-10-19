__global__ void initArray(float* arr, int N)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < N)
    {
        arr[id] = id;
    }
}

__global__ void addVectors(float* first, float* second, float* result, int N)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < N)
    {
        result[id] = first[id] + second[id];
    }
}

int main()
{
    int N = 1 << 20;

    float *a, *b, *c;

    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));

    int numThreads = 1<<9;
    int numBlocks = N/numThreads;
    initArray<<<numBlocks, numThreads>>> (a, N);
    initArray<<<numBlocks, numThreads>>> (b, N);

    cudaDeviceSynchronize();

    addVectors<<<numBlocks, numThreads>>> (a, b, c, N);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
