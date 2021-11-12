#include <stdio.h>
#include <limits>
#include <algorithm>
#include <iterator>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BLK_SIZE 512

__global__ void bfs_kerenel(int *nodes, int *edgesArr, bool *frontierArr, bool *visitedArr, int *costArr, bool* frontierExists, int numNodes, int numEdges)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < numNodes)
    {
        if(frontierArr[idx])
        {
            frontierArr[idx] = false;
            visitedArr[idx] = true;
            int iEnd = (idx + 1) < numNodes ? nodes[idx+1] : numEdges;
            for(int i = nodes[idx]; i < iEnd; i++)
            {
                if (!visitedArr[edgesArr[i]])
                {
                    costArr[edgesArr[i]] = costArr[idx] + 1;
                    frontierArr[edgesArr[i]] = true;
                    if(!(*frontierExists))
                    {
                        atomicCAS((int *)frontierExists, (int) false, (int) true);
                    }
                }
            }
        }
    }
}

int main()
{
    const int numNodes = 6;
    const int numEdges = 16;
    int h_nodes[numNodes] = {0, 2, 5, 7, 10, 14};
    int h_edgesArr[numEdges] = {            // Node  Start Position
                                1, 2,       // 0        0
                                0, 3, 4,    // 1        2
                                0, 4,       // 2        5
                                1, 4, 5,    // 3        7
                                1, 2, 3, 5, // 4        10
                                3, 4        // 5        14
                            };
    bool h_frontierArr[numNodes] {false};
    bool h_visitedArr[numNodes] {false};
    int h_costArr[numNodes] {std::numeric_limits<int>::max()};
    bool h_frontierExists{true};

    h_frontierArr[0] = true;
    h_costArr[0] = 0;

    int NODES_SIZE = numNodes * sizeof(int);
    int BOOLS_SIZE = numNodes * sizeof(bool);
    int EDGES_SIZE = numEdges * sizeof(int);

    int *d_nodes;
    int *d_edgesArr;
    bool *d_frontierArr;
    bool *d_visitedArr;
    int *d_costArr;
    bool *d_frontierExists;

    cudaMalloc(&d_nodes, NODES_SIZE);
    cudaMalloc(&d_edgesArr, EDGES_SIZE);
    cudaMalloc(&d_frontierArr, BOOLS_SIZE);
    cudaMalloc(&d_visitedArr, BOOLS_SIZE);
    cudaMalloc(&d_costArr, NODES_SIZE);
    cudaMalloc(&d_frontierExists, sizeof(bool));

    cudaMemcpy(d_nodes, h_nodes, NODES_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgesArr, h_edgesArr, EDGES_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierArr, h_frontierArr, BOOLS_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visitedArr, h_visitedArr, BOOLS_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_costArr, h_costArr, EDGES_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice);

    while(h_frontierExists)
    {
        h_frontierExists = false;
        cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice);
        bfs_kerenel<<<1, BLK_SIZE>>>(d_nodes, d_edgesArr, d_frontierArr, d_visitedArr, d_costArr, d_frontierExists, numNodes, numEdges);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(h_costArr, d_costArr, NODES_SIZE, cudaMemcpyDeviceToHost);

    for(int i = 0; i < numNodes; i++)
    {
        printf("Node: %d, cost: %d\n", i, h_costArr[i]);
    }

    cudaFree(d_nodes);
    cudaFree(d_edgesArr);
    cudaFree(d_frontierArr);
    cudaFree(d_visitedArr);
    cudaFree(d_costArr);
    cudaFree(d_frontierExists);
}