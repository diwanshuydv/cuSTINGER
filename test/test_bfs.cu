#include <cuda_runtime.h>
#include <iostream>
#include "cuStinger.hpp"
#include "../src/main.cu"

void testBFS() {
    // Simple graph with 5 vertices
    int numVertices = 5;
    int numEdges = 6;
    
    // CSR format
    int h_offsets[] = {0, 2, 3, 4, 6, 6};
    int h_edges[] = {1, 2, 3, 4, 0, 1};
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Run BFS from vertex 0
    runBFS(numVertices, numEdges, h_offsets, h_edges, 0, prop);
    
    // Expected output:
    // Vertex 0: 0
    // Vertex 1: 1
    // Vertex 2: 1
    // Vertex 3: 2
    // Vertex 4: 2
}

int main() {
    testBFS();
    return 0;
}
