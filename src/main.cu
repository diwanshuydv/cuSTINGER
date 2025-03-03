#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include "utils.hpp"
#include "update.hpp"
#include "memoryManager.hpp"
#include "cuStinger.hpp"
#include <chrono>
#include <iomanip>

using namespace std;

// CUDA error checking macro
#define CHECK_CUDA(call) {                                            \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}
struct Update {
    bool is_addition; // true if edge is added, false if removed
    vertexId_t u, v;  // edge between vertices u and v
};

/**
 * CUDA kernel for BFS traversal
 * @param numVertices Number of vertices in the graph
 * @param d_offsets CSR offsets array (device)
 * @param d_edges CSR edges array (device)
 * @param d_distances Distance array (device)
 * @param d_frontier Current frontier vertices (device)
 * @param frontierSize Size of current frontier
 * @param d_next_frontier Next frontier vertices (device)
 * @param d_next_frontier_size Size of next frontier (device)
 * @param currentLevel Current BFS level
 */



//    __global__ void bfs_kernel(int numVertices, 
//     const int *d_offsets, 
//     const int *d_edges, 
//     int *d_distances, 
//     const int *d_frontier, 
//     int frontierSize, 
//     int *d_next_frontier, 
//     int *d_next_frontier_size, 
//     int currentLevel)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= frontierSize) return;
//     // printf("idx = %d\n", idx); 
//     // printf("frontierSize = %d\n", frontierSize);    
//     int u = d_frontier[idx];
//     int start = d_offsets[u];
//     int end = d_offsets[u + 1];

//     for (int e = start; e < end; e++) {
//         int v = d_edges[e];
//         // Atomic operation to set the distance if it has not been visited
//         if (atomicCAS(&d_distances[v], -1, currentLevel) == -1) {
//             // Add vertex to next frontier
//             int pos = atomicAdd(d_next_frontier_size, 1);
//             d_next_frontier[pos] = v;
//         }
//     }
// }

// /**
//  * Runs BFS on the given graph starting from source vertex
//  * @param numVertices Number of vertices in the graph
//  * @param numEdges Number of edges in the graph
//  * @param h_offsets CSR offsets array (host)
//  * @param h_edges CSR edges array (host)
//  * @param source Source vertex for BFS
//  * @param prop CUDA device properties for optimal kernel configuration
//  */
// void runBFS(int numVertices, int numEdges, const int *h_offsets, const int *h_edges, int source, const cudaDeviceProp& prop)
// {
//     int *d_offsets, *d_edges, *d_distances;
//     CHECK_CUDA(cudaMalloc((void**)&d_offsets, (numVertices + 1) * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void**)&d_edges, numEdges * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void**)&d_distances, numVertices * sizeof(int)));
    
//     CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_edges, h_edges, numEdges * sizeof(int), cudaMemcpyHostToDevice));
    
//     int *h_distances = (int*) malloc(numVertices * sizeof(int));
//     for (int i = 0; i < numVertices; i++) {
//         h_distances[i] = -1;
//     }
//     h_distances[source] = 0;
//     CHECK_CUDA(cudaMemcpy(d_distances, h_distances, numVertices * sizeof(int), cudaMemcpyHostToDevice));
    
//     int *d_frontier, *d_next_frontier, *d_next_frontier_size;
//     CHECK_CUDA(cudaMalloc((void**)&d_frontier, numVertices * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void**)&d_next_frontier, numVertices * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void**)&d_next_frontier_size, sizeof(int)));
    
//     int frontierSize = 1;
//     CHECK_CUDA(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    
//     int currentLevel = 0;
//     while (frontierSize > 0) {
//         // Reset the next frontier size to zero
//         CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(int)));
        
//         // Use optimal block size based on GPU properties
//         int blockSize = prop.maxThreadsPerBlock / 2; // Start with half max threads
//         blockSize = min(blockSize, 256); // Cap at 256 for better occupancy
//         blockSize = max(blockSize, 32); // Ensure minimum of 32 threads
//         int gridSize = (frontierSize + blockSize - 1) / blockSize;
//          // Print the number of threads used for this kernel launch
//          cout << "Launching bfs_kernel with gridSize = " << gridSize 
//          << " and blockSize = " << blockSize 
//          << " (total threads = " << gridSize * blockSize << ")" << endl;
//         // Pass currentLevel+1 so neighbors get the proper distance value
//         bfs_kernel<<<gridSize, blockSize>>>(numVertices, d_offsets, d_edges, d_distances, 
//                                             d_frontier, frontierSize, d_next_frontier, 
//                                             d_next_frontier_size, currentLevel + 1);
//         CHECK_CUDA(cudaDeviceSynchronize());
        
//         int nextFrontierSize;
//         CHECK_CUDA(cudaMemcpy(&nextFrontierSize, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
        
//         if (nextFrontierSize == 0)
//             break;
        
//         // Update the frontier for the next level
//         CHECK_CUDA(cudaMemcpy(d_frontier, d_next_frontier, nextFrontierSize * sizeof(int), cudaMemcpyDeviceToDevice));
//         frontierSize = nextFrontierSize;
//         currentLevel++;
//     }
    
//     CHECK_CUDA(cudaMemcpy(h_distances, d_distances, numVertices * sizeof(int), cudaMemcpyDeviceToHost));
    
//     cout << "BFS distances from source vertex " << source << ":\n";
//     for (int i = 0; i < numVertices; i++) {
//         cout << "Vertex " << i << ": " << h_distances[i] << "\n";
//     }
    
//     cudaFree(d_offsets);
//     cudaFree(d_edges);
//     cudaFree(d_distances);
//     cudaFree(d_frontier);
//     cudaFree(d_next_frontier);
//     cudaFree(d_next_frontier_size);
//     free(h_distances);
// }

// //////////////////////////////////////
// // BFS Kernel
// //////////////////////////////////////
// __global__ void bfs_kernel(const cuStinger::cusVertexData* dVD,
//     const vertexId_t nv,
//     const int current_level,
//     const int* frontier,
//     const int frontier_size,
//     int* levels,
//     vertexId_t* next_frontier,
//     int* next_count) {
// // thread id
// int tid = blockIdx.x * blockDim.x + threadIdx.x;
// // to not allocate unnecessary amount of threads
// if (tid < frontier_size) {
// int v = frontier[tid];  // Get vertex ID from the current frontier
// // dVD->used = how many edges are used in the graph (dVD is cuSTINGER)
// int numNeighbors = dVD->used[v];
// // cuStinger::cusEdgeData* neighbors = dVD->adj[v];
// // forced typecasting reinterpret_cast
// vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);

// for (int i = 0; i < numNeighbors; i++) {
// // Correct by dereferencing the pointer to obtain a vertexId_t:

// vertexId_t nbr = nbrArray[i];

// // Update the level using atomicCAS. Use the address of levels[nbr]
// // atomic Compare And Set 
// // if both threads are updating the same thing, then only one gets permission/access 
// if (atomicCAS(&levels[nbr], -1, current_level + 1) == -1) {
// int pos = atomicAdd(next_count, 1);
// next_frontier[pos] = nbr;
// }
// }
// }
// }

//////////////////////////////////////
// BFS Kernel
//////////////////////////////////////
__global__ void bfs_kernel(const cuStinger::cusVertexData* dVD,
    const vertexId_t nv,
    const int current_level,
    const int* frontier,
    const int frontier_size,
    int* levels,
    vertexId_t* next_frontier,
    int* next_count) {
  // thread id
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];  // current vertex from frontier
    int numNeighbors = dVD->used[v];
    // reinterpret pointer to edge memory
    vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
    for (int i = 0; i < numNeighbors; i++) {
      vertexId_t nbr = nbrArray[i];
      // If not yet discovered then set level and add to next frontier.
      if (atomicCAS(&levels[nbr], -1, current_level + 1) == -1) {
        int pos = atomicAdd(next_count, 1);
        next_frontier[pos] = nbr;
      }
    }
  }
}

//////////////////////////////////////
// Host-Side BFS Implementation
//////////////////////////////////////
int* runBFS(cuStinger* graph, int source ,int* h_levels,int* d_frontier) {
    // nv = number of vertices
    int nv = graph->nv;

    // Allocate and initialize levels array on host and device
    // array of levels on CPU
    // int* h_levels = (int*)malloc(nv * sizeof(int));
    // for (int i = 0; i < nv; i++) {
    //     h_levels[i] = -1;  // -1 indicates undiscovered
    // }
    // h_levels[source] = 0;

    // array of levels on GPU
    int* d_levels;

    // if CUDA has error -> inform
    CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));

    // copy memory from host to device
    CHECK_CUDA(cudaMemcpy(d_levels, h_levels, nv * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate frontiers on device
    // whichever is the border
    // int* d_frontier;

    // 0 - source (d_frontier = &source)
    // 1 - vertices associated
    // ...

    // next border
    int* d_next_frontier;
    // CHECK_CUDA(cudaMalloc((void**)&d_frontier, nv * sizeof(vertexId_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_next_frontier, nv * sizeof(vertexId_t)));

    // Start frontier contains only the source vertex
    // how many elements in CPU queue
    int h_frontier_size = 1;
    CHECK_CUDA(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate device counter for the next frontier
    int* d_next_count;
    CHECK_CUDA(cudaMalloc((void**)&d_next_count, sizeof(int)));

    int current_level = 0;
    while (h_frontier_size > 0) {
        // Reset next frontier count
        CHECK_CUDA(cudaMemset(d_next_count, 0, sizeof(int)));

        // Launch BFS kernel
        int blockSize = 256;
        // which thread runs where 
        int gridSize = (h_frontier_size + blockSize - 1) / blockSize;
        bfs_kernel<<<gridSize, blockSize>>>(graph->dVD, nv, current_level, d_frontier,
                                            h_frontier_size, d_levels, d_next_frontier, d_next_count);
        // cout<<"BFS KERNEL LAUNCHED"<<endl;
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy next frontier size back to host
        int h_next_count;
        CHECK_CUDA(cudaMemcpy(&h_next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap frontiers
        int* temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;

        h_frontier_size = h_next_count;
        current_level++;
    }

    // Copy levels array back to host
    CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));

    // Print BFS levels
    for (int i = 0; i < nv; i++) {
        printf("Vertex %d: Level %d\n", i, h_levels[i]);
    }

    // Free allocated memory
    // free(h_levels);
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_frontier));
    CHECK_CUDA(cudaFree(d_next_frontier));
    CHECK_CUDA(cudaFree(d_next_count));
    return h_levels;
}


//////////////////////////////////////
// BFS Update Kernel for Edge Additions
//////////////////////////////////////
// For each new edge (u,v), if u is discovered and can provide a shorter
// path to v, update v's level and add it to the update frontier.
__global__ void bfs_update_add_kernel(const cuStinger::cusVertexData* dVD,  
    const int* add_edges,   // packed as: [u0, v0, u1, v1, ...]  
    const int num_edges,  
    int* levels,  
    vertexId_t* update_frontier,  
    int* update_count) {  
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid < num_edges) {  
  printf("tid: %d\n", tid); 

    int u = add_edges[2 * tid];  
    int v = add_edges[2 * tid + 1];  
    // Process only if u is discovered.
    printf("u: %d, v: %d\n", u, v);
    printf("fine");
    printf("levels[u]: %d\n", levels[u]);
    if (levels[u] != -1) {  
      int new_level = levels[u] + 1;  
      // Use atomicCAS to update levels[v] only if it is still undiscovered (-1)
      int old = atomicCAS(&levels[v], -1, new_level);  
      printf("old: %d\n", old);
      
      int old_2 = atomicMin(&levels[v], new_level);
      

      if (old == -1 || old_2>new_level) {  // Successful update: v was undiscovered  
        printf("update_count: %d\n", *(update_count));
        int pos = atomicAdd(update_count, 1);  
        printf("update_count: %d\n", *(update_count));
        printf("pos: %d\n", pos);   
        update_frontier[pos] = v;  
      }  
    }  
  }  
}

//////////////////////////////////////
// BFS Update Kernel for Edge Deletions
//////////////////////////////////////
// For each deleted edge (u,v), if v's current level comes from u then
// mark v for recomputation by invalidating its level.
__global__ void bfs_update_del_kernel(const cuStinger::cusVertexData* dVD,
    const int* del_edges,   // packed as: [u0, v0, u1, v1, ...]
    const int num_edges,
    int* levels,
    vertexId_t* update_frontier,
    int* update_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_edges) {
    int u = del_edges[2 * tid];
    int v = del_edges[2 * tid + 1];
    // Only if the deleted edge was on the BFS tree (v was reached from u)
    if (levels[u] != -1 && levels[v] == levels[u] + 1) {
      // Invalidate v; a more sophisticated implementation would try to find an alternate parent.
      levels[v] = -1;
      int pos = atomicAdd(update_count, 1);
      update_frontier[pos] = v;
    }
  }
}

//////////////////////////////////////
// Host-Side Streaming Update Handler
//////////////////////////////////////
// This function applies a batch of update edges (additions or deletions)
// and then “propagates” the changes via BFS until no further level changes occur.
void updateBFSUpdates(cuStinger* graph, int* h_levels, int* d_frontier,
                      int* h_update_edges, int num_updates, bool isAddition) {
  int nv = graph->nv;

  // Copy update edges to device memory.
  int* d_update_edges;
  CHECK_CUDA(cudaMalloc((void**)&d_update_edges, 2 * num_updates * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_update_edges, h_update_edges,
                        2 * num_updates * sizeof(int), cudaMemcpyHostToDevice));

  int* d_levels;
  CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_levels, h_levels,
                        nv * sizeof(int), cudaMemcpyHostToDevice));

  // Allocate temporary frontier and counter for propagating update changes.
  vertexId_t* d_update_frontier;
  CHECK_CUDA(cudaMalloc((void**)&d_update_frontier, nv * sizeof(vertexId_t)));
  int* d_update_count;
  CHECK_CUDA(cudaMalloc((void**)&d_update_count, sizeof(int)));

  int h_update_count = 0;
  int blockSize = 256;
  int gridSize = (num_updates + blockSize - 1) / blockSize;

  // Launch the appropriate update kernel.
  if (isAddition) {
    bfs_update_add_kernel<<<gridSize, blockSize>>>(graph->dVD, d_update_edges, num_updates,
                                                     d_levels, d_update_frontier, d_update_count);
  } else {
    bfs_update_del_kernel<<<gridSize, blockSize>>>(graph->dVD, d_update_edges, num_updates,
                                                     d_levels, d_update_frontier, d_update_count);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Get the count of vertices in the update frontier.
  CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));

  // Propagate the update changes until the frontier is empty.
  int current_level = 0;  // You might want to adjust this if levels are relative to a root.
  while (h_update_count > 0) {
    // Reset the counter for the next propagation step.
    CHECK_CUDA(cudaMemset(d_update_count, 0, sizeof(int)));
    gridSize = (h_update_count + blockSize - 1) / blockSize;
    // Use the original BFS kernel to propagate updated levels.
    bfs_kernel<<<gridSize, blockSize>>>(graph->dVD, nv, current_level, d_frontier,
                                          h_update_count, d_levels, d_update_frontier, d_update_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    // Swap frontiers.
    int* temp = d_frontier;
    d_frontier = (int*)d_update_frontier;
    d_update_frontier = (vertexId_t*)temp;
    // Get the new update frontier count.
    CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
    current_level++;
  }
   // Copy levels array back to host
   CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));
   // Print BFS levels
   for (int i = 0; i < nv; i++) {
    printf("Vertex %d: Level %d\n", i, h_levels[i]);
}

  // Free temporary device memory.
  CHECK_CUDA(cudaFree(d_update_edges));
  CHECK_CUDA(cudaFree(d_update_frontier));
  CHECK_CUDA(cudaFree(d_update_count));
}


// Printer utility function for cuStinger
void printcuStingerUtility(cuStinger custing, bool allInfo) {
    length_t used, allocated;
    used = custing.getNumberEdgesUsed();
    allocated = custing.getNumberEdgesAllocated();
    if (allInfo)
        cout << "," << used << "," << allocated << "," << (float)used / (float)allocated;  
    else
        cout << "," << (float)used / (float)allocated;
}

// Generate random edge updates
void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst) {
    for (int e = 0; e < numEdges; e++) {
        edgeSrc[e] = rand() % nv;
        edgeDst[e] = rand() % nv;
    }
}

// RMAT edge generation (helper functions)
typedef struct dxor128_env {
    unsigned x, y, z, w;
} dxor128_env_t;

double dxor128(dxor128_env_t * e) {
    unsigned t = e->x ^ (e->x << 11);
    e->x = e->y; e->y = e->z; e->z = e->w;
    e->w = (e->w ^ (e->w >> 19)) ^ (t ^ (t >> 8));
    return e->w * (1.0 / 4294967296.0);
}

void dxor128_init(dxor128_env_t * e) {
    e->x = 123456789;
    e->y = 362436069;
    e->z = 521288629;
    e->w = 88675123;
}

void dxor128_seed(dxor128_env_t * e, unsigned seed) {
    e->x = 123456789;
    e->y = 362436069;
    e->z = 521288629;
    e->w = seed;
}

void rmat_edge(int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env)
{
    int64_t i = 0, j = 0;
    int64_t bit = ((int64_t) 1) << (SCALE - 1);

    while (1) {
        const double r = ((double) rand() / (RAND_MAX));
        if (r > A) {
            if (r <= A + B)
                j |= bit;
            else if (r <= A + B + C)
                i |= bit;
            else {
                j |= bit;
                i |= bit;
            }
        }
        if (1 == bit)
            break;

        A *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
        B *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
        C *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
        D *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;

        {
            const double norm = 1.0 / (A + B + C + D);
            A *= norm; B *= norm; C *= norm;
        }
        D = 1.0 - (A + B + C);
        bit >>= 1;
    }
    *iout = i;
    *jout = j;
}

// Generate RMAT edge updates
void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,
                             double A, double B, double C, double D, dxor128_env_t * env)
{
    int64_t src, dst;
    int scale = (int)log2(double(nv));
    for (int32_t e = 0; e < numEdges; e++) {
        rmat_edge(&src, &dst, scale, A, B, C, D, env);
        edgeSrc[e] = src;
        edgeDst[e] = dst;
    }
}

// void testBFS() {
//     // Simple graph with 5 vertices
//     int numVertices = 5;
//     int numEdges = 6;
    
//     // CSR format
//     int h_offsets[] = {0, 2, 3, 4, 6, 6};
//     int h_edges[] = {1, 2, 3, 4, 0, 1};
    
//     // Get GPU properties
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
    
//     // Run BFS from vertex 1
//     runBFS(numVertices, numEdges, h_offsets, h_edges, 0, prop);
    
//     // Expected output:
//     // Vertex 0: 0
//     // Vertex 1: 1
//     // Vertex 2: 1
//     // Vertex 3: 2
//     // Vertex 4: 2
// }

int main(const int argc, char *argv[])
{  
    //  testBFS();
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne, *off;
    vertexId_t *adj;
    int isRmat = 0;

    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <graph_file> <graphName> [options]\n";
        return 1;
    }
 
    char* graphName = argv[2];
    srand(100);

    bool isDimacs, isSNAP, isMM;
    string filename(argv[1]);
    isDimacs = (filename.find(".graph") != string::npos);
    isSNAP   = (filename.find(".txt") != string::npos);
    isMM     = (filename.find(".mtx") != string::npos);
    isRmat   = (filename.find("kron") != string::npos);

    bool undirected = hasOption("--undirected", argc, argv);

    if (isDimacs) {
        readGraphDIMACS(argv[1], &off, &adj, &nv, &ne, isRmat);
    } else if (isSNAP) {
        readGraphSNAP(argv[1], &off, &adj, &nv, &ne, undirected);
    } else if (isMM) {
        readGraphMatrixMarket(argv[1], &off, &adj, &nv, &ne, undirected);
    } else { 
        cout << "Unknown graph type" << endl;
        return 1;
    }
 

    // Set up for cuStinger
    cudaEvent_t ce_start, ce_stop;
    cuStingerInitConfig cuInit;
    cuInit.initState   = eInitStateCSR;
    cuInit.maxNV       = nv + 1;
    cuInit.useVWeight  = false;
    cuInit.isSemantic  = false;
    cuInit.useEWeight  = false;
    
    cuInit.csrNV       = nv;
    cuInit.csrNE       = ne;
    cuInit.csrOff      = off;
    cuInit.csrAdj      = adj;
    cuInit.csrVW       = NULL;
    cuInit.csrEW       = NULL;
    cuStinger custing2(defaultInitAllocater, defaultUpdateAllocater);
    custing2.initializeCuStinger(cuInit);
    // Run BFS on the original CSR arrays
    cout << "Running BFS on the input graph ...\n";
    // cout<<"size of off "<<sizeof(off)<<" "<<"size of adj "<<sizeof(adj)<<endl;  


    // cout<<"off:";

    // for (length_t i = 0; i < nv+1; i++) {
    //     cout << off[i] << " ";
    // }
    // cout << endl;   
    // cout<<"adj:";   
    // for (int i=0;i<ne;i++) {
    //     cout << adj[i] << " ";
    // }
    // cout << endl;
    int* h_levels = (int*)malloc(nv * sizeof(int));
    for (int i = 0; i < nv; i++) {
        h_levels[i] = -1;  // -1 indicates undiscovered
    }
    h_levels[1] = 0;
    int* d_frontier;
    CHECK_CUDA(cudaMalloc((void**)&d_frontier, nv * sizeof(vertexId_t)));
    auto start = std::chrono::high_resolution_clock::now();
    int * levels = runBFS(&custing2, 1 ,h_levels,d_frontier);
    auto end = std::chrono::high_resolution_clock::now();

    cout << "Finished BFS.\n\n";
    cout<<"nv: "<<cuInit.csrNV<<" "<<"ne: "<<cuInit.csrNE<<endl;  

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Convert duration to hours, minutes, seconds, and milliseconds
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    // Display the elapsed time
    std::cout << "Time taken for BFS: "
              << hours.count() << " hours, "
              << minutes.count() << " minutes, "
              << seconds.count() << " seconds, "
              << milliseconds.count() << " milliseconds" << std::endl;


    // 
    ////////////////
    //TESTING STREAMING BFS
    ////////////////////
        // int* d_levels;
        // CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));
        // CHECK_CUDA(cudaMemcpy(d_levels, h_levels, nv * sizeof(int), cudaMemcpyHostToDevice));

        // Load updates (additions or deletions)
        int update_edges[2] = {1, 6};   // Packed as: [source, destination]
        int num_updates = 1;            // Only one edge update in this batch
        bool isAddition = true;         // Indicate that this is an edge addition


        updateBFSUpdates(&custing2, levels, d_frontier, update_edges,num_updates, isAddition);

        // Sync and process further as required
            













    // Test scalability of edge insertions and deletions
    // for (int numBatchEdges = 1; numBatchEdges < ne; numBatchEdges *= 10) {
    //     for (int32_t i = 0; i < 1; i++) {
    //         cuStinger custing2(defaultInitAllocater, defaultUpdateAllocater);
    //         start_clock(ce_start, ce_stop);
    //         custing2.initializeCuStinger(cuInit);
    //         float initTime = end_clock(ce_start, ce_stop);
            
    //         cout << graphName << "," << nv << "," << ne << "," << numBatchEdges;
    //         cout << "," << initTime << flush;
            
    //         printcuStingerUtility(custing2, true);
            
    //         BatchUpdateData bud(numBatchEdges, true);
    //         if (isRmat) {
    //             double a = 0.55, b = 0.15, c = 0.15, d = 0.25;
    //             dxor128_env_t env;
    //             generateEdgeUpdatesRMAT(nv, numBatchEdges, bud.getSrc(), bud.getDst(), a, b, c, d, &env);
    //         } else {
    //             generateEdgeUpdates(nv, numBatchEdges, bud.getSrc(), bud.getDst());
    //         }
    //         BatchUpdate bu(bud);
            
    //         length_t allocs;
    //         start_clock(ce_start, ce_stop);
    //         custing2.edgeInsertions(bu, allocs);
    //         cout << "," << end_clock(ce_start, ce_stop);
    //         cout << "," << allocs;
    //         cout<<"bfs after edge inst"<<endl;
    //         runBFS(&custing2, 1 );
    //         printcuStingerUtility(custing2, false);
            
    //         start_clock(ce_start, ce_stop);
    //         custing2.edgeDeletions(bu);
    //         cout << "," << end_clock(ce_start, ce_stop);
    //         custing2.verifyEdgeDeletions(bu);

    //         cout<<"bfs after edge del"<<endl;
    //         runBFS(&custing2, 1 );
    //         printcuStingerUtility(custing2, false);
    //         cout << endl << flush;
            
    //         custing2.freecuStinger();
    //     } 
    // }

    
    
    free(off);
    free(adj);
    return 0;	
}
