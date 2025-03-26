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

#define CHECK_CUDA(call) {                                            \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

struct Update {
    bool is_addition; 
    vertexId_t u, v;  
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
__global__ void bfs_kernel(const cuStinger::cusVertexData* dVD,
    const vertexId_t nv,
    const int current_level,
    const int* frontier,
    const int frontier_size,
    int* levels,
    vertexId_t* next_frontier,
    int* next_count) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];  
    int numNeighbors = dVD->used[v];
    
    vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
    for (int i = 0; i < numNeighbors; i++) {
      vertexId_t nbr = nbrArray[i];
      
      
      if (atomicCAS(&levels[nbr], -1, current_level + 1) == -1) {
        int pos = atomicAdd(next_count, 1);
        next_frontier[pos] = nbr;
      }
    }
  }
}

int* runBFS(cuStinger* graph, int source ,int* h_levels,int* d_frontier) {
    
    int nv = graph->nv;
    
    int* d_levels;

    
    CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));

    
    CHECK_CUDA(cudaMemcpy(d_levels, h_levels, nv * sizeof(int), cudaMemcpyHostToDevice));

    int* d_next_frontier;
    
    CHECK_CUDA(cudaMalloc((void**)&d_next_frontier, nv * sizeof(vertexId_t)));

    
    
    int h_frontier_size = 1;
    CHECK_CUDA(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));

    
    int* d_next_count;
    CHECK_CUDA(cudaMalloc((void**)&d_next_count, sizeof(int)));

    int current_level = 0;
    while (h_frontier_size > 0) {
        
        CHECK_CUDA(cudaMemset(d_next_count, 0, sizeof(int)));

        
        int blockSize = 256;
        
        int gridSize = (h_frontier_size + blockSize - 1) / blockSize;
        bfs_kernel<<<gridSize, blockSize>>>(graph->dVD, nv, current_level, d_frontier,
                                            h_frontier_size, d_levels, d_next_frontier, d_next_count);
        
        CHECK_CUDA(cudaDeviceSynchronize());

        
        int h_next_count;
        CHECK_CUDA(cudaMemcpy(&h_next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

        
        int* temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;

        h_frontier_size = h_next_count;
        current_level++;
    }
    
    CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));

    
    for (int i = 0; i < nv; i++) {
        printf("Vertex %d: Level %d\n", i, h_levels[i]);
    }

    
    
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_frontier));
    CHECK_CUDA(cudaFree(d_next_frontier));
    CHECK_CUDA(cudaFree(d_next_count));
    return h_levels;
}

__global__ void bfs_update_add_kernel(const cuStinger::cusVertexData* dVD,  
    const int* add_edges,   
    const int num_edges,  
    int* levels,  
    vertexId_t* update_frontier,  
    int* update_count) {  
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid < num_edges) {  
    int u = add_edges[2 * tid];  
    int v = add_edges[2 * tid + 1];  
    
    if (levels[u] != -1) {  
      int new_level = levels[u] + 1;  
      
      int old = atomicCAS(&levels[v], -1, new_level);  
      
      
      int old_2 = atomicMin(&levels[v], new_level);
      

      if (old == -1 || old_2>new_level) {  
        
        int pos = atomicAdd(update_count, 1);  
        
        
        update_frontier[pos] = v;  
      }  
    }  
  }  
}






__global__ void bfs_update_del_kernel(const cuStinger::cusVertexData* dVD,
    const int* del_edges,   
    const int num_edges,
    int* levels,
    vertexId_t* update_frontier,
    int* update_count) {
    
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_edges) {
    int u = del_edges[2 * tid];
    int v = del_edges[2 * tid + 1];
    
    
    if (levels[u] == -1 || levels[v] != levels[u] + 1)
      return;
      
    
    
    int old_level = atomicExch(&levels[v], -1);
    
    if(old_level != -1) {
      int pos = atomicAdd(update_count, 1);
      update_frontier[pos] = v;
      
    }
  }
}

__global__ void bfs_invalidate_levels_kernel(const cuStinger::cusVertexData* dVD,
    const int frontier_size,
    const vertexId_t* frontier,
    int* levels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];
    if (levels[v] == -1) {  
      int numNeighbors = dVD->used[v];
      vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
      for (int i = 0; i < numNeighbors; i++) {
        int nbr = nbrArray[i];
        if (levels[nbr] > levels[v]) {  
          levels[nbr] = -1;
        }
      }
    }
  }
}

__global__ void bfs_recompute_kernel(const cuStinger::cusVertexData* dVD,
    const int nv,
    const int* frontier,
    const int frontier_size,
    int* levels,
    vertexId_t* next_frontier,
    int* next_count) {
    
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];
    
    int candidate = INT_MAX;
    int numNeighbors = dVD->used[v];
    vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
    
    for (int i = 0; i < numNeighbors; i++) {
      int nbr = nbrArray[i];
      int nbr_level = levels[nbr];
      if(nbr_level >= 0 && nbr_level != INT_MAX) {
        int old = min(candidate, nbr_level + 1);
      }
    }
    
    int new_level = (candidate == INT_MAX) ? -1 : candidate;
    
    
    
    int old_level = INT_MAX;
    old_level = atomicExch(&levels[v], new_level);

    if (new_level != old_level) {
      for (int i = 0; i < numNeighbors; i++) {
        int nbr = nbrArray[i];
        
        
        if (levels[nbr] != -1 && levels[nbr] > new_level + 1) {
          int prev = atomicMin(&levels[nbr], new_level + 1);
          if (prev > new_level + 1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = nbr;
          }
        }
      }
    }
  }
}

__global__ void bfs_kernel_update(const cuStinger::cusVertexData* dVD,
    const vertexId_t nv,
    
    const int* frontier,
    const int frontier_size,
    int* levels,
    vertexId_t* next_frontier,
    int* next_count
  ) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];  
    int numNeighbors = dVD->used[v];    
    vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
    for (int i = 0; i < numNeighbors; i++) {
      vertexId_t nbr = nbrArray[i];
      
      int current_level = levels[v];
      
      
      int old = atomicMin(&levels[nbr], current_level + 1);
      if (old == -1 || old > current_level + 1) {
        
        int pos = atomicAdd(next_count, 1);
        next_frontier[pos] = nbr;
      }
    }
  }
}

__global__ void bfs_kernel_connected(const cuStinger::cusVertexData* dVD,
  const vertexId_t nv,
  const int current_level,
  const int* frontier,
  const int frontier_size,
  int* levels,
  vertexId_t* next_frontier,
  int* next_count,
  const vertexId_t target_node,
  int* target_reached) {

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < frontier_size) {
  int v = frontier[tid];  
  int numNeighbors = dVD->used[v];
  vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
  for (int i = 0; i < numNeighbors; i++) {
    vertexId_t nbr = nbrArray[i];    
    if (nbr == target_node) {
      atomicAdd(target_reached, 1); 
      return;
    }
    
    if (atomicExch(&levels[nbr], current_level) == -1) {
      int pos = atomicAdd(next_count, 1);
      next_frontier[pos] = nbr;
    }
  }
}
}

__global__ void bfs_kernel_bt_src_v(const cuStinger::cusVertexData* dVD,
  const vertexId_t nv,
  const int current_level,
  const int* frontier,
  const int frontier_size,
  int* levels,
  vertexId_t* next_frontier,
  int* next_count,
int* v,
int* found,
int* len_v) {

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < frontier_size && found<len_v) {
  int v = frontier[tid];  
  int numNeighbors = dVD->used[v];
  
  vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
  for (int i = 0; i < numNeighbors; i++) {
    vertexId_t nbr = nbrArray[i];
    if (nbr == v) {
      atomicAdd(found, 1); 
      return;
    }
    
    
    if (atomicCAS(&levels[nbr], -1, current_level + 1) == -1) {
      int pos = atomicAdd(next_count, 1);
      next_frontier[pos] = nbr;
    }
  }
}
}

int* runBFS_bt_src_v(cuStinger* graph, int source ,int* h_levels,int* d_frontier,int* v, int* len_v) {  
  int nv = graph->nv;
  int* d_levels;
  CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_levels, h_levels, nv * sizeof(int), cudaMemcpyHostToDevice));
  int* d_next_frontier;
  CHECK_CUDA(cudaMalloc((void**)&d_next_frontier, nv * sizeof(vertexId_t)));
  int h_frontier_size = 1;
  CHECK_CUDA(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
  int* d_next_count;
  CHECK_CUDA(cudaMalloc((void**)&d_next_count, sizeof(int)));
  int* v_d;
  CHECK_CUDA(cudaMalloc((void**)&v_d, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(v_d,v, sizeof(int),cudaMemcpyHostToDevice));
  int* found;
  CHECK_CUDA(cudaMalloc((void**)&found, sizeof(int)));
  int* len_v_d;
  CHECK_CUDA(cudaMalloc((void**)&len_v_d, nv*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(len_v_d,len_v,nv*sizeof(int),cudaMemcpyHostToDevice));
  int current_level = 0;
  while (h_frontier_size > 0) {
      
      CHECK_CUDA(cudaMemset(d_next_count, 0, sizeof(int)));

      
      int blockSize = 256;
      
      int gridSize = (h_frontier_size + blockSize - 1) / blockSize;
      bfs_kernel_bt_src_v<<<gridSize, blockSize>>>(graph->dVD, nv, current_level, d_frontier,
                                          h_frontier_size, d_levels, d_next_frontier, d_next_count, v_d, found,len_v_d);
      
      CHECK_CUDA(cudaDeviceSynchronize());

      
      int h_next_count;
      CHECK_CUDA(cudaMemcpy(&h_next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

      
      int* temp = d_frontier;
      d_frontier = d_next_frontier;
      d_next_frontier = temp;

      h_frontier_size = h_next_count;
      current_level++;
  }

  
  CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));

  
  for (int i = 0; i < nv; i++) {
      printf("Vertex %d: Level %d\n", i, h_levels[i]);
  }

  
  
  CHECK_CUDA(cudaFree(d_levels));
  CHECK_CUDA(cudaFree(d_frontier));
  CHECK_CUDA(cudaFree(d_next_frontier));
  CHECK_CUDA(cudaFree(d_next_count));
  return h_levels;
}

void updateBFSUpdates(cuStinger* graph, int* h_levels, int* d_frontier,
                      int* h_update_edges, int num_updates, bool isAddition, int source) {
  int nv = graph->nv;

  
  int* d_update_edges;
  CHECK_CUDA(cudaMalloc((void**)&d_update_edges, 2 * num_updates * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_update_edges, h_update_edges,
                        2 * num_updates * sizeof(int), cudaMemcpyHostToDevice));

  int* d_levels;
  CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_levels, h_levels,
                        nv * sizeof(int), cudaMemcpyHostToDevice));

  
  vertexId_t* d_update_frontier;
  vertexId_t* d_next_frontier;

  vertexId_t* h_update_frontier = (vertexId_t*)malloc(nv * sizeof(vertexId_t));
  CHECK_CUDA(cudaMalloc((void**)&d_update_frontier, nv * sizeof(vertexId_t)));
  CHECK_CUDA(cudaMalloc((void**)&d_next_frontier, nv * sizeof(vertexId_t)));

  int* d_update_count;
  CHECK_CUDA(cudaMalloc((void**)&d_update_count, sizeof(int)));

  int h_update_count = 0;
  int blockSize = 256;
  int gridSize = (num_updates + blockSize - 1) / blockSize;

  
  if (isAddition) {
    bfs_update_add_kernel<<<gridSize, blockSize>>>(graph->dVD, d_update_edges, num_updates,
                                                     d_levels, d_update_frontier, d_update_count);

      
    CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    

  
    
    
    while (h_update_count > 0) {

        
        CHECK_CUDA(cudaMemset(d_update_count, 0, sizeof(int)));
        gridSize = (h_update_count + blockSize - 1) / blockSize;
        
        bfs_kernel_update<<<gridSize, blockSize>>>(graph->dVD, nv, d_update_frontier,
                                            h_update_count, d_levels, d_next_frontier, d_update_count);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        vertexId_t* temp;
        CHECK_CUDA(cudaMalloc((void**)&temp, nv * sizeof(vertexId_t)));
        d_update_frontier = d_next_frontier;
        d_next_frontier = temp;
        
        CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
  }      
   CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));
   
   for (int i = 0; i < nv; i++) {
    printf("Vertex %d: Level %d\n", i, h_levels[i]);
}                                           
   
  } else {
    printf("reached else condition\n");
    int del_current_level = 0;
    int v[num_updates];
    for (int i=0;i<num_updates;i++) {
      v[i] = h_update_edges[2*i+1];
    }
    int* level =runBFS_bt_src_v(graph, source,h_levels, d_frontier, v, &num_updates);
    
    
    for (int i=0;i<v[num_updates-1];i++) {
      h_levels[i] = level[i];
    }
    h_update_count = num_updates;
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(v, d_update_frontier, nv*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyHostToDevice));
    while (h_update_count > 0) {

      
      CHECK_CUDA(cudaMemset(d_update_count, 0, sizeof(int)));
      gridSize = (h_update_count + blockSize - 1) / blockSize;
      
      bfs_kernel_update<<<gridSize, blockSize>>>(graph->dVD, nv, d_update_frontier,
                                          h_update_count, d_levels, d_next_frontier, d_update_count);
      CHECK_CUDA(cudaDeviceSynchronize());
      
      vertexId_t* temp;
      CHECK_CUDA(cudaMalloc((void**)&temp, nv * sizeof(vertexId_t)));
      d_update_frontier = d_next_frontier;
      d_next_frontier = temp;
      
      CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
}      
 CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));
 
 for (int i = 0; i < nv; i++) {
  printf("Vertex %d: Level %d\n", i, h_levels[i]);
}  
  }                       

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(d_update_edges));
  CHECK_CUDA(cudaFree(d_update_frontier));
  CHECK_CUDA(cudaFree(d_update_count));
}

void printcuStingerUtility(cuStinger custing, bool allInfo) {
    length_t used, allocated;
    used = custing.getNumberEdgesUsed();
    allocated = custing.getNumberEdgesAllocated();
    if (allInfo)
        cout << "," << used << "," << allocated << "," << (float)used / (float)allocated;  
    else
        cout << "," << (float)used / (float)allocated;
}

void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst, int* update_edges) {
    for (int e = 0; e < numEdges; e++) {
        edgeSrc[e] = update_edges[2*e] % nv;
        edgeDst[e] = update_edges[2*e+1] % nv;
    }
}


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



int main(const int argc, char *argv[])
{  
    
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

    
    int* h_levels = (int*)malloc(nv * sizeof(int));
    for (int i = 0; i < nv; i++) {
        h_levels[i] = -1;  
    }
    h_levels[1] = 0;
    int* d_frontier;
    CHECK_CUDA(cudaMalloc((void**)&d_frontier, nv * sizeof(vertexId_t)));

    cout << "Running BFS on the input graph ...\n";
    auto start = std::chrono::high_resolution_clock::now();
    int * levels = runBFS(&custing2, 1 ,h_levels,d_frontier);
    auto end = std::chrono::high_resolution_clock::now();

    cout << "Finished BFS.\n\n";
    cout<<"nv: "<<cuInit.csrNV<<" "<<"ne: "<<cuInit.csrNE<<endl;  

    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    
auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
duration -= hours;
auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
duration -= minutes;
auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
duration -= seconds;
auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
duration -= milliseconds;
auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);


std::cout << "Time taken for BFS: "
          << hours.count() << " hours, "
          << minutes.count() << " minutes, "
          << seconds.count() << " seconds, "
          << milliseconds.count() << " milliseconds, "
          << microseconds.count() << " microseconds" << std::endl;

        
        int update_edges[4] = {1, 8, 1, 6};   
        int num_updates = 2;            
        bool isAddition = true;         

       

        
        updateBFSUpdates(&custing2, levels, d_frontier, update_edges,num_updates, isAddition,1);
printf(" ADDITION DONE\n");
        
        
        
        int numEdgesToDelete = 2;
        BatchUpdateData bud(numEdgesToDelete, true, nv);  
        vertexId_t* src = bud.getSrc();
        vertexId_t* dst = bud.getDst();

        
        
        
        


{
  int update_edges_fwd[] = {2, 4, 6, 7}; 
  length_t numEdges = 2;  
  BatchUpdateData budFwd(numEdges, true);
  generateEdgeUpdates(nv, numEdges, budFwd.getSrc(), budFwd.getDst(), update_edges_fwd);
  BatchUpdate buFwd(budFwd);
  custing2.edgeDeletions(buFwd);
  
  custing2.verifyEdgeDeletions(buFwd);
}


{
  int update_edges_rev[] = { 4,2,7,6 }; 
  length_t numEdges = 2;  
  BatchUpdateData budRev(numEdges, true);
  generateEdgeUpdates(nv, numEdges, budRev.getSrc(), budRev.getDst(), update_edges_rev);
  BatchUpdate buRev(budRev);
  custing2.edgeDeletions(buRev);
  
  custing2.verifyEdgeDeletions(buRev);
}
        int update_edges_del[] = {2, 4, 6, 7}; 
        int update_edges_batch_del[] = {4,2,7,6}; 
        printf("Running BFS with edge deletions...\n"); 
        int* h_levels_del = (int*)malloc(nv * sizeof(int));
        for (int i = 0; i < nv; i++) {
            h_levels_del[i] = -1;  
        }
        h_levels_del[1] = 0;
        int* d_frontier_del;
        CHECK_CUDA(cudaMalloc((void**)&d_frontier_del, nv * sizeof(vertexId_t)));
        int * levels_del=runBFS(&custing2, 1 ,h_levels_del,d_frontier_del);
        printf("Running with updateBFSupdates..\n\n");

        updateBFSUpdates(&custing2, levels, d_frontier, update_edges_del ,num_updates, false,1);

        int* h_levels_rev = (int*)malloc(nv * sizeof(int));
        for (int i = 0; i < nv; i++) {
            h_levels_rev[i] = -1;  
        }
        h_levels_rev[7] = 0;
        int* d_frontier_rev;
        CHECK_CUDA(cudaMalloc((void**)&d_frontier_rev, nv * sizeof(vertexId_t)));
        int * levels_rev=runBFS(&custing2, 7 ,h_levels_rev,d_frontier_rev);
        custing2.freecuStinger();

    
    
    free(off);
    free(adj);
    return 0;	
}
