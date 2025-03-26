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
      // printf("v-->nbr: %d-->%d\n", v,nbr);
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

    ///////Print BFS levels
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
  // printf("tid: %d\n", tid); 

    int u = add_edges[2 * tid];  
    int v = add_edges[2 * tid + 1];  
    // Process only if u is discovered.
    // printf("u: %d, v: %d\n", u, v);
    // printf("fine");
    // printf("levels[u]: %d\n", levels[u]);
    if (levels[u] != -1) {  
      int new_level = levels[u] + 1;  
      // Use atomicCAS to update levels[v] only if it is still undiscovered (-1)
      int old = atomicCAS(&levels[v], -1, new_level);  
      // printf("old: %d\n", old);
      
      int old_2 = atomicMin(&levels[v], new_level);
      

      if (old == -1 || old_2>new_level) {  // Successful update: v was undiscovered  
        // printf("update_count: %d\n", *(update_count));
        int pos = atomicAdd(update_count, 1);  
        // printf("update_count: %d\n", *(update_count));
        // printf("pos: %d\n", pos);   
        update_frontier[pos] = v;  
      }  
    }  
  }  
}



//////////////////////////////////////
// BFS Update Kernel for Edge Deletions
//////////////////////////////////////
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
    // Only process if (u,v) was used in the BFS tree:
    // u must have been reached and v’s level must be exactly levels[u] + 1.
    if (levels[u] == -1 || levels[v] != levels[u] + 1)
      return;
      
    // Mark v for recomputation: use atomicExch so that only one thread marks v.
    //Thats wrong as we are marking v as -1 i tmay be connected to other nodes
    int old_level = atomicExch(&levels[v], -1);
    // If v was not already marked, add it to the update frontier.
    if(old_level != -1) {
      int pos = atomicAdd(update_count, 1);
      update_frontier[pos] = v;
      // printf("---------inside kernel update_frontier: %d-----v:%d-----\n", update_frontier[pos],v); 
    }
  }
}

//////////////////////////////////////
// Kernel to Invalidate Neighbor Levels
//////////////////////////////////////

__global__ void bfs_invalidate_levels_kernel(const cuStinger::cusVertexData* dVD,
    const int frontier_size,
    const vertexId_t* frontier,
    int* levels) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];
    if (levels[v] == -1) {  // Process only invalidated nodes
      int numNeighbors = dVD->used[v];
      vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
      for (int i = 0; i < numNeighbors; i++) {
        int nbr = nbrArray[i];
        if (levels[nbr] > levels[v]) {  // Mark higher-level nodes for recomputation
          levels[nbr] = -1;
        }
      }
    }
  }
}

//////////////////////////////////////
// BFS Recompute Kernel for Propagation
//////////////////////////////////////

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
    // Recompute the new level for v.
    int candidate = INT_MAX;
    int numNeighbors = dVD->used[v];
    vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
    
    // For each neighbor, consider candidate level = levels[nbr] + 1.
    // Only consider neighbors with a valid (non-INT_MAX, non -1) level.
    for (int i = 0; i < numNeighbors; i++) {
      int nbr = nbrArray[i];
      int nbr_level = levels[nbr];
      if(nbr_level >= 0 && nbr_level != INT_MAX) {
        int old = min(candidate, nbr_level + 1);
      }
    }
    
    // If no valid candidate was found, v remains unreachable.
    int new_level = (candidate == INT_MAX) ? -1 : candidate;
    
    // Update v’s level if it differs from the (marked) value.
    // Note: v was marked as INT_MAX, so we expect new_level != INT_MAX.
    int old_level = INT_MAX;
    old_level = atomicExch(&levels[v], new_level);

    if (new_level != old_level) {
      for (int i = 0; i < numNeighbors; i++) {
        int nbr = nbrArray[i];
        // If neighbor has a valid level and its current level is greater than new_level + 1,
        // try to update it.
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

//////////////////////////////////////
// BFS Kernel to handke updates
//////////////////////////////////////
__global__ void bfs_kernel_update(const cuStinger::cusVertexData* dVD,
    const vertexId_t nv,
    // int current_level,
    const int* frontier,
    const int frontier_size,
    int* levels,
    vertexId_t* next_frontier,
    int* next_count
  ) {
  // thread id
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    int v = frontier[tid];  // current vertex from frontier
    // printf("----tid---: %d\n", tid);
    // printf("---v---: %d\n", v);
    int numNeighbors = dVD->used[v];
    // printf("----numNeighbors---: %d\n", numNeighbors);
    // reinterpret pointer to edge memory
    vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
    for (int i = 0; i < numNeighbors; i++) {
      vertexId_t nbr = nbrArray[i];
      // printf("----nbr---: %d\n", nbr);
      int current_level = levels[v];
      // printf("----current_level_: %d\n", current_level);
      // If not yet discovered then set level and add to next frontier.
      int old = atomicMin(&levels[nbr], current_level + 1);
      if (old == -1 || old > current_level + 1) {
        // printf("Sauda chala\n");
        int pos = atomicAdd(next_count, 1);
        next_frontier[pos] = nbr;
      }
    }
  }
}
//////////////////////////////////////
// BFS Kernel to check connected or not 
//////////////////////////////////////
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
// Thread ID
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < frontier_size) {
  int v = frontier[tid];  // Current vertex from frontier
  int numNeighbors = dVD->used[v];
  // printf("numNeighbors: %d\n", numNeighbors);
  // Reinterpret pointer to edge memory
  vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
  for (int i = 0; i < numNeighbors; i++) {
    vertexId_t nbr = nbrArray[i];
    // while(nbr == -1 && i<10) {
    //   i++;
    //   nbr = nbrArray[i];
    //   printf("i--- %d\n", i);
    //   printf("nbr--- %d\n", nbr);
      
    // }
    
    // printf("i=%d\n", i);
    // printf("v-->nbr: %d-->%d\n", v,nbr);  
    // If the target node is reached
    if (nbr == target_node) {
      atomicAdd(target_reached, 1); // Signal discovery
      return;
    }
    // If not yet discovered, set level and add to next frontier
    if (atomicExch(&levels[nbr], current_level) == -1) {
      int pos = atomicAdd(next_count, 1);
      next_frontier[pos] = nbr;
    }
  }
}
}

// //////////////////////////////////////
// // BFS Kernel to find if subgraph is connected
// //////////////////////////////////////
// __global__ void is_connected(const cuStinger::cusVertexData* dVD,
//   const vertexId_t nv,
//   const int current_level,
//   const int* frontier,
//   const int frontier_size,
//   int* levels,
//   vertexId_t* next_frontier,
//   int* next_count,
//   const vertexId_t target,   // target vertex (b)
//   int* found)                // global flag (0: not found, 1: found)
// {
// // thread id
// int tid = blockIdx.x * blockDim.x + threadIdx.x;

// // Early exit: if target already found by another thread, skip processing.
// if (atomicAdd(found, 0) == 1) return;

// if (tid < frontier_size) {
// int v = frontier[tid];  // current vertex from frontier
// int numNeighbors = dVD->used[v];

// // reinterpret pointer to edge memory
// vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);

// // Process each neighbor
// for (int i = 0; i < numNeighbors; i++) {
// // Check global flag frequently so we don't do extra work
// if (atomicAdd(found, 0) == 1) return;

// vertexId_t nbr = nbrArray[i];

// // Attempt to mark the neighbor as discovered (set its level)
// if (atomicCAS(&levels[nbr], -1, current_level + 1) == -1) {
// // If this neighbor is our target vertex, signal discovery.
// if (nbr == target) {
// atomicExch(found, 1);
// // Optionally, you can break out immediately since the target is found.
// return;
// }
// // Otherwise, add this neighbor to the next frontier.
// int pos = atomicAdd(next_count, 1);
// next_frontier[pos] = nbr;
// }
// }
// }
// }
//////////////////////////////////////
// BFS Kernel betweeen src and v
//////////////////////////////////////
__global__ void bfs_kernel_bt_src_v(const cuStinger::cusVertexData* dVD,
  const vertexId_t nv,
  const int current_level,
  const int* frontier,
  const int frontier_size,
  int* levels,
  vertexId_t* next_frontier,
  int* next_count,
vertexId_t* v_d,
int* found,
int* len_v,
int * found_ver_arr) {
// thread id
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < frontier_size && found<len_v) {
  int v = frontier[tid];  // current vertex from frontier
  printf("----v: %d\n", v);

  int numNeighbors = dVD->used[v];
  // reinterpret pointer to edge memory
  vertexId_t* nbrArray = reinterpret_cast<vertexId_t*>(dVD->edMem[v]);
  for (int i = 0; i < numNeighbors; i++) {
    vertexId_t nbr = nbrArray[i];
    printf("--nbr: %d\n", nbr);

    // printf("---v: %d\n", (v_d[1]));
    // printf("---len_v: %d\n", *len_v);

    for(int j=0;j<*len_v;j++) {
      printf("v_d[j]: %d\n", v_d[j]);
      if (nbr == v_d[j]) {
        printf("nbr is in v_d\n");
        found_ver_arr[*found] = nbr;
        atomicAdd(found, 1); // Signal discovery
      }
    }
    // printf("v-->nbr: %d-->%d\n", v,nbr);
 
    // printf("v-->nbr: %d-->%d\n", v,nbr);
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
int* runBFS_bt_src_v(cuStinger* graph, int source ,int* h_levels,int* d_frontier,vertexId_t* v, int* len_v , int * found_cnt, int * found_ver_arr) {
  // nv = number of vertices
  int nv = graph->nv;
  // array of levels on GPU
  int* d_levels;
  int * d_found_ver_arr;

  // if CUDA has error -> inform
  CHECK_CUDA(cudaMalloc((void**)&d_levels, nv * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_found_ver_arr, (*len_v) * sizeof(int)));

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

  vertexId_t* v_d;
  CHECK_CUDA(cudaMalloc((void**)&v_d, (*(len_v))*sizeof(vertexId_t)));
  CHECK_CUDA(cudaMemcpy(v_d,v, (*(len_v))*sizeof(vertexId_t),cudaMemcpyHostToDevice));
  printf("v_d[0]: %d\n", v[1]);

  int* found;
  CHECK_CUDA(cudaMalloc((void**)&found, sizeof(int)));
  // CHECK_CUDA(cudaMemcpy(v_d,v, sizeof(int),cudaMemcpyHostToDevice));
  int found_h = 0;
  int* len_v_d;
  CHECK_CUDA(cudaMalloc((void**)&len_v_d, nv*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(len_v_d,len_v,nv*sizeof(int),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(found,&found_h, sizeof(int),cudaMemcpyHostToDevice));

  int current_level = 0;
  while (h_frontier_size > 0&& found_h<*len_v) {
      // Reset next frontier count
      CHECK_CUDA(cudaMemset(d_next_count, 0, sizeof(int)));

      // Launch BFS kernel
      int blockSize = 256;
      // which thread runs where 
      int gridSize = (h_frontier_size + blockSize - 1) / blockSize;
      bfs_kernel_bt_src_v<<<gridSize, blockSize>>>(graph->dVD, nv, current_level, d_frontier,
                                          h_frontier_size, d_levels, d_next_frontier, d_next_count, v_d, found,len_v_d,d_found_ver_arr);
      // cout<<"BFS KERNEL LAUNCHED"<<endl;
      CHECK_CUDA(cudaDeviceSynchronize());

      // Copy next frontier size back to host
      int h_next_count;
      CHECK_CUDA(cudaMemcpy(&h_next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

      // Swap frontiers
      int* temp = d_frontier;
      d_frontier = d_next_frontier;
      d_next_frontier = temp;
      CHECK_CUDA(cudaMemcpy(found,&found_h, sizeof(int),cudaMemcpyHostToDevice));
      
      h_frontier_size = h_next_count;
      current_level++;
      printf("--current_level: %d\n", current_level);
      printf("--found: %d\n", found_h);
  }

  // Copy levels array back to host
  CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));

  ///////Print BFS levels
  for (int i = 0; i < nv; i++) {
      printf("src - v Vertex %d: Level %d\n", i, h_levels[i]);
  }
  *found_cnt = found_h;

  // Free allocated memory
  // free(h_levels);
  CHECK_CUDA(cudaFree(d_levels));
  CHECK_CUDA(cudaFree(d_frontier));
  CHECK_CUDA(cudaFree(d_next_frontier));
  CHECK_CUDA(cudaFree(d_next_count));
  return h_levels;
}

//////////////////////////////////////
// Host-Side Streaming Update Handler
//////////////////////////////////////
// This function applies a batch of update edges (additions or deletions)
// and then “propagates” the changes via BFS until no further level changes occur.
void updateBFSUpdates(cuStinger* graph, int* h_levels, int* d_frontier,
                      int* h_update_edges, int num_updates, bool isAddition, int source) {
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
  vertexId_t* d_next_frontier;

  vertexId_t* h_update_frontier = (vertexId_t*)malloc(nv * sizeof(vertexId_t));
  CHECK_CUDA(cudaMalloc((void**)&d_update_frontier, nv * sizeof(vertexId_t)));
  CHECK_CUDA(cudaMalloc((void**)&d_next_frontier, nv * sizeof(vertexId_t)));

  int* d_update_count;
  CHECK_CUDA(cudaMalloc((void**)&d_update_count, sizeof(int)));

  int h_update_count = 0;
  int blockSize = 256;
  int gridSize = (num_updates + blockSize - 1) / blockSize;

  // Launch the appropriate update kernel.
  if (isAddition) {
    bfs_update_add_kernel<<<gridSize, blockSize>>>(graph->dVD, d_update_edges, num_updates,
                                                     d_levels, d_update_frontier, d_update_count);

      // Get the count of vertices in the update frontier.
    CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    

  // Propagate the update changes until the frontier is empty.
    // int current_level = 0;  // You might want to adjust this if levels are relative to a root.
    // printf("h_update_count: %d\n", h_update_count);
    while (h_update_count > 0) {

        // Reset the counter for the next propagation step.
        CHECK_CUDA(cudaMemset(d_update_count, 0, sizeof(int)));
        gridSize = (h_update_count + blockSize - 1) / blockSize;
        // Use the original BFS kernel to propagate updated levels.
        bfs_kernel_update<<<gridSize, blockSize>>>(graph->dVD, nv, d_update_frontier,
                                            h_update_count, d_levels, d_next_frontier, d_update_count);
        CHECK_CUDA(cudaDeviceSynchronize());
        // Swap frontiers.
        vertexId_t* temp;
        CHECK_CUDA(cudaMalloc((void**)&temp, nv * sizeof(vertexId_t)));
        d_update_frontier = d_next_frontier;
        d_next_frontier = temp;
        // Get the new update frontier count.
        CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
        // printf("h_update_count: %d\n", h_update_count);

        // current_level++;
  }      // Copy levels array back to host
   CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));
   // Print BFS levels
   for (int i = 0; i < nv; i++) {
    printf("Vertex %d: Level %d\n", i, h_levels[i]);
}                                           
           
  } else {
    printf("reached else condition\n");
    // bfs_update_del_kernel<<<gridSize, blockSize>>>(graph->dVD, d_update_edges, num_updates,
    //     d_levels, d_update_frontier, d_update_count);
    //   CHECK_CUDA(cudaDeviceSynchronize());
    //   CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
    //   //needed only till DEVESH DON't SEE BATCHUPDATES CLASS
    //   //
    //   /////////////////////////
    //   is_connected<<<gridSize, blockSize>>>(graph->dVD, nv, 0, d_update_frontier, h_update_count, d_levels, d_next_frontier, d_update_count, 1, d_update_count);
    //   CHECK_CUDA(cudaDeviceSynchronize());
    //     printf("h_update_count: %d\n", h_update_count);
    //     while (h_update_count > 0) {

    //         // Reset the counter for the next propagation step.
    //         CHECK_CUDA(cudaMemset(d_update_count, 0, sizeof(int)));
    //         gridSize = (h_update_count + blockSize - 1) / blockSize;
    //         // Use the original BFS kernel to propagate updated levels.
    //       // For deletions, use the recompute kernel to recalc affected levels.
    //       bfs_recompute_kernel<<<gridSize, blockSize>>>(graph->dVD, nv, d_update_frontier,
    //         h_update_count, d_levels, d_next_frontier, d_update_count);

    //         CHECK_CUDA(cudaDeviceSynchronize());
    //         // Swap frontiers.
    //         vertexId_t* temp;
    //         CHECK_CUDA(cudaMalloc((void**)&temp, nv * sizeof(vertexId_t)));
    //         d_update_frontier = d_next_frontier;
    //         d_next_frontier = temp;
    //         // Get the new update frontier count.
    //         CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
    //         printf("h_update_count: %d\n", h_update_count);

    //         // current_level++;
    
    int del_current_level = 0;
    vertexId_t* v_arr=(vertexId_t*)malloc(num_updates*sizeof(vertexId_t));
    
    for (int i=0;i<num_updates;i++) {
      v_arr[i] = h_update_edges[2*i+1];
      cout<<"v_arr[i]: "<<v_arr[i]<<endl;
    }
    printf("starting bfs bt src and v\n");
    int* h_levels_del = (int*)malloc(nv * sizeof(int));
        for (int i = 0; i < nv; i++) {
            h_levels_del[i] = -1;  // -1 indicates undiscovered
        }
        h_levels_del[1] = 0;
    int found = 0;
    int * found_ver_arr= (int*)malloc(num_updates*sizeof(int));
    int* level =runBFS_bt_src_v(graph, source,h_levels_del, d_frontier, v_arr, &num_updates,&found,found_ver_arr);
    printf("bfs bt src and v done\n");
    cout<<"found: "<<found<<endl;
    cout<<"num_updates: "<<num_updates<<endl;
    // bfs_update_del_kernel<<<gridSize, blockSize>>>(graph->dVD, d_update_edges, num_updates,
    //       d_levels, d_update_frontier, d_update_count);
   
    if(found<num_updates){
      for (int i=0;i<nv;i++) {
        h_levels[i] = level[i];
      printf("...Vertex %d: Level %d\n", i, level[i]);

      }
    return;
    }

    for (int i=0;i<=v_arr[num_updates-1];i++) {
      h_levels[i] = level[i];
    }


    h_update_count = found;
    CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaMemcpy(&num_updates, d_update_count, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_update_frontier,found_ver_arr, num_updates*sizeof(vertexId_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_levels, h_levels, nv * sizeof(int), cudaMemcpyHostToDevice));
    while (h_update_count > 0) {

      // Reset the counter for the next propagation step.
      CHECK_CUDA(cudaMemset(d_update_count, 0, sizeof(int)));
      gridSize = (h_update_count + blockSize - 1) / blockSize;
      // Use the original BFS kernel to propagate updated levels.
      bfs_kernel_update<<<gridSize, blockSize>>>(graph->dVD, nv, d_update_frontier,
                                          h_update_count, d_levels, d_next_frontier, d_update_count);
      CHECK_CUDA(cudaDeviceSynchronize());
      // Swap frontiers.
      vertexId_t* temp;
      CHECK_CUDA(cudaMalloc((void**)&temp, nv * sizeof(vertexId_t)));
      d_update_frontier = d_next_frontier;
      d_next_frontier = temp;
      // Get the new update frontier count.
      CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
      // printf("h_update_count: %d\n", h_update_count);

      // current_level++;
}      // Copy levels array back to host
 CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));
 // Print BFS levels
 for (int i = 0; i < nv; i++) {
  printf("----Vertex %d: Level %d\n", i, h_levels[i]);
}  

//     CHECK_CUDA(cudaMemcpy(&h_update_count, d_update_count, sizeof(int), cudaMemcpyDeviceToHost));
//     CHECK_CUDA(cudaMemcpy(h_update_frontier, d_update_frontier, nv*sizeof(int), cudaMemcpyDeviceToHost));

//     int* d_next_count;
//     int connected = 0;
//     int h_frontier_size;
//     vertexId_t* d_del_frontier;
//     cudaMalloc((void**)&d_del_frontier, nv*sizeof(vertexId_t));
//     printf("h_update_count: %d\n", h_update_count);
 

//     for (int i = 0; i < h_update_count; i++) {
//       //kernel to check if the all node are connected or not
//       connected = 0;
//       h_frontier_size = 1;
//       CHECK_CUDA(cudaMemcpy(d_del_frontier, &(h_update_frontier[i]), sizeof(vertexId_t), cudaMemcpyHostToDevice));
//       CHECK_CUDA(cudaMalloc((void**)&d_next_count, sizeof(int)));
//       while(!connected && h_frontier_size > 0) {
//         printf("reached while loop with h_update_fronyier[i]%d\n", h_update_frontier[i]);
//         // Reset next frontier count
//         CHECK_CUDA(cudaMemset(d_next_count, 0, sizeof(int)));

//         // Launch BFS kernel
//         int blockSize = 256;
//         // which thread runs where 
//         int gridSize = (h_frontier_size + blockSize - 1) / blockSize;
//         bfs_kernel_connected<<<gridSize, blockSize>>>(graph->dVD, nv, del_current_level,d_del_frontier,h_frontier_size, d_levels, d_next_frontier, d_next_count, source, &connected);
//         CHECK_CUDA(cudaDeviceSynchronize());
//         int h_next_count;
//         CHECK_CUDA(cudaMemcpy(&h_next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

//         // Swap frontiers
//         int* temp = d_del_frontier;
//         d_del_frontier = d_next_frontier;
//         d_next_frontier = temp;

//         h_frontier_size = h_next_count;
//       }
//       printf("connected: %d\n", connected);
//       if (connected) { 
//         for (int i = 0; i < nv; i++) {
//           h_levels[i] = -1;  // -1 indicates undiscovered
//       }
//       h_levels[1] = 0;
//         runBFS(graph, source, h_levels, d_del_frontier);


//     } 
//     else {
//       printf("The graph is not connected\n");
//       CHECK_CUDA(cudaMemcpy(h_levels, d_levels, nv * sizeof(int), cudaMemcpyDeviceToHost));
//    // Print BFS levels
//    for (int i = 0; i < nv; i++) {
//     printf("Vertex %d: Level %d\n", i, h_levels[i]);
// }  
//     }







  
  }                       

  CHECK_CUDA(cudaDeviceSynchronize());


   

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
void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst, int* update_edges) {
    for (int e = 0; e < numEdges; e++) {
        edgeSrc[e] = update_edges[2*e] % nv;
        edgeDst[e] = update_edges[2*e+1] % nv;
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

    
    int* h_levels = (int*)malloc(nv * sizeof(int));
    for (int i = 0; i < nv; i++) {
        h_levels[i] = -1;  // -1 indicates undiscovered
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

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Convert duration to hours, minutes, seconds, milliseconds, and microseconds
auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
duration -= hours;
auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
duration -= minutes;
auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
duration -= seconds;
auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
duration -= milliseconds;
auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);

// Display the elapsed time
std::cout << "Time taken for BFS: "
          << hours.count() << " hours, "
          << minutes.count() << " minutes, "
          << seconds.count() << " seconds, "
          << milliseconds.count() << " milliseconds, "
          << microseconds.count() << " microseconds" << std::endl;

        // Load updates (additions or deletions)
        int update_edges[4] = {1, 8, 1, 6};   // Packed as: [source, destination]
        int num_updates = 2;            // Only one edge update in this batch
        bool isAddition = true;         // Indicate that this is an edge addition

       

        
        updateBFSUpdates(&custing2, levels, d_frontier, update_edges,num_updates, isAddition,1);
printf(" ADDITION DONE\n");
        /////////////////////////
        //DELETION OF EDGE
        /////////////////////////
        int numEdgesToDelete = 2;
        BatchUpdateData bud(numEdgesToDelete, true, nv);  // Make sure nv is set to the number of vertices in your graph
        vertexId_t* src = bud.getSrc();
        vertexId_t* dst = bud.getDst();

        // src[0] = 2; dst[0] =4;   // Delete edge 2-4
        // src[1] = 6; dst[1] = 7;   // Delete edge 6-7
        // src[0] = 4; dst[0] =2;   // Delete edge 2-4
        // src[1] = 7; dst[1] = 6;   // Delete edge 6-7

// For bidirected edge deletion, first delete 1->5
{
  int update_edges_fwd[] = {2, 4, 6, 7}; // first direction
  length_t numEdges = 2;  // One deletion update
  BatchUpdateData budFwd(numEdges, true);
  generateEdgeUpdates(nv, numEdges, budFwd.getSrc(), budFwd.getDst(), update_edges_fwd);
  BatchUpdate buFwd(budFwd);
  custing2.edgeDeletions(buFwd);
  // Optionally verify deletion: 
  custing2.verifyEdgeDeletions(buFwd);
}

// Then, delete the reverse edge 5->1
{
  int update_edges_rev[] = { 4,2,7,6 }; // reverse direction
  length_t numEdges = 2;  // One deletion update
  BatchUpdateData budRev(numEdges, true);
  generateEdgeUpdates(nv, numEdges, budRev.getSrc(), budRev.getDst(), update_edges_rev);
  BatchUpdate buRev(budRev);
  custing2.edgeDeletions(buRev);
  // Optionally verify deletion: 
  custing2.verifyEdgeDeletions(buRev);
}
        int update_edges_del[] = {2, 4, 6, 7}; // Edge 1->2 and 6->7
        int update_edges_batch_del[] = {4,2,7,6}; // Edge 1->2 and 6->7

        // generateEdgeUpdates(nv, 2, bud.getSrc(), bud.getDst(), update_edges_del);
        // BatchUpdate bu(bud);
        // length_t allocs;
        // custing2.edgeDeletions(bu);
        // custing2.verifyEdgeDeletions(bu);
        // generateEdgeUpdates(nv, 2, bud.getSrc(), bud.getDst(), update_edges_batch_del);
        // BatchUpdate bub(bud);
        // // length_t allocs;
        // custing2.edgeDeletions(bub);
        // custing2.verifyEdgeDeletions(bub);
        // //Run BFS with edge deletions
        printf("Running BFS with edge deletions...\n"); 
        int* h_levels_del = (int*)malloc(nv * sizeof(int));
        for (int i = 0; i < nv; i++) {
            h_levels_del[i] = -1;  // -1 indicates undiscovered
        }
        h_levels_del[1] = 0;
        int* d_frontier_del;
        CHECK_CUDA(cudaMalloc((void**)&d_frontier_del, nv * sizeof(vertexId_t)));
        int * levels_del=runBFS(&custing2, 1 ,h_levels_del,d_frontier_del);
        printf("Running with updateBFSupdates..\n\n");

        updateBFSUpdates(&custing2, levels, d_frontier, update_edges_del ,num_updates, false,1);


        
        // printf("Running BFS on reverting the deletions on input graph ...\n");

        // // int update_edges[4] = {1, 8, 1, 6};   // Packed as: [source, destination]
        // // int num_updates = 2;            // Only one edge update in this batch
        // // bool isAddition = true;         // Indicate that this is an edge addition
        // printf("updates\n");
        // updateBFSUpdates(&custing2, levels_del, d_frontier_del, update_edges_del,num_updates, isAddition,1);
        // printf("run bfs\n");
        // int* h_levels_rev = (int*)malloc(nv * sizeof(int));
        // for (int i = 0; i < nv; i++) {
        //     h_levels_rev[i] = -1;  // -1 indicates undiscovered
        // }
        // h_levels_rev[7] = 0;
        // int* d_frontier_rev;
        // CHECK_CUDA(cudaMalloc((void**)&d_frontier_rev, nv * sizeof(vertexId_t)));
        // int * levels_rev=runBFS(&custing2, 7 ,h_levels_rev,d_frontier_rev);
        custing2.freecuStinger();

    
    
    free(off);
    free(adj);
    return 0;	
}
