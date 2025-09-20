/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A shader that adds two arrays of floats.
*/

#include <metal_stdlib>
using namespace metal;
/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.
kernel void EwiseAddKernel(device const float* inA,
                           device const float* inB,
                           device float* result,
                           uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}

kernel void EwisePowerKernel(device const float* inA,
                             device const float* inB,
                             device float* result,
                             uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = pow(inA[index], inB[index]);
}

kernel void EwiseMaximumKernel(device const float* inA,
                               device const float* inB,
                               device float* result,
                               uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = max(inA[index], inB[index]);
}

kernel void EwiseEqKernel(device const float* inA,
                          device const float* inB,
                          device float* result,
                          uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index]==inB[index]?1:0;
}

kernel void EwiseGeKernel(device const float* inA,
                          device const float* inB,
                          device float* result,
                          uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index]>=inB[index]?1:0;
}

kernel void EwiseLogKernel(device const float* inA,
                           device float* result,
                           uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = log(inA[index]);
}

kernel void EwiseExpKernel(device const float* inA,
                           device float* result,
                           uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = exp(inA[index]);
}

kernel void EwiseTanhKernel(device const float* inA,
                           device float* result,
                           uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = tanh(inA[index]);
}

kernel void EwiseMulKernel(device const float* inA,
                           device const float* inB,
                           device float* result,
                           uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] * inB[index];
}

kernel void EwiseDivKernel(device const float* inA,
                           device const float* inB,
                           device float* result,
                           uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] / inB[index];
}

kernel void CompactKernel(device const float* a,
                          device float* out,
                          device uint* shape,
                          device uint* strides,
                          device uint* info,
                          uint index [[thread_position_in_grid]])
{
    // info: {size, num_dim, offset}
    uint size = info[0];
    uint numDim = info[1];
    uint offset = info[2];
    
    size_t mapped_idx = offset;
    size_t idx_trace = index;
    for (int axis=numDim-1; axis>=0; axis--){
        mapped_idx += (idx_trace%shape[axis])*strides[axis];
        idx_trace /= shape[axis];
    }
    out[index] = a[mapped_idx];
}

kernel void EwiseSetitemKernel(device const float* a,
                               device float* out,
                               device uint* shape,
                               device uint* strides,
                               device uint* info,
                               uint index [[thread_position_in_grid]])
{
    // info: {size, num_dim, offset}
    uint size = info[0];
    uint num_dim = info[1];
    uint offset = info[2];

    size_t mapped_idx = offset;
    size_t idx_trace = index;
    for (unsigned long axis=num_dim-1; axis>=0; axis--){
        mapped_idx += (idx_trace%shape[axis])*strides[axis];
        idx_trace /= shape[axis];
    }
    out[mapped_idx] = a[index];
}

#define TILE_SIZE 16
kernel void MatmulKernel(device const float* a,
                                        device const float* b,
                                        device float* out,
                                        device uint* dimensions,
                                        uint2 gid [[ thread_position_in_grid ]],
                                        uint2 tid [[ thread_position_in_threadgroup ]])
{
    threadgroup float ATile[TILE_SIZE][TILE_SIZE];
    threadgroup float BTile[TILE_SIZE][TILE_SIZE];

    // M: rows of a, N: rows of b / cols of a, P: cols of b
    // info = {M, N, P}
    uint M = dimensions[0];
    uint N = dimensions[1];
    uint P = dimensions[2];
    
    // get the thread global position <=> the output index
    uint out_row = gid.x;
    uint out_col = gid.y;

    // get the thread group position <=> the tile index
    uint thread_row = tid.x;
    uint thread_col = tid.y;

    float result = 0.0f;
    
    // make sure it does not go out of bounds
    for (uint tile_idx=0; tile_idx< (TILE_SIZE + N - 1) / TILE_SIZE; tile_idx++){
        // loading from the corresponding location, 0 if out of bounds
        ATile[thread_row][thread_col] =
            (out_row < M) && (tile_idx*TILE_SIZE+thread_col<N) ? 
            a[out_row*N+(tile_idx*TILE_SIZE+thread_col)] : 0;
        BTile[thread_row][thread_col] =
            (out_col < P) && (tile_idx*TILE_SIZE+thread_row<N) ?
            b[(thread_row+tile_idx*TILE_SIZE)*P+out_col] : 0;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        
        // doing matrix multiplication within a tile
        for (size_t k=0; k<TILE_SIZE; k++){
            result += ATile[thread_row][k] * BTile[k][thread_col];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    if (out_row < M && out_col < P){
        out[out_row*P+out_col] = result;
    }
}

kernel void ReduceMaxKernel(device const float* a,
                            device float* out,
                            device uint* info,
                            uint index [[thread_position_in_grid]])
{
    // info = {n, reduce_size}
    uint n = info[0];
    uint reduce_size = info[1];
    uint start_idx = index * reduce_size;
    float max = 0;
    for (size_t offset=0; offset<reduce_size; offset++){
        float curr = a[start_idx+offset];
        max = max > curr ? max: curr;
    }
    out[index] = max;
}

kernel void ReduceSumKernel(device const float* a,
                            device float* out,
                            device uint* info,
                            uint index [[thread_position_in_grid]])
{
    // info = {n, reduce_size}
    uint n = info[0];
    uint reduce_size = info[1];
    uint start_idx = index * reduce_size;
    float sum = 0;
    for (uint offset=0; offset<reduce_size; offset++){
        sum += a[start_idx+offset];
    }
    out[index] = sum;
}
