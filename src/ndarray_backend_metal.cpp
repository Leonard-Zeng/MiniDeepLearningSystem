#include <cassert>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <simd/simd.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
    namespace metal {

#define ALIGNMENT 256
#define TILE 8
    typedef unsigned char byte;
        typedef float scalar_t;
        const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
        struct MetalArray {
            MetalArray(const size_t size) {
                // transfer this array to Metal device
                ptr = MTL::CreateSystemDefaultDevice()->newBuffer(size*ELEM_SIZE, MTL::ResourceStorageModeShared);
                this->size = size;
            }
            ~MetalArray() { ptr->release(); }
            size_t ptr_as_int() {return (size_t)ptr; }
            MTL::Buffer* ptr;
            size_t size;
        };

        void Fill(MetalArray* out, scalar_t val) {
            /**
             * Fill the values of an aligned array with val
             */
             scalar_t * ptr = (scalar_t *)out->ptr->contents();
             size_t size = out->size;
             std::fill(ptr, ptr+size, val);
        }




        void Compact(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
                     std::vector<int32_t> strides, size_t offset) {
            /**
             * Compact an array in memory
             *
             * Args:
             *   a: non-compact representation of the array, given as input
             *   out: compact version of the array to be written
             *   shape: shapes of each dimension for a and out
             *   strides: strides of the *a* array (not out, which has compact strides)
             *   offset: offset of the *a* array (not out, which has zero offset, being compact)
             *
             * Returns:
             *  void (you need to modify out directly, rather than returning anything; this is true for all the
             *  function will implement here, so we won't repeat this note.)
             */
            /// BEGIN YOUR SOLUTION
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
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

            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("CompactKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            uint32_t size = (uint32_t)a.size;
            uint32_t numDim = (uint32_t)shape.size();
            uint32_t info[] = {size, numDim, (uint32_t)offset};
            
            MTL::Buffer* shapeBuffer = device->newBuffer(shape.data(), numDim*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            MTL::Buffer* stridesBuffer = device->newBuffer(strides.data(), numDim*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            MTL::Buffer* infoBuffer = device->newBuffer(info, 3*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            commandEncoder->setBuffer(shapeBuffer, 0, 2);
            commandEncoder->setBuffer(stridesBuffer, 0, 3);
            commandEncoder->setBuffer(infoBuffer, 0, 4);
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            /// END YOUR SOLUTION
        }

        void EwiseSetitem(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
                          std::vector<int32_t> strides, size_t offset) {
            /**
             * Set items in a (non-compact) array
             *
             * Args:
             *   a: _compact_ array whose items will be written to out
             *   out: non-compact array whose items are to be written
             *   shape: shapes of each dimension for a and out
             *   strides: strides of the *out* array (not a, which has compact strides)
             *   offset: offset of the *out* array (not a, which has zero offset, being compact)
             */
            /// BEGIN YOUR SOLUTION
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseSetitemKernel(device const float* a,
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
                    out[mapped_idx] = a[index];
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseSetitemKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            uint32_t size = (uint32_t)a.size;
            uint32_t numDim = (uint32_t)shape.size();
            uint32_t info[] = {size, numDim, (uint32_t)offset};
            
            MTL::Buffer* shapeBuffer = device->newBuffer((uint32_t*)shape.data(), numDim*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            MTL::Buffer* stridesBuffer = device->newBuffer((uint32_t*)strides.data(), numDim*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            MTL::Buffer* infoBuffer = device->newBuffer((uint32_t*)info, 3*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            commandEncoder->setBuffer(shapeBuffer, 0, 2);
            commandEncoder->setBuffer(stridesBuffer, 0, 3);
            commandEncoder->setBuffer(infoBuffer, 0, 4);
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            /// END YOUR SOLUTION
        }

        void ScalarSetitem(const size_t size, scalar_t val, MetalArray* out, std::vector<int32_t> shape,
                           std::vector<int32_t> strides, size_t offset) {
            /**
             * Set items is a (non-compact) array
             *
             * Args:
             *   size: number of elements to write in out array (note that this will note be the same as
             *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
             *         product of items in shape, but convenient to just pass it here.
             *   val: scalar value to write to
             *   out: non-compact array whose items are to be written
             *   shape: shapes of each dimension of out
             *   strides: strides of the out array
             *   offset: offset of the out array
             */

            /// BEGIN YOUR SOLUTION
            MetalArray* a = new MetalArray(out->size);
            Fill(a, val);
            EwiseSetitem(*a, out, shape, strides, offset);
            delete a;
            /// END YOUR SOLUTION
        }

        void EwiseAdd(const MetalArray& a, const MetalArray& b, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseAddKernel(device const float* inA,
                                           device const float* inB,
                                           device float* result,
                                           uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = inA[index] + inB[index];
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseAddKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void ScalarAdd(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwiseAdd(a, *b, out);
            delete b;
        }


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
        void EwiseMul(const MetalArray& a, const MetalArray& b, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseMulKernel(device const float* inA,
                                           device const float* inB,
                                           device float* result,
                                           uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = inA[index] * inB[index];
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseMulKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void ScalarMul(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwiseMul(a, *b, out);
            delete b;
        }

        void EwiseDiv(const MetalArray& a, const MetalArray& b, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseDivKernel(device const float* inA,
                                           device const float* inB,
                                           device float* result,
                                           uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = inA[index] / inB[index];
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseDivKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void ScalarDiv(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwiseDiv(a, *b, out);
            delete b;
        }
    
        void EwisePower(const MetalArray& a, const MetalArray& b, MetalArray* out) {
        /**
         * Set entries in out to be the sum of correspondings entires in a and b.
         */
        using NS::StringEncoding::UTF8StringEncoding;
        const char* shaderSrc = R"(
            #include <metal_stdlib>
            using namespace metal;
            kernel void EwisePowerKernel(device const float* inA,
                                         device const float* inB,
                                         device float* result,
                                         uint index [[thread_position_in_grid]])
            {
                // the for-loop is replaced with a collection of threads, each of which
                // calls this function.
                result[index] = pow(inA[index], inB[index]);
            }
        )";
        
        
        MTL::Device * device = MTL::CreateSystemDefaultDevice();
        NS::Error* pError = nullptr;
        MTL::Library* gpuFunctionLibrary = device->newLibrary(
                          NS::String::string(shaderSrc, UTF8StringEncoding),
                          nullptr,
                          &pError
        );
        
        MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwisePowerKernel"));
        if (!kernelFunc){
            std::cout<<"No Kernel Function Found"<<std::endl;
            if (gpuFunctionLibrary->functionNames()->count()>0){
                for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                    std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                }
            } else{
                std::cout<<"No Kernel Function Found in Library"<<std::endl;
            }
        }
        NS::Error * stateErr = nullptr;
        MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
        
        MTL::CommandQueue* commandQueue = device->newCommandQueue();
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
        commandEncoder->setComputePipelineState(computePiplineState);
        
        commandEncoder->setBuffer(a.ptr, 0, 0);
        commandEncoder->setBuffer(b.ptr, 0, 1);
        commandEncoder->setBuffer(out->ptr, 0, 2);
        
        size_t size = a.size;
        
        MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
        NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
        MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
        commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
        
        commandEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

        void ScalarPower(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwisePower(a, *b, out);
            delete b;
        }

        void EwiseMaximum(const MetalArray& a, const MetalArray& b, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseMaximumKernel(device const float* inA,
                                               device const float* inB,
                                               device float* result,
                                               uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = max(inA[index], inB[index]);
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseMaximumKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void ScalarMaximum(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwiseMaximum(a, *b, out);
            delete b;
        }

        void EwiseEq(const MetalArray& a, const MetalArray& b, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseEqKernel(device const float* inA,
                                          device const float* inB,
                                          device float* result,
                                          uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = inA[index]==inB[index]?1:0;
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseEqKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void ScalarEq(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwiseEq(a, *b, out);
            delete b;
        }

        void EwiseGe(const MetalArray& a, const MetalArray& b, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseGeKernel(device const float* inA,
                                          device const float* inB,
                                          device float* result,
                                          uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = inA[index]>=inB[index]?1:0;
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseGeKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void ScalarGe(const MetalArray& a, scalar_t val, MetalArray* out) {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            MetalArray* b = new MetalArray(a.size);
            Fill(b, val);
            EwiseGe(a, *b, out);
            delete b;
        }

        void EwiseLog(const MetalArray& a, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseLogKernel(device const float* inA,
                                           device float* result,
                                           uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = log(inA[index]);
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseLogKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void EwiseExp(const MetalArray& a, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseExpKernel(device const float* inA,
                                           device float* result,
                                           uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = exp(inA[index]);
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseExpKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }

        void EwiseTanh(const MetalArray& a, MetalArray* out) {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void EwiseTanhKernel(device const float* inA,
                                           device float* result,
                                           uint index [[thread_position_in_grid]])
                {
                    // the for-loop is replaced with a collection of threads, each of which
                    // calls this function.
                    result[index] = tanh(inA[index]);
                }
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("EwiseTanhKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            
            size_t size = a.size;
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }
/// END YOUR SOLUTION
#define TILE_SIZE 16
        void Matmul(const MetalArray& a, const MetalArray& b, MetalArray* out, uint32_t m, uint32_t n,
                    uint32_t p) {
            /**
             * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
             * you can use the "naive" three-loop algorithm.
             *
             * Args:
             *   a: compact 2D array of size m x n
             *   b: compact 2D array of size n x p
             *   out: compact 2D array of size m x p to write the output to
             *   m: rows of a / out
             *   n: columns of a / rows of b
             *   p: columns of b / out
             */

            /// BEGIN YOUR SOLUTION
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #define TILE_SIZE 16
                kernel void MatmulKernel(device const float* a,
                                        device const float* b,
                                        device float* out,
                                        device uint32_t* dimensions,
                                        uint2 gid [[ thread_position_in_grid ]],
                                        uint2 tid [[ thread_position_in_threadgroup ]])
                // {
                //     // Shared memory for the input matrices tiles
                //     threadgroup float aTile[TILE_SIZE][TILE_SIZE];
                //     threadgroup float bTile[TILE_SIZE][TILE_SIZE];

                //     // Calculate the row and column index for this thread
                //     uint row = gid.x;
                //     uint col = gid.y;

                //     // Initialize the result for this thread
                //     float result = 0.0f;

                //     // Loop over the tiles of the input matrices
                //     for (uint t = 0; t < (TILE_SIZE + dimensions[1] - 1)/TILE_SIZE; t++) {

                //         // Load the tiles of the input matrices into shared memory
                //         if (row < dimensions[0] && t*TILE_SIZE + tid.y < dimensions[1]) {
                //             aTile[tid.x][tid.y] = a[row*dimensions[1] + t*TILE_SIZE + tid.y];
                //         } else {
                //             aTile[tid.x][tid.y] = 0.0f;
                //         }

                //         if (t*TILE_SIZE + tid.x < dimensions[1] && col < dimensions[2]) {
                //             bTile[tid.x][tid.y] = b[(t*TILE_SIZE + tid.x)*dimensions[2] + col];
                //         } else {
                //             bTile[tid.x][tid.y] = 0.0f;
                //         }

                //         // Synchronize to make sure the tiles are loaded
                //         threadgroup_barrier(metal::mem_flags::mem_threadgroup);

                //         // Perform the matrix multiplication on the tiles
                //         for (uint i = 0; i < TILE_SIZE; i++) {
                //             result += aTile[tid.x][i] * bTile[i][tid.y];
                //         }

                //         // Synchronize to make sure the computation is done before loading the next tile
                //         threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                //     }

                //     // Write the result to the output matrix
                //     if (row < dimensions[0] && col < dimensions[2]) {
                //         out[row*dimensions[2] + col] = result;
                //     }
                // }
                {
                    threadgroup float ATile[TILE_SIZE][TILE_SIZE];
                    threadgroup float BTile[TILE_SIZE][TILE_SIZE];

                    // M: rows of a, N: rows of b / cols of a, P: cols of b
                    // info = {M, N, P}
                    uint M = dimensions[0];
                    uint N = dimensions[1];
                    uint P = dimensions[2];
                    
                    // get the thread global position <=> the output location
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
            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            if (pError)
                std::cout<<pError->description()->utf8String()<<std::endl;
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("MatmulKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()<=0){
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            uint32_t info[] = {m, n, p};
            MTL::Buffer* infoBuffer = device->newBuffer(
                (uint32_t*)info, 3*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(b.ptr, 0, 1);
            commandEncoder->setBuffer(out->ptr, 0, 2);
            commandEncoder->setBuffer(infoBuffer, 0, 3);
            
            MTL::Size threadsPerGrid = MTL::Size((m + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE, (p + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE, 1);
            MTL::Size threadsPerThreadgroup = MTL::Size(TILE_SIZE, TILE_SIZE, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            /// END YOUR SOLUTION
        }

        void ReduceMax(const MetalArray& a, MetalArray* out, size_t reduce_size) {
            /**
             * Reduce by taking maximum over `reduce_size` contiguous blocks.
             *
             * Args:
             *   a: compact array of size a.size = out.size * reduce_size to reduce over
             *   out: compact array to write into
             *   reduce_size: size of the dimension to reduce over
             */

            /// BEGIN YOUR SOLUTION
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
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

            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("ReduceMaxKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            uint32_t size = (uint32_t)a.size;
            uint32_t info[] = {size, (uint32_t)reduce_size};
            
            MTL::Buffer* infoBuffer = device->newBuffer(info, 3*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            commandEncoder->setBuffer(infoBuffer, 0, 2);
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            /// END YOUR SOLUTION
        }

        void ReduceSum(const MetalArray& a, MetalArray* out, size_t reduce_size) {
            /**
             * Reduce by taking sum over `reduce_size` contiguous blocks.
             *
             * Args:
             *   a: compact array of size a.size = out.size * reduce_size to reduce over
             *   out: compact array to write into
             *   reduce_size: size of the dimension to reduce over
             */

            /// BEGIN YOUR SOLUTION
            using NS::StringEncoding::UTF8StringEncoding;
            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;
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

            )";
            
            
            MTL::Device * device = MTL::CreateSystemDefaultDevice();
            NS::Error* pError = nullptr;
            MTL::Library* gpuFunctionLibrary = device->newLibrary(
                              NS::String::string(shaderSrc, UTF8StringEncoding),
                              nullptr,
                              &pError
            );
            
            MTL::Function* kernelFunc = gpuFunctionLibrary->newFunction(NS::MakeConstantString("ReduceSumKernel"));
            if (!kernelFunc){
                std::cout<<"No Kernel Function Found"<<std::endl;
                if (gpuFunctionLibrary->functionNames()->count()>0){
                    for (uint i=0; i<gpuFunctionLibrary->functionNames()->count(); i++){
                        std::cout<<gpuFunctionLibrary->functionNames()->object(i)<<std::endl;
                    }
                } else{
                    std::cout<<"No Kernel Function Found in Library"<<std::endl;
                }
            }
            NS::Error * stateErr = nullptr;
            MTL::ComputePipelineState* computePiplineState = device->newComputePipelineState(kernelFunc, &stateErr);
            
            MTL::CommandQueue* commandQueue = device->newCommandQueue();
            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
            commandEncoder->setComputePipelineState(computePiplineState);
            
            uint32_t size = (uint32_t)a.size;
            uint32_t info[] = {size, (uint32_t)reduce_size};
            
            MTL::Buffer* infoBuffer = device->newBuffer(info, 3*sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            commandEncoder->setBuffer(a.ptr, 0, 0);
            commandEncoder->setBuffer(out->ptr, 0, 1);
            commandEncoder->setBuffer(infoBuffer, 0, 2);
            
            MTL::Size threadsPerGrid = MTL::Size(a.size, 1, 1);
            NS::UInteger maxThreadsPerThreadgroup = computePiplineState->maxTotalThreadsPerThreadgroup();
            MTL::Size threadsPerThreadgroup = MTL::Size(maxThreadsPerThreadgroup, 1, 1);
            commandEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
            
            commandEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            /// END YOUR SOLUTION
        }

    }  // namespace metal
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_metal, m) {
namespace py = pybind11;
using namespace needle;
using namespace metal;

m.attr("__device_name__") = "metal";
m.attr("__tile_size__") = TILE;

py::class_<MetalArray>(m, "Array")
.def(py::init<size_t>(), py::return_value_policy::take_ownership)
.def("ptr", &MetalArray::ptr_as_int)
.def_readonly("size", &MetalArray::size);

// return numpy array (with copying for simplicity, otherwise garbage
// collection is a pain)
m.def("to_numpy", [](const MetalArray& a, std::vector<size_t> shape,
                     std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, (scalar_t *)a.ptr->contents() + offset);
});

// convert from numpy (with copying)
m.def("from_numpy", [](py::array_t<scalar_t> a, MetalArray* out) {
    std::memcpy(out->ptr->contents(), a.request().ptr, out->size * ELEM_SIZE);
});

m.def("fill", Fill);
m.def("compact", Compact);
m.def("ewise_setitem", EwiseSetitem);
m.def("scalar_setitem", ScalarSetitem);
m.def("ewise_add", EwiseAdd);
m.def("scalar_add", ScalarAdd);

m.def("ewise_mul", EwiseMul);
m.def("scalar_mul", ScalarMul);
m.def("ewise_div", EwiseDiv);
m.def("scalar_div", ScalarDiv);
m.def("scalar_power", ScalarPower);

m.def("ewise_maximum", EwiseMaximum);
m.def("scalar_maximum", ScalarMaximum);
m.def("ewise_eq", EwiseEq);
m.def("scalar_eq", ScalarEq);
m.def("ewise_ge", EwiseGe);
m.def("scalar_ge", ScalarGe);

m.def("ewise_log", EwiseLog);
m.def("ewise_exp", EwiseExp);
m.def("ewise_tanh", EwiseTanh);

m.def("matmul", Matmul);

m.def("reduce_max", ReduceMax);
m.def("reduce_sum", ReduceSum);
}
