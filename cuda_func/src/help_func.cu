#include "help_func.cuh"
#include <iostream>
#include <stdint.h>
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName((cudaError_t)result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
__host__ __device__ T div_round_up(T val,T divisor){
    return (val+divisor-1)/divisor;
}


__global__ void merge_kernel(
   unsigned char* dst,int dst_channel,int step_dst,unsigned char* src1,unsigned char* src2,unsigned char* src3,int step_src,int width,int height 
){
    uint32_t x=threadIdx.x+blockDim.x*blockIdx.x;
    uint32_t y=threadIdx.y+blockDim.y*blockIdx.y;
    if(x>=width||y>=height)
        return;
    if(src1!=nullptr)
        dst[dst_channel*x+y*step_dst+0]=src1[x+y*step_src];
    if(src2!=nullptr)
        dst[dst_channel*x+y*step_dst+1]=src2[x+y*step_src];
    if(src3!=nullptr)
        dst[dst_channel*x+y*step_dst+2]=src3[x+y*step_src];

}

__global__ void merge_kernel(
   float* dst,int dst_channel,int step_dst,float* src1,float* src2,float* src3,int step_src,int width,int height 
){
    uint32_t x=threadIdx.x+blockDim.x*blockIdx.x;
    uint32_t y=threadIdx.y+blockDim.y*blockIdx.y;
    if(x>=width||y>=height)
        return;
    if(src1!=nullptr)
        dst[dst_channel*x+y*step_dst+0]=src1[x+y*step_src];
    if(src2!=nullptr)
        dst[dst_channel*x+y*step_dst+1]=src2[x+y*step_src];
    if(src3!=nullptr)
        dst[dst_channel*x+y*step_dst+2]=src3[x+y*step_src];

}




void merge_func_8u(unsigned char* dst,int dst_channel,int step_dst,unsigned char* src1,unsigned char* src2,unsigned char* src3,int step_src,int width,int height,cudaStream_t stream){
    const dim3 threads={16,8,1};

    const dim3 blocks={div_round_up<uint32_t>(width,threads.x),div_round_up<uint32_t>(height,threads.y),1};
    merge_kernel<<<blocks,threads,0,stream>>>(dst,dst_channel,step_dst,src1,src2,src3,step_src,width,height);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void merge_func_float(float* dst,int dst_channel,int step_dst,float* src1,float* src2,float* src3,int step_src,int width,int height,cudaStream_t stream){
    const dim3 threads={16,8,1};

    const dim3 blocks={div_round_up<uint32_t>(width,threads.x),div_round_up<uint32_t>(height,threads.y),1};
    merge_kernel<<<blocks,threads,0,stream>>>(dst,dst_channel,step_dst,src1,src2,src3,step_src,width,height);
    checkCudaErrors(cudaStreamSynchronize(stream));
}