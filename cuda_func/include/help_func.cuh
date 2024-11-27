#ifndef __HELP_FUNC_H__
#define __HELP_FUNC_H__

extern "C" void merge_func_8u(unsigned char* dst,int dst_channel,int step_dst,unsigned char* src1,unsigned char* src2,unsigned char* src3,int step_src,int width,int height,cudaStream_t stream);

extern "C" void merge_func_float(float* dst,int dst_channel,int step_dst,float* src1,float* src2,float* src3,int step_src,int width,int height,cudaStream_t stream);


#endif
