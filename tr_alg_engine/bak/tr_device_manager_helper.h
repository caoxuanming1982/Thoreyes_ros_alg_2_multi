#ifndef __TR_DEVICE_MANAGER_HELPER_H__
#define __TR_DEVICE_MANAGER_HELPER_H__

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <iostream>   
#include <unistd.h>

#include<vector>
#ifdef USE_BM
#include<bmruntime.h>
#include<bmruntime_cpp.h>
#include<bmcv_api.h>
#else
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <nvml.h>
class gpu_dev_stat;
#endif
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <rclcpp/rclcpp.hpp>
#include "tr_alg_engine/common.h"




typedef struct MEMPACKED         //定义一个mem occupy的结构体  
{  
    char name1[20];      //定义一个char类型的数组名name有20个元素  
    unsigned long MemTotal;  
    char name2[20];  
    unsigned long MemFree;  
    char name3[20];  
    unsigned long MemAvailable;  
    char name4[20];  
    unsigned long Buffers;  
    char name5[20];  
    unsigned long Cached;  
    char name6[20];  
    unsigned long SwapCached;  
}MEM_OCCUPY;  
    
typedef struct CPUPACKED         //定义一个cpu occupy的结构体  
{  
    char name[20];      //定义一个char类型的数组名name有20个元素  
    unsigned int user; //定义一个无符号的int类型的user  
    unsigned int nice; //定义一个无符号的int类型的nice  
    unsigned int system;//定义一个无符号的int类型的system  
    unsigned int idle; //定义一个无符号的int类型的idle  
    unsigned int lowait;  
    unsigned int irq;  
    unsigned int softirq;  
}CPU_OCCUPY;



void get_memoccupy(MEM_OCCUPY &m) ;
  
int get_cpuoccupy(CPU_OCCUPY &cpu_occupy) ;

float  cal_cpuoccupy(CPU_OCCUPY &o,CPU_OCCUPY &n);

#ifdef USE_BM
void get_tpu_occupy(std::vector<bm_handle_t> device_handle,std::vector<bm_dev_stat_t>& res);
#else
void get_tpu_occupy(std::vector<torch::Device> device_handle,std::vector<gpu_dev_stat>& res);
gpu_dev_stat get_tpu_occupy_single(torch::Device);

#endif
#endif