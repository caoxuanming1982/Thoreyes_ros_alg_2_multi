#ifndef __DEVICE_MONITER_H__
#define __DEVICE_MONITER_H__

#include <network_engine/device_handle.h>
#include <shared_mutex>
#include <iostream>
#include <vector>
#include <cv_lib/type_def.h>


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


class Device_dev_stat {
public:
  int mem_total;
  int mem_used;
  int tpu_util;
};

class Host_dev_stat {
public:
  int mem_total;
  int mem_used;
  int cpu_util;
};

class Moniter{
    CPU_OCCUPY last_cpu_stat;
    float cpu_num=0;
    bool inited=false;
public:
    Moniter(){};
    virtual ~Moniter(){};
    virtual Device_dev_stat get_device_stat(std::shared_ptr<Device_Handle> device)=0;
    virtual Host_dev_stat get_host_stat();
    virtual void init();
};


extern "C" std::shared_ptr<Moniter> get_device_stat();
extern "C" int get_device_count();
#endif