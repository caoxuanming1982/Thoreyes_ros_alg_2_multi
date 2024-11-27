#include "tr_alg_engine/tr_device_manager_helper.h"
#include <rclcpp/rclcpp.hpp>
#include "tr_alg_engine/common.h"

void get_memoccupy(MEM_OCCUPY &m) //对无类型get函数含有一个形参结构体类弄的指针O  
{  
    FILE *fd;  
    char buff[256];  
      
    fd = fopen("/proc/meminfo", "r");  
    //MemTotal: 515164 kB  
    //MemFree: 7348 kB  
    //Buffers: 7892 kB  
    //Cached: 241852  kB  
    //SwapCached: 0 kB  
    //从fd文件中读取长度为buff的字符串再存到起始地址为buff这个空间里   
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m.name1, &m.MemTotal);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m.name2, &m.MemFree);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m.name3, &m.MemAvailable);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m.name4, &m.Buffers);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m.name5, &m.Cached);  
    fgets(buff, sizeof(buff), fd);   
    sscanf(buff, "%s %lu", m.name6, &m.SwapCached);  
      
    fclose(fd);     //关闭文件fd  
}  ;
  
  
int get_cpuoccupy(CPU_OCCUPY &cpu_occupy) //对无类型get函数含有一个形参结构体类弄的指针O  
{  
    FILE *fd;  
    char buff[256];  
      
    fd = fopen("/proc/stat", "r");  
    fgets(buff, sizeof(buff), fd);  
      
    sscanf(buff, "%s %u %u %u %u %u %u %u", cpu_occupy.name, &cpu_occupy.user, &cpu_occupy.nice, 
        &cpu_occupy.system, &cpu_occupy.idle, &cpu_occupy.lowait, &cpu_occupy.irq, &cpu_occupy.softirq);  
      
      
    fclose(fd);  
      
    return 0;  
}  ;


float  cal_cpuoccupy(CPU_OCCUPY &o,CPU_OCCUPY &n)  
{  
    unsigned long od,nd;  
    float cpu_use = 0;  
    
    od = (unsigned long)(o.user + o.nice + o.system + o.idle + o.lowait + o.irq + o.softirq);//第一次(用户+优先级+系统+空闲)的时间再赋给od  
    nd = (unsigned long)(n.user + n.nice + n.system + n.idle + n.lowait + n.irq + n.softirq);//第一次(用户+优先级+系统+空闲)的时间再赋给od  
    
    float sum = nd - od;  
    float idle = n.idle - o.idle;  
    cpu_use = idle / sum;  
//    idle = n.user + n.system + n.nice - o.user - o.system - o.nice;  
 //   cpu_use = idle / sum; 
    return cpu_use; 
} ;
#ifdef USE_BM
void get_tpu_occupy(std::vector<bm_handle_t> device_handle,std::vector<bm_dev_stat_t>& res){
    res.resize(device_handle.size());
    for(int i=0;i<device_handle.size();i++){
        bm_get_stat(device_handle[i],&res[i]);
    }
};

#else
gpu_dev_stat get_tpu_occupy_single(torch::Device device){
    nvmlInit();
    gpu_dev_stat res;
        cudaDeviceProp deviceProp;
        int idx=device.index();
        cudaGetDeviceProperties(&deviceProp, idx);
        size_t freeMemory, totalMemory;
        cudaSetDevice(idx);
        cudaMemGetInfo(&freeMemory, &totalMemory);
        nvmlDevice_t device1;
        nvmlDeviceGetHandleByIndex(idx,&device1);
        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device1,&util);
        res.mem_total=totalMemory/1024/1024;
        res.mem_used=totalMemory/1024/1024-freeMemory/1024/1024;
        res.tpu_util=util.gpu;
//        std::cout<<idx<<"\t"<<res.mem_total<<"\t"<<res.mem_used<<"\t"<<res.tpu_util<<std::endl;
    return res;
};


void get_tpu_occupy(std::vector<torch::Device> device_handle,std::vector<gpu_dev_stat>& res){
    res.resize(device_handle.size());
    for(int i=0;i<device_handle.size();i++){
        res[i]=get_tpu_occupy_single(device_handle[i]);
    }
};

#endif