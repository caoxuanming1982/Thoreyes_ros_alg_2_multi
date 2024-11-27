#include "moniter/moniter.h"
#include<chrono>
#include <unistd.h>
#include <thread>

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
Host_dev_stat Moniter::get_host_stat(){
    if(inited==false){
        init();
//        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }


    Host_dev_stat result;
    CPU_OCCUPY temp;
    get_cpuoccupy(temp);
    result.cpu_util=cal_cpuoccupy(this->last_cpu_stat,temp)*cpu_num;
    this->last_cpu_stat=temp;
    MEM_OCCUPY mem_info;
    get_memoccupy(mem_info);
    result.mem_total=(float)mem_info.MemTotal/1024;
    result.mem_used=result.mem_total-(float)(mem_info.MemAvailable)/1024;
    return result;
};

void Moniter::init(){
    cpu_num=sysconf(_SC_NPROCESSORS_ONLN);
    get_cpuoccupy(this->last_cpu_stat);
    inited=true;
};
