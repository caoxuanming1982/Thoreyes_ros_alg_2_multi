#include <cstdio>
#include "tr_alg_engine/tr_alg_engine.h"
#include "tr_alg_engine/tr_check_channel.h"
#include "tr_alg_engine/tr_device_manager.h"
#include <signal.h>
#include <execinfo.h>

#define SIZE 1000
void *buffer[SIZE];
void fault_trap(int n,siginfo_t *siginfo,void *myact)
{
        int i, num;
        char **calls;
        printf("Fault address:%X\n",siginfo->si_addr);   
        num = backtrace(buffer, SIZE);
        calls = backtrace_symbols(buffer, num);
        for (i = 0; i < num; i++)
                printf("%s\n", calls[i]);
        exit(1);
}
void setuptrap()
{
    struct sigaction act;
        sigemptyset(&act.sa_mask);   
        act.sa_flags=SA_SIGINFO;    
        act.sa_sigaction=fault_trap;
        sigaction(SIGSEGV,&act,NULL);
}

int main(int argc, char ** argv)
{

  setuptrap();
  (void) argc;
  (void) argv;
  rclcpp::init(argc, argv);
  int n_thread=16;
  if(argc>1){
    n_thread=atoi(argv[1]);
    if(n_thread<=0)
      n_thread=16;
  }

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),n_thread);
  auto node1=get_engine_node();

  if(argc>2){
    if(atoi(argv[2])>0)
      node1->show_util=true;
  }
  if(argc>3){
    if(argc==4){
      if(node1->init(argv[3])==false)
        return -1;
    }
    else if (argc>5){
      if(node1->init(argv[3],argv[4])==false)
        return -1;
    }
    else{
      if(node1->init()==false)
        return -1;

    }
  }
  else{
    if(node1->init()==false)
      return -1;

  }


  executor.add_node(node1);

  auto node2=get_check_channel_node();
  executor.add_node(node2);

  auto node3=get_device_manager_node();
  executor.add_node(node3);

  executor.spin();

  printf("hello world tr_alg_common package\n");
  rclcpp::shutdown();
  return 0;
}
