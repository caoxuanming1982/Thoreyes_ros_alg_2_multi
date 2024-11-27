#ifndef __NETWORK_PARALLEL_H__
#define __NETWORK_PARALLEL_H__

#include<vector>
#include<map>
#include<queue>
#include<shared_mutex>
#include<chrono>
#include <shared_mutex>

#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>


#include<unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>


#include "common.h"
#include "network_engine/device_handle.h"
#include "network_engine/network_kernel.h"


class Unit_counter{
    std::shared_mutex mutex;
    long last_start_time=0;
    std::vector<long> time_intervals;
    std::vector<long> busy_time_intervals;
    int max_counter_cnt=10;

public:
    Unit_counter();
    void tick_start_time(long start_time);
    void tick_busy_time(long busy_time);
    void reset();
    float get_util();

};

class Network_instance{

public:

    std::shared_ptr<Device_Handle> handle;
    std::vector<Shape_t> cache_inputs_shapes;


    std::string file_name;
    std::string model_name;

    Network_kernel* instance=nullptr;
    Unit_counter util_counter;
    std::shared_mutex mutex;

public:

    Network_instance(std::shared_ptr<Device_Handle> handle,std::string file_name,std::string model_name,std::vector<Shape_t>& input_shapes);
    std::shared_ptr<Device_Handle> get_handle();
    Network_instance* copy(std::shared_ptr<Device_Handle> handle);
    std::vector<Shape_t> get_input_shapes();
    void set_input_shapes(std::vector<Shape_t>& shapes);
    void set_input_scale_offset(std::vector<cv::Scalar> scale,std::vector<cv::Scalar> offset);
    int forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs);
    int forward(std::vector<cv::Mat>& inputs,std::vector<Output>& outputs);

    bool reload_model(std::string file_name,std::string model_name);
    bool load_model();
    bool free_model();
    ~Network_instance();
    float get_util();

};

class Network_parallel{
public:
    std::map<int,std::shared_ptr<Device_Handle>> device_handles;
    std::vector<Shape_t> cache_inputs_shapes;

    std::pair<int,int>temp;
    std::vector<std::shared_ptr<Network_instance>> instances;
    std::queue<std::shared_ptr<Network_instance>> processor_queue;

    std::string file_name;
    std::string model_name;

    std::shared_mutex mutex;
    int max_instance_cnt=6;
    
    std::vector<cv::Scalar> input_scale;
    std::vector<cv::Scalar> input_offset;

private:
    std::shared_ptr<Network_instance> get_instance(int card_id=-1);


public:


    Network_parallel();
    void set_device_ids(std::vector<int> device_id);
    void set_input_scale_offset(std::map<int,cv::Scalar> scale,std::map<int,cv::Scalar> offset);

    void set_device_handles(std::map<int,std::shared_ptr<Device_Handle>> device_handles);
    int forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs);
    void load_model(std::string file_name,std::string model_name);
    void set_input_shapes(std::vector<Shape_t>& shapes);
    std::vector<Shape_t> get_input_shapes();
    int forward(std::vector<cv::Mat>& inputs,std::vector<Output>& outputs);

    bool add_inference_instance(int device_id);
    bool remove_last_inference_instance();


    float check_util();
    std::vector<float> get_util();
};


#endif