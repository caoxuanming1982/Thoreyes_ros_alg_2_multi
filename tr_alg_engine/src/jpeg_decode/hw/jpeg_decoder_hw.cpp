#include "jpeg_decode/hw/jpeg_decoder_hw.h"

Jpeg_Decoder_hw::Jpeg_Decoder_hw(int max_wait, int max_decoder) : Jpeg_Decoder(max_wait, max_decoder) {

};
Jpeg_Decoder_hw::~Jpeg_Decoder_hw() {

};
void Jpeg_Decoder_hw::init(std::vector<std::shared_ptr<Device_Handle>> devices_handles, int init_cnt)
{
    if(this->devices_handles.size()>0)
        return;
    if(devices_handles.size()<=0)
        return;
    std::unique_lock lock(mutex);
    //std::cout<<"init jped decoder pool start"<<std::endl;
    this->devices_handles=devices_handles;
    this->decoders.resize(max_decoder);
    this->decoder_states.resize(max_decoder);
    for(int i=0;i<init_cnt;i++){        
        std::shared_ptr<Device_Handle> device=devices_handles[i%devices_handles.size()];
   //     std::cout<<"init jped decoder "<<i<<"\t"<<device_id<<std::endl;
        this->decoders[i].init(device);
        
        this->decoder_states[i]=0;
    }
    this->current_cnt=init_cnt;
};
std::shared_ptr<QyImage> Jpeg_Decoder_hw::decode(const std::vector<unsigned char> &data)
{
    std::shared_ptr<QyImage> result;
    if (current_cnt <= 0)
        return result;
    long start_time = get_time();
    int select_idx = -1;

    while (true)
    {

        select_idx = get_instance_idx();
        if (select_idx >= 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        long current_time = get_time();
        if (current_time - start_time > this->max_wait)
        {
            if (current_cnt < this->decoders.size())
            {
                select_idx = add_instance(nullptr);
                break;
            }
        }
    }
    if (select_idx >= 0)
    {
        bool res = decoders[select_idx].decode(data,result);
        //        std::cout<<"decoce jpeg image by "<<select_idx<<"\t"<<res<<" width:"<<dst.cols()<<" height:"<<dst.rows()<<std::endl;
        std::unique_lock lock(mutex);
        decoder_states[select_idx] = 0;
        return result;
    }
    return result;

};

int Jpeg_Decoder_hw::add_instance(std::shared_ptr<Device_Handle> device)
{
    std::unique_lock lock(mutex);
    if (device == nullptr)
        device = devices_handles[current_cnt % devices_handles.size()];
    //  std::cout<<" add instance "<<device_id<<"\t"<<current_cnt<<std::endl;
    decoders[current_cnt].init( device);
    decoder_states[current_cnt] = 1;
    current_cnt += 1;
    return current_cnt - 1;
};
int Jpeg_Decoder_hw::get_instance_idx()
{
    std::unique_lock lock(mutex);
    for (int i = 0; i < current_cnt; i++)
    {
        if (decoder_states[i] == 0)
        {
            decoder_states[i] = 1;
            return i;
        }
    }
    return -1;
};


extern "C" std::shared_ptr<Jpeg_Decoder> get_jpeg_decoder(int max_wait,int max_decoder){
    std::shared_ptr<Jpeg_Decoder_hw> decoder=std::make_shared<Jpeg_Decoder_hw>(max_wait,max_decoder);
    std::shared_ptr<Jpeg_Decoder> res=decoder;
    return res;
};
