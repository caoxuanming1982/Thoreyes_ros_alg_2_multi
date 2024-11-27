#include "alg_module_burst_into_ban_detection.h"
#include <iostream>
#include <fstream>

float iou(Result_item_burst_into_ban_detection& box1, Result_item_burst_into_ban_detection& box2)
{
    float x1 = std::max(box1.x1, box2.x1);      //left
    float y1 = std::max(box1.y1, box2.y1);      //top
    float x2 = std::min((box1.x2), (box2.x2));  //right
    float y2 = std::min((box1.y2), (box2.y2));  //bottom
    if(x1>=x2||y1>=y2)
        return 0;
    float over_area = (x2 - x1) * (y2 - y1);
    float box1_w = box1.x2 - box1.x1;
    float box1_h = box1.y2 - box1.y1;
    float box2_w = box2.x2 - box2.x1;
    float box2_h = box2.y2 - box2.y1;
    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);
    return iou;
};


float iou(Result_item_burst_into_ban_detection* box1, Result_item_burst_into_ban_detection* box2)
{
    float x1 = std::max(box1->x1, box2->x1);        //left
    float y1 = std::max(box1->y1, box2->y1);        //top
    float x2 = std::min((box1->x2), (box2->x2));    //right
    float y2 = std::min((box1->y2), (box2->y2));    //bottom
    if(x1>=x2||y1>=y2)
        return 0;
    float over_area = (x2 - x1) * (y2 - y1);
    float box1_w = box1->x2 - box1->x1;
    float box1_h = box1->y2 - box1->y1;
    float box2_w = box2->x2 - box2->x1;
    float box2_h = box2->y2 - box2->y1;
    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);
    return iou;
};
float cos_distance(std::vector<float> a,std::vector<float> b)
{
    int length=a.size();
    if(length>b.size())
        length=b.size();
    float temp1=0;
    float temp2=0;
    float temp3=0;
    for(int i=0;i<length;i++){
        temp1+=a[i]*b[i];
        temp2+=a[i]*a[i];
        temp3+=b[i]*b[i];
    }
    return temp1/(sqrt(temp2)*sqrt(temp3));

};
void roi_pooling(Output& net_output_feature,std::vector<Result_item_burst_into_ban_detection>& output,int img_h,int img_w)
{
    float* features=(float*)net_output_feature.data.data();

    int f_c=net_output_feature.shape[0];
    int f_h=net_output_feature.shape[1];
    int f_w=net_output_feature.shape[2];
    float factor_h=1.0*f_h/img_h;
    float factor_w=1.0*f_w/img_w;
    int f_size=f_h*f_w;
    for(int i=0;i<output.size();i++)
    {
        bool need_save_worker_conf = false;
        float worker_conf = 0.0;
        if(output[i].feature.size() > 0){
            // cout<<"output[i].feature: "<<output[i].feature.size()<<endl;
            worker_conf = output[i].feature[0];
            need_save_worker_conf = true;
        }

        output[i].feature.clear();
        int x2=int(output[i].x2*factor_w+1);
        int y2=int(output[i].y2*factor_h+1);
        int x1=int(output[i].x1*factor_w);
        int y1=int(output[i].y1*factor_h);
        float sub=(y2-y1)*(x2-x1);
        output[i].feature.resize(f_c);
        if(x1<0)
            x1=0;
        if(y1<0)
            y1=0;
        if(x2-x1<=0)
            x2=x1+1;
        if(y2-y1<=0)
            y2=y1+1;
        if(x2>f_w)
            x2=f_w;
        if(y2>f_h)
            y2=f_h;
        if(x2-x1<=0)
            x1=x2-1;
        if(y2-y1<=0)
            y1=y2-1;

        for(int c=0;c<f_c;c++)
        {
            float val=0;
            int offset_c=c*f_size;
            for(int h=y1;h<y2;h++)
            {
                int offset_h=h*f_w;
                for(int w=x1;w<x2;w++)
                {
                    val+=features[w+offset_h+offset_c];
                }
            }
            if(isinf(val))
            {
                val=1.8e17;
            }
            else if(isinf(val)==-1)
            {
                val=-1.8e17;
            }
            else if(isnan(val))
            {
                val=0;
            }
            else
            {
                val=val/sub;
                if(val>1.8e17)
                    val=1.8e17;
                else if(val<-1.8e17)
                    val=-1.8e17;

            }
            output[i].feature[c]=val;
        }

        if(need_save_worker_conf){
            output[i].feature.clear();
            std::vector<float> feature_conf = {worker_conf};
            output[i].feature = feature_conf;
        }

    }

};
bool sort_score(Result_item_burst_into_ban_detection& box1, Result_item_burst_into_ban_detection& box2)
{
    return (box1.score > box2.score);


};
inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
};
inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
};
void nms_yolo(Output& net_output, std::vector<Result_item_burst_into_ban_detection>& output, std::vector<int> class_filter, float threshold_score=0.25, float threshold_iou=0.45)
{
    float* input = (float*)net_output.data.data();

    // input: x1, y1, x2, y2, conf, cls

    int dim1 = net_output.shape[1];
    int dim2 = net_output.shape[2];
    std::vector<Result_item_burst_into_ban_detection> result;
    float threshold_score_stage_1 = threshold_score*0.77;
    for (int k=0, i=0; k<dim1; k++, i+=dim2)
    {
        float obj_conf = input[i + 9];
        obj_conf = sigmoid(obj_conf);
        if (obj_conf > threshold_score_stage_1)
        {
            Result_item_burst_into_ban_detection item;
            float max_class_conf = input[i + 10];
            int max_class_id = 0;
            for (int j=1; j<dim2-10; j++)
            {
                if (input[i + 10 + j] > max_class_conf) {
                    max_class_conf = input[i + 10 + j];
                    max_class_id = j;
                }
            }
            max_class_conf = obj_conf*sigmoid(max_class_conf);
            if (max_class_conf > threshold_score_stage_1)
            {

                float cx = (sigmoid(input[i+5])*2 + input[i]) * input[i+4];
                float cy = (sigmoid(input[i+6])*2 + input[i+1]) * input[i+4];
                float w = sigmoid(input[i+7])*2;
                w = w*w*input[i+2];
                float h = sigmoid(input[i+8])*2;
                h = h*h*input[i+3];
                float threshold_score_stage_2 = threshold_score*0.77;
                if(abs(cx-320)-160>0)
                    threshold_score_stage_2 = threshold_score*(0.67+0.4*w/(abs(cx-320)-160));
                if(abs(cy-176)-88>0)
                    threshold_score_stage_2 = threshold_score*(0.67+0.4*h/(abs(cy-176)-88));
                if(threshold_score_stage_2 > threshold_score*1.2)
                    threshold_score_stage_2 = threshold_score*1.2;

                if(max_class_conf < threshold_score_stage_2)
                    continue;

                item.x1 = cx-w/2;
                item.y1 = cy-h/2;
                item.x2 = cx+w/2;
                item.y2 = cy+h/2;

                item.score = max_class_conf;
                item.class_id = max_class_id;
                item.x1 += item.class_id * 4096;
                item.x2 += item.class_id * 4096;
                if (class_filter.size() > 0)
                {
                    if (find(class_filter.begin(), class_filter.end(), max_class_id) != class_filter.end())
                    {
                        result.push_back(item);
                    }
                }
                else
                {
                    result.push_back(item);
                }
            }
        }
    }
    output.clear();
    if (result.size() <= 0)
        return;

    while (result.size() > 0)
    {
        std::sort(result.begin(), result.end(), sort_score);
        output.push_back(result[0]);
        for (int i = 0;i < result.size() - 1;i++)
        {
            float iou_value = iou(result[0], result[i + 1]);
            if (iou_value > threshold_iou)
            {
                result.erase(result.begin()+i + 1);
                i-=1;
            }
        }
        result.erase(result.begin());
    }
    std::vector<Result_item_burst_into_ban_detection>::iterator iter=output.begin();
    for(;iter!=output.end();iter++){
        iter->x1-=iter->class_id*4096;
        iter->x2-=iter->class_id*4096;
    }

    return ;
};

Duplicate_remover::Duplicate_remover()
{

};
Duplicate_remover::~Duplicate_remover()
{

};
void Duplicate_remover::set_min_interval(int interval)
{
    this->min_interval=interval;
    if(this->min_interval<1)
        this->min_interval=1;
    if(this->max_interval<this->min_interval)
        this->max_interval=this->min_interval;
};
void Duplicate_remover::set_max_interval(int interval)
{
    this->max_interval=interval;
    if(this->max_interval<this->min_interval)
        this->max_interval=this->min_interval;

};
void Duplicate_remover::set_accept_sim_thres(float thres)
{
    this->accept_sim_thres=thres;
    if(this->accept_sim_thres>0.99)
        this->accept_sim_thres=0.99;
    if(this->accept_sim_thres<-1)
        this->accept_sim_thres=-1;
};
void Duplicate_remover::set_trigger_sim_thres(float thres)
{
    this->trigger_sim_thres=thres;
    if(this->trigger_sim_thres>0.99)
        this->trigger_sim_thres=0.99;
    if(this->trigger_sim_thres<-1)
        this->trigger_sim_thres=-1;

};
void Duplicate_remover::set_iou_thres(float thres)
{
    this->iou_thres=thres;
};
std::vector<std::pair<int,float>> Duplicate_remover::check_score(std::vector<Result_item_burst_into_ban_detection>& result)
{
    std::vector<std::pair<int,float>> res;
    for(int i=0;i<result.size();i++){
        Result_item_burst_into_ban_detection* item_1=&(result[i]);
        int last_item_idx=-1;
        float last_sim_score=-1;
        for(int j=0;j<last_accept_res.size();j++){
            Result_item_burst_into_ban_detection* item_2=&(last_accept_res[j].first);
            if(item_1->class_id!=item_2->class_id)
                continue;
            float iou_value=iou(item_1,item_2);
            if(iou_value>=iou_thres){
                float sim_score=cos_distance(item_1->feature,item_2->feature);
                if(sim_score>last_sim_score){
                    last_item_idx=j;
                    last_sim_score=sim_score;
                }

            }
        }
        res.push_back(std::make_pair(last_item_idx,last_sim_score));
    }
    return res;
};
std::vector<Result_item_burst_into_ban_detection> Duplicate_remover::process(std::vector<Result_item_burst_into_ban_detection> result)
{
    for (int i=0; i<last_accept_res.size(); i++) {
        last_accept_res[i].second++;
    }
    for (auto iter=last_accept_res.begin();iter!=last_accept_res.end();iter++) {
        if ((*iter).second>max_interval+10) {
            iter=last_accept_res.erase(iter);
            iter--;
        }
    }

    std::vector<std::pair<int,float>> sim_score=check_score(result);
    std::set<int> filted_res_idx;
    for (int i=0;i<result.size();i++){
        std::pair<int,float> idx_score=sim_score[i];
        if(idx_score.first<0||idx_score.second<trigger_sim_thres){
            filted_res_idx.insert(i);
        }
    }
    for (int i=0;i<result.size();i++){
        if(filted_res_idx.find(i)!=filted_res_idx.end()){
            continue;
        }
        std::pair<int,float> idx_score=sim_score[i];
        if(idx_score.second<accept_sim_thres){
            int interval=(idx_score.second-trigger_sim_thres)/(accept_sim_thres-trigger_sim_thres)*(max_interval-min_interval);
            if(last_accept_res[idx_score.first].second>interval){
                filted_res_idx.insert(i);
            }
        }
        else{
            if(last_accept_res[idx_score.first].second>=max_interval){
                filted_res_idx.insert(i);
            }

        }
    }
    std::vector<Result_item_burst_into_ban_detection> filted_res;
    for (int i=0;i<result.size();i++){
        if (filted_res_idx.find(i)==filted_res_idx.end()) {
            continue;
        }

        std::pair<int,float> idx_score=sim_score[i];
        if (idx_score.first<0) {
            // cout<<idx_score.first<<"\t"<<idx_score.second<<"\t"<<"-1\t";
            last_accept_res.push_back(std::make_pair(result[i],0));
        } else {
            // cout<<idx_score.first<<"\t"<<idx_score.second<<"\t"<<last_accept_res[idx_score.first].second<<"\t";
            last_accept_res[idx_score.first]=std::make_pair(result[i],0);
        }
        filted_res.push_back(result[i]);
        // cout<<endl;
    }

    return filted_res;
};

Channel_cfg_Burst_Into_Ban_Detection::Channel_cfg_Burst_Into_Ban_Detection(std::string channel_name):Channel_cfg_base(channel_name)
{
    this->channel_name = channel_name;
};
Channel_cfg_Burst_Into_Ban_Detection::~Channel_cfg_Burst_Into_Ban_Detection()
{

};
std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> Channel_cfg_Burst_Into_Ban_Detection::copy_bounary()
{
    return this->boundary;
};

Channel_data_Burst_Into_Ban_Detection::Channel_data_Burst_Into_Ban_Detection(std::string channel_name):Channel_data_base(channel_name)
{
    this->channel_name = channel_name;

    //set
    this->accepted_boundary_name.insert("vehicle ban");
    this->accepted_boundary_name.insert("机动车闯入_道路");
    this->accepted_boundary_name.insert("non-motor vehicle ban");
    this->accepted_boundary_name.insert("两轮车闯入_道路");
    this->accepted_boundary_name.insert("pedestrian ban");
    this->accepted_boundary_name.insert("行人闯入_道路");

    //map
    this->class_id2class_name.insert(std::make_pair(0,"行人"));
    this->class_id2class_name.insert(std::make_pair(1,"非机动车"));
    this->class_id2class_name.insert(std::make_pair(2,"机动车"));
    this->class_id2class_name.insert(std::make_pair(3,"非机动车"));
    this->class_id2class_name.insert(std::make_pair(4,"机动车"));
    this->class_id2class_name.insert(std::make_pair(5,"机动车"));
    this->class_id2class_name.insert(std::make_pair(6,"机动车"));
    {
        std::set<int> temp;
        temp.insert(0);
        this->region_depended_class.insert(std::make_pair("pedestrian ban", temp));
        this->region_depended_class.insert(std::make_pair("行人闯入_道路", temp));
    }

    {
        std::set<int> temp;
        temp.insert(1);
        temp.insert(3);
        this->region_depended_class.insert(std::make_pair("non-motor vehicle ban", temp));
        this->region_depended_class.insert(std::make_pair("两轮车闯入_道路", temp));

    }

    {
        std::set<int> temp;
        temp.insert(2);
        temp.insert(4);
        temp.insert(5);
        this->region_depended_class.insert(std::make_pair("vehicle ban", temp));
        this->region_depended_class.insert(std::make_pair("机动车闯入_道路", temp));
    }

    this->check_count_thres = 1;
    this->need_count_check = false;
    this->check_exist = true;
};
Channel_data_Burst_Into_Ban_Detection::~Channel_data_Burst_Into_Ban_Detection()
{

};
void Channel_data_Burst_Into_Ban_Detection::set_boundarys(std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> boundary)
{
    for (int mask_idx = 0; mask_idx < boundary.size(); mask_idx++)
    {
        string region_name = boundary[mask_idx].first;
        std::vector<std::pair<float,float>> boundary_tmp = boundary[mask_idx].second;
        if (this->accepted_boundary_name.find(region_name) == this->accepted_boundary_name.end()) {
            continue;
        }
        this->mask_idx2mask_name[mask_idx] = region_name;

        std::vector<cv::Point2f> boundary_;
        for (int i=0; i<boundary_tmp.size(); i++) {
            boundary_.push_back(cv::Point2f(boundary_tmp[i].first, boundary_tmp[i].second));
        }
        this->boundarys_[mask_idx] = boundary_;
    }
};
void Channel_data_Burst_Into_Ban_Detection::init_buffer(int width, int height)
{
    this->grid_width = int(this->grid_factor*width+1);
    this->grid_height = int(this->grid_factor*height+1);
    for(auto iter=this->boundarys_.begin(); iter!=this->boundarys_.end(); iter++)
    {
        int mask_idx = iter->first;
        this->mask_grids[mask_idx] = cv::Mat::zeros(this->grid_height, this->grid_width, CV_16UC1);
        std::vector<cv::Point2f> boundary_ = iter->second;
        std::vector<std::vector<cv::Point>> temp_boundary;
        std::vector<cv::Point> temp_bondary_;

        for (int i=0; i<boundary_.size(); i++)
        {
            cv::Point2f point = boundary_[i];
            temp_bondary_.push_back(cv::Point(int(point.x*(this->grid_width-1)),int(point.y*(this->grid_height-1))));
        }

        temp_boundary.push_back(temp_bondary_);

        grid_temp[mask_idx] = cv::Mat::zeros(this->grid_height, this->grid_width, CV_16UC1);

        cv::fillPoly(this->mask_grids[mask_idx], temp_boundary, cv::Scalar(1));
        this->mask_fg_cnt[mask_idx] = cv::sum(this->mask_grids[mask_idx])[0];

        if (this->need_count_check)
        {
            this->count_grids[mask_idx] = cv::Mat::zeros(this->grid_height, this->grid_width, CV_16UC1);
        }
    }
};
void Channel_data_Burst_Into_Ban_Detection::region_check(std::vector<Result_item_burst_into_ban_detection> detect_result)
{
    for(int i=0; i<detect_result.size(); i++)
    {
        Result_item_burst_into_ban_detection& item=detect_result[i];
        cv::Rect roi=cv::Rect(item.x1*grid_factor+0.5,item.y1*grid_factor+0.5,(item.x2-item.x1)*grid_factor+0.5,(item.y2-item.y1)*grid_factor+0.5);
        
        if (roi.x<0)
            roi.x=0;
        if (roi.y<0)
            roi.y=0;
        if (roi.x+roi.width>grid_width)
            roi.width=grid_width-roi.x;
        if (roi.y+roi.height>grid_height)
            roi.height=grid_height-roi.y;

        if(roi.height<=0)
            roi.height=1;
        if(roi.width<=0)
            roi.width=1;

        for (auto iter=mask_grids.begin(); iter!=mask_grids.end(); iter++)
        {
            string mask_name=mask_idx2mask_name[iter->first];
            if (region_depended_class.find(mask_name)!=region_depended_class.end())
            {
                auto class_idxes=region_depended_class.find(mask_name)->second;
                if (class_idxes.size()==0||class_idxes.find(item.class_id)!=class_idxes.end())
                {
                    cv::Mat roi_mask=iter->second(roi);
                    cv::Mat roi_temp=grid_temp[iter->first](roi);
                    roi_temp+=roi_mask;
                }
            }
        }
    }
    for(auto iter=grid_temp.begin();iter!=grid_temp.end();iter++)
    {
        cv::Mat temp=grid_temp[iter->first]>0;
        temp.convertTo(grid_temp[iter->first],grid_temp[iter->first].type(),1/255.0);
    }


};
void Channel_data_Burst_Into_Ban_Detection::count_add()
{
    for(auto iter=mask_grids.begin();iter!=mask_grids.end();iter++){
        if(need_count_check){
            if(check_exist)
            {
                cv::Mat temp=count_grids[iter->first]+grid_temp[iter->first];
                cv::Mat mask=grid_temp[iter->first]>0;
                cv::Mat mask1=grid_temp[iter->first]<=0;
                cv::Mat mask_t,mask_t1;
                (mask).convertTo(mask_t,temp.type(),1/255.0);
                (mask1).convertTo(mask_t1,temp.type(),1/255.0);
                temp=(temp).mul(mask_t);
                count_grids[iter->first]=temp+mask_t1*(-10);
            }
            else{
                count_grids[iter->first]+=iter->second;
                cv::Mat mask=grid_temp[iter->first]<=0;
                cv::Mat mask_t;
                (mask).convertTo(mask_t,count_grids[iter->first].type(),1/255.0);
                count_grids[iter->first]=count_grids[iter->first].mul(mask_t);
            }
        }

    }
};
void Channel_data_Burst_Into_Ban_Detection::count_check()
{

    if(need_count_check){
        if(check_exist){
            for(auto iter=grid_temp.begin();iter!=grid_temp.end();iter++){
                cv::Mat temp=count_grids[iter->first]>check_count_thres;
                temp.convertTo(iter->second,iter->second.type(),1/255.0);
//              iter->second=count_grids[iter->first]>check_count_thres;
            }
        }
        else{
            for(auto iter=grid_temp.begin();iter!=grid_temp.end();iter++){
                cv::Mat temp=count_grids[iter->first]>check_count_thres;
                temp.convertTo(iter->second,iter->second.type(),1/255.0);
//              iter->second=(count_grids[iter->first]>check_count_thres);
            }
        }
    }
    else{
        if(check_exist){
            for(auto iter=grid_temp.begin();iter!=grid_temp.end();iter++){
                cv::Mat temp=iter->second>0;
                temp.convertTo(iter->second,iter->second.type(),1/255.0);

//              iter->second=iter->second>0;
            }
        }
        else{
            for(auto iter=grid_temp.begin();iter!=grid_temp.end();iter++){
                cv::Mat mask=grid_temp[iter->first]<=0;
                cv::Mat mask_t;
                (mask).convertTo(mask_t,grid_temp[iter->first].type(),1/255.0);

                iter->second=mask_t.mul(mask_grids[iter->first])>0;
            }
        }

    }

};
std::vector<Result_item_burst_into_ban_detection> Channel_data_Burst_Into_Ban_Detection::get_result(std::vector<Result_item_burst_into_ban_detection> detect_result)
{
    std::vector<Result_item_burst_into_ban_detection> result;
    if (this->check_exist)
    {

        for (int i=0; i<detect_result.size(); i++)
        {
            Result_item_burst_into_ban_detection& item=detect_result[i];
            Result_item_burst_into_ban_detection& item_orig=result_orig[i];
            cv::Rect roi = cv::Rect(item.x1*grid_factor+0.5,item.y1*grid_factor+0.5,(item.x2-item.x1)*grid_factor+0.5,(item.y2-item.y1)*grid_factor+0.5);
            if (roi.x<0)
                roi.x=0;
            if (roi.y<0)
                roi.y=0;
            if (roi.x+roi.width>grid_width)
                roi.width=grid_width-roi.x;
            if (roi.y+roi.height>grid_height)
                roi.height=grid_height-roi.y;
            float score=0;
            int region_idx=-1;
            for (auto iter=grid_temp.begin(); iter!=grid_temp.end(); iter++)
            {
                string mask_name=mask_idx2mask_name[iter->first];
                if (region_depended_class.find(mask_name)!=region_depended_class.end())
                {
                    auto class_idxes=region_depended_class.find(mask_name)->second;
                    if (class_idxes.size()==0||class_idxes.find(item.class_id)!=class_idxes.end())
                    {
                        cv::Mat roi_temp=grid_temp[iter->first](roi);
                        int sum_value=cv::sum(roi_temp)[0];
                        int box_size=(item.x2-item.x1)*(item.y2-item.y1)*grid_factor*grid_factor;
                        float temp_score=1.0*sum_value/box_size*item.score;
                        // float temp_score=cv::sum(roi_temp)[0]*1.0/(item.x2-item.x1)/(item.y2-item.y1)*item.score;
                        if (temp_score>score)
                        {
                            score=temp_score;
                            region_idx=iter->first;
                        }
                    }
                }
            }
            try {
                throw 1;
            }
            catch(...) {

            }
            if (score>0.1)
            {
                Result_item_burst_into_ban_detection res;
                res.x1=item_orig.x1;
                res.y1=item_orig.y1;
                res.x2=item_orig.x2;
                res.y2=item_orig.y2;
                res.class_id=item_orig.class_id;
                res.feature=item_orig.feature;
                res.region_idx=region_idx;
                res.score=score;
                res.contour=std::vector<std::pair<float,float>>();
                res.contour.push_back(std::make_pair(res.x1,res.y1));
                res.contour.push_back(std::make_pair(res.x2,res.y1));
                res.contour.push_back(std::make_pair(res.x2,res.y2));
                res.contour.push_back(std::make_pair(res.x1,res.y2));
                result.push_back(res);
            }
        }
    }
    else
    {
        for (auto iter=grid_temp.begin();iter!=grid_temp.end();iter++)
        {
            int fg_cng=cv::sum(iter->second)[0];
            if (fg_cng<this->check_sensitivity_thres*mask_fg_cnt[iter->first]||fg_cng<4)
            {
                continue;
            }
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(grid_temp[iter->first],contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
            for (int i=0;i<contours.size();i++)
            {
                cv::Rect rect=cv::boundingRect(contours[i]);
                Result_item_burst_into_ban_detection res;
                res.x1=(rect.x+0.5)/grid_factor;
                res.y1=(rect.y+0.5)/grid_factor;
                res.x2=(rect.x+rect.width+0.5)/grid_factor;
                res.y2=(rect.y+rect.height+0.5)/grid_factor;
                res.class_id=iter->first;
                res.region_idx=iter->first;
                std::vector<cv::Point> contour;
                approxPolyDP(contours[i],contour,3,true);
                res.score=cv::contourArea(contour)/mask_fg_cnt[iter->first];
                if (res.score<0.1)
                    continue;
                res.contour=std::vector<std::pair<float,float>>();
                for (int j=0;j<contour.size();j++)
                {
                    cv::Point& point=contour[j];
                    res.contour.push_back(std::make_pair((point.x+0.5)/grid_factor,(point.y+0.5)/grid_factor));
                }
                result.push_back(res);
            }
        }
    }

    return result;

};
// IoU 计算函数，假设已经在类中定义好
float Channel_data_Burst_Into_Ban_Detection::bboxSimilarity(const Result_item_burst_into_ban_detection& a, const Result_item_burst_into_ban_detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float bbox1Area = (a.x2 - a.x1) * (a.y2 - a.y1);
    float bbox2Area = (b.x2 - b.x1) * (b.y2 - b.y1);
    float unionArea = bbox1Area + bbox2Area - intersectionArea;

    if (unionArea <= 0.0f) {
        return 0.0f; // 避免除以零
    }

    return intersectionArea / unionArea; // 返回 IoU
}


// 去掉在车里的人
std::vector<Result_item_burst_into_ban_detection> Channel_data_Burst_Into_Ban_Detection::remove_person_in_car(
    std::vector<Result_item_burst_into_ban_detection> yolo_res, float iou_threshold) 
{
    std::vector<Result_item_burst_into_ban_detection> res;
    if (yolo_res.size() == 0) return res;

    // names: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']

    for (auto& person : yolo_res) {
        // 如果检测的不是人，直接加入结果
        if (person.class_id != 0) {
            res.push_back(person);
            continue;
        }

        bool person_in_car = false;
        for (const auto& car : yolo_res) {
            // 检查是否是车辆
            if (car.class_id == 0){
                // cout<<"person: "<<car.score<<endl;
                continue;
            }

            // 检查人的包围框是否完全在车辆内
            bool fully_inside = (person.x1 >= car.x1 && person.y1 >= car.y1 && person.x2 <= car.x2 && person.y2 <= car.y2);

            // 计算人和车辆之间的IoU
            float iou = bboxSimilarity(person, car);

            // 如果人完全在车内或 IoU 超过给定阈值 则不保留这个人的检测
            if (fully_inside ) {
                // cout<<"人完全在车内"<<endl;
                person_in_car = true;
                break;
            }

            if(iou > iou_threshold){
                // cout<<"发现人在车旁边, Iou: "<<iou<<endl;
                person_in_car = true;
                break;
            }
        }

        // 如果人不在任何车里或者IoU未超过阈值 则保留
        if (!person_in_car) {
            // 如果认为人类不在车内 再次检查人的包围框形状
            float person_box_w = person.x2 - person.x1;  // 113
            float person_box_h = person.y2 - person.y1;  // 130
            // 接近于正方形的人类包围框
            if((1.0 < person_box_w / person_box_h && person_box_w / person_box_h < 1.3) || (1.0 < person_box_h / person_box_w && person_box_h / person_box_w < 1.3)){
                // cout<<"接近于正方形的人类包围框"<<endl;
                continue;
            }
            if((1.29 < person_box_w / person_box_h && person_box_w / person_box_h < 1.55) || (1.29 < person_box_h / person_box_w && person_box_h / person_box_w < 1.55)){
                person.score *= 0.85;  //降低置信度
                // cout<<"降低置信度"<<endl;
            }
            res.push_back(person);
        }
    }

    return res;
}

std::string Channel_data_Burst_Into_Ban_Detection::decode_tag(Result_item_burst_into_ban_detection item)
{
    if(check_exist)
        if (item.class_id == 0 || item.class_id == 1 || item.class_id == 3)
            return class_id2class_name[item.class_id]+" 闯入 ";
        else
            return class_id2class_name[item.class_id]+" 闯入 "+mask_idx2mask_name[item.region_idx];
    else
        return mask_idx2mask_name[item.class_id]+" 丢失或移位";

};

Alg_Module_Burst_Into_Ban_Detection::Alg_Module_Burst_Into_Ban_Detection():Alg_Module_Base_private("burst_into_ban_detection")
{   //参数是模块名，使用默认模块名初始化

};
Alg_Module_Burst_Into_Ban_Detection::~Alg_Module_Burst_Into_Ban_Detection()
{

};

bool Alg_Module_Burst_Into_Ban_Detection::init_from_root_dir(std::string root_dir)
{
    bool load_res;
    //加载模块配置文件
    load_res = this->load_module_cfg(root_dir+ "/cfgs/" + this->node_name + "/module_cfg.xml");
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load module cfg failed",this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }

    std::shared_ptr<Module_cfg_Burst_Into_Ban_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Burst_Into_Ban_Detection>(this->get_module_cfg());

    //如果文件中有运行频率的字段，则使用文件中设定的频率

    //如果文件中有运行频率的字段，则使用文件中设定的频率
    float remove_person_iou_thres;
    load_res = module_cfg->get_float("remove_person_iou_thres", remove_person_iou_thres);
    if (load_res)
        this->remove_person_iou_thres = remove_person_iou_thres;
    else
        this->remove_person_iou_thres = 0.45;


    int tick_interval;
    load_res = module_cfg->get_int("tick_interval", tick_interval);
    if (load_res)
        this->tick_interval_ms = tick_interval_ms;
    else
        this->tick_interval_ms = 100;

    //加载模块参数
    load_res = module_cfg->get_string("model_path", this->model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("model_name", this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("model_cfg_path", this->model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    //加载交警分类模型参数
    load_res = module_cfg->get_string("is_police_model_path", this->is_police_model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("is_police_model_name", this->is_police_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("is_police_model_cfg_path", this->is_police_model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    load_res = module_cfg->get_string("worker_classify_model_path", this->worker_classify_model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("worker_classify_model_name", this->worker_classify_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("worker_classify_model_cfg_path", this->worker_classify_model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }


//    load_res = module_cfg->get_int("debug", this->debug);
  //  if (!load_res) throw Alg_Module_Exception("load debug failed", this->node_name, Alg_Module_Exception::Stage::check);

    //加载模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name +"/"+ this->model_cfg_path, this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model_cfg failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    //加载交警判定模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name +"/"+ this->is_police_model_cfg_path, this->is_police_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model_cfg failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    //加载施工工人分类模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name +"/"+ this->worker_classify_model_cfg_path, this->worker_classify_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model_cfg failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    //加载模型
    load_res = this->load_model(root_dir + "/models/" + this->model_path , this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }
    //加载交警判定模型
    load_res = this->load_model(root_dir + "/models/" + this->is_police_model_path , this->is_police_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }
    // 加载施工工人判定模型    
    load_res = this->load_model(root_dir + "/models/" + this->worker_classify_model_path, this->worker_classify_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }

    
    return true;
};
bool Alg_Module_Burst_Into_Ban_Detection::is_police_forward(std::map<std::string, std::shared_ptr<InputOutput>>& input, int x1, int x2, int y1, int y2)
{   
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }

    //获取指定的模型实例
    auto resnet = this->get_model_instance(this->is_police_model_name);
    if (resnet == nullptr) {
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);       //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        return false;
    }

    //获取计算卡推理核心的handle
    std::shared_ptr<Device_Handle> handle;
//    torch::Device handle(torch::DeviceType::CPU);
    this->get_device(handle);                       //libtorch下大部分情况下应该可以不获取handle了

    //判断模型是否已经加载
    auto input_shapes = resnet->get_input_shapes();
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;
    }
    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        handle=input["image"]->data.image->get_handle();
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

    }
    
        

    int width=input_image->get_width();                            //不同类型图片下，获取的图片宽度
    int height=input_image->get_height();                           //不同类型图片下，获取的图片高度

    int sub_img_w = (int)(x2 - x1);         //子图的宽度
    int sub_img_h = (int)(y2 - y1);         //子图的高度

    std::vector<Output> net_output;                                         
    cv::Rect crop_rect;                                                             //剪裁区域
    crop_rect.x = x1;
    crop_rect.y = y1;
    crop_rect.width = sub_img_w;
    crop_rect.height= sub_img_h;
    if(crop_rect.x<0){
        crop_rect.x=0;
    }
    if(crop_rect.y<0){
        crop_rect.y=0;
    }
    if(crop_rect.x+crop_rect.width>=width){
        crop_rect.width=width-crop_rect.x-1;
    }
    if(crop_rect.y+crop_rect.height>=height){
        crop_rect.height=height-crop_rect.y-1;
    }
    std::shared_ptr<QyImage> sub_image=input_image->crop_resize(crop_rect,384,384);
    sub_image=sub_image->cvtcolor(true);
    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(sub_image);

    resnet->forward(net_input, net_output);                                 


    float* res = (float*)net_output[0].data.data();

    float no_police = *res;
    ++res;
    float police = *res;

    // cout<<no_police<<" "<<police<<endl;
    if(no_police > police) {
        // cout<<"非警察"<<endl;
        return false;
    }
    return true;
}

bool Alg_Module_Burst_Into_Ban_Detection::classify_engineering_worker(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>> &input, std::vector<Result_item_burst_into_ban_detection> &results){

    //获取计算卡推理核心的handle
    std::shared_ptr<Device_Handle> handle;
//    torch::Device handle(torch::DeviceType::CPU);
    this->get_device(handle);                       //libtorch下大部分情况下应该可以不获取handle了

    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        handle=input["image"]->data.image->get_handle();
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

    }

    std::shared_ptr<Model_cfg_Burst_Into_Ban_Detection> model_cfg = std::dynamic_pointer_cast<Model_cfg_Burst_Into_Ban_Detection>(this->get_model_cfg(this->model_name)); //获取模型配置，派生的模型配置文件类指针需要手动转换为子类
    std::shared_ptr<Channel_data_Burst_Into_Ban_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Burst_Into_Ban_Detection>(this->get_channal_data(channel_name));
    std::shared_ptr<Module_cfg_Burst_Into_Ban_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Burst_Into_Ban_Detection>(this->get_module_cfg());             //获取模型配置，派生的模型配置文件类指针需要手动转换为子类

    //获取指定的模型实例
    auto net = this->get_model_instance(this->worker_classify_model_name);
    if (net == nullptr) {
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference); 
        return false;
    }
   
    //检查参数设置
    float thresh_score;
    bool load_res = true;
    load_res &= module_cfg->get_float("engineering_worker_score", thresh_score);
    if (load_res == false) {
        throw Alg_Module_Exception("Error:\t load module param failed",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }

    //判断模型是否已经加载
    auto input_shapes = net->get_input_shapes();
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }

    int width=input_image->get_width();                            //不同类型图片下，获取的图片宽度
    int height=input_image->get_height();                           //不同类型图片下，获取的图片高度


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (auto &result : results) {
        
        // 只对行人进行分类
        if (result.class_id != 0){
            continue;
        }
        
        cv::Rect crop_rect;                                                             //剪裁区域
        crop_rect.x = result.x1;
        crop_rect.y = result.y1;
        crop_rect.width = result.x2 - result.x1;   
        crop_rect.height= result.y2 - result.y1;
        if(crop_rect.x<0){
            crop_rect.x=0;
        }
        if(crop_rect.y<0){
            crop_rect.y=0;
        }
        if(crop_rect.x+crop_rect.width>=width){
            crop_rect.width=width-crop_rect.x-1;
        }
        if(crop_rect.y+crop_rect.height>=height){
            crop_rect.height=height-crop_rect.y-1;
        }
        if(crop_rect.width <= 0 || crop_rect.height <= 0){
            std::vector<float> feature_conf = {1.0};  // 异常边界框 去除这个值
            result.feature = feature_conf;
            continue;
        }

        

        float image_width = result.x2 - result.x1;
        float image_height = result.y2 - result.y1;

        auto input_shape_ = input_shapes[0];    //[channel, height, width]

        std::vector<Output> net_output;
        std::shared_ptr<QyImage> sub_image=input_image->crop_resize_keep_ratio(crop_rect,input_shape_.dims[3],input_shape_.dims[2],0);
        std::vector<std::shared_ptr<QyImage>> net_input;
        net_input.push_back(sub_image);
        net->forward(net_input, net_output);


        float* y = (float*)net_output[0].data.data();
        // std::cout << "\n 置信度1: " << y[0];
        // std::cout << "\n 置信度2: " << y[1] << std::endl;
        // result.score = result.score * y[0] * (1-y[1]);
        // 借用feature 暂存施工工人的置信度
        // vector<float> feature_conf = {result.score * y[0] * (1-y[1])};
        std::vector<float> feature_conf = {y[0]};
        result.feature = feature_conf;
        // std::cout << "\n 是施工工人的置信度" << result.feature[0] << std::endl;
    }

    std::vector<Result_item_burst_into_ban_detection>::iterator result = results.begin();
    for (; result != results.end(); )
    {
        if (result->class_id == 0 && result->feature[0] > thresh_score){
            // cout<<"result->feature[0]: "<<result->feature[0]<<endl;
            result = results.erase(result); 
        } else {
            result++;
        }
    }
    return true;

};    //进行模型推理

bool Alg_Module_Burst_Into_Ban_Detection::detect_person(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::vector<Result_item_burst_into_ban_detection> &detections) 
{
    //获取计算卡推理核心的handle，但大部分情况下不需要获取handle了
    std::shared_ptr<Device_Handle> handle;
//    torch::Device handle(torch::DeviceType::CPU);
    this->get_device(handle);                       //libtorch下大部分情况下应该可以不获取handle了

    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        handle=input["image"]->data.image->get_handle();
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

    }


    std::shared_ptr<Model_cfg_Burst_Into_Ban_Detection> model_cfg = std::dynamic_pointer_cast<Model_cfg_Burst_Into_Ban_Detection>(this->get_model_cfg(this->model_name)); //获取模型配置，派生的模型配置文件类指针需要手动转换为子类
    std::shared_ptr<Channel_data_Burst_Into_Ban_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Burst_Into_Ban_Detection>(this->get_channal_data(channel_name));
    std::shared_ptr<Module_cfg_Burst_Into_Ban_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Burst_Into_Ban_Detection>(this->get_module_cfg());             //获取模型配置，派生的模型配置文件类指针需要手动转换为子类

    //检查参数设置
    std::vector<int> classes;
    float thresh_iou;
    float thresh_score;
    bool load_res = true;
    load_res &= module_cfg->get_int_vector("classes", classes);
    load_res &= module_cfg->get_float("thresh_score", thresh_score);
    load_res &= module_cfg->get_float("thresh_iou", thresh_iou);
    if (load_res == false) {
        throw Alg_Module_Exception("Error:\t load module param failed",this->node_name,Alg_Module_Exception::Stage::inference);         //找不到必要的配置参数，检查配置文件是否有对应的字段，检查类型，检测名称
        return false;
    }

    //获取指定的模型实例
    auto net = this->get_model_instance(this->model_name);
    if (net == nullptr) {
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);       //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        return false;
    }

    //判断模型是否已经加载
    auto input_shapes = net->get_input_shapes();
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;
    }

    auto input_shape_ = input_shapes[0];    //[channel, height, width]  实际模型输入的形状

    int width=input_image->get_width();                            //不同类型图片下，获取的图片宽度
    int height=input_image->get_height();                           //不同类型图片下，获取的图片高度

    float factor1 = input_shape_.dims[3] * 1.0 / width;
    float factor2 = input_shape_.dims[2] * 1.0 / height;
    float factor = factor1 > factor2 ? factor2 : factor1;   //选择较小的比例

    std::vector<Output> net_output;
    std::shared_ptr<QyImage> sub_image=input_image->resize_keep_ratio(input_shape_.dims[3],input_shape_.dims[2],0);
    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(sub_image);
    net->forward(net_input, net_output);


    // std::vector<Result_item_burst_into_ban_detection> detections;
    nms_yolo(net_output[0], detections, classes, thresh_score, thresh_iou);

    for (auto iter = detections.begin(); iter != detections.end(); iter++)
    {
        iter->x1 = (iter->x1 + 0.5) / factor;
        iter->y1 = (iter->y1 + 0.5) / factor;
        iter->x2 = (iter->x2 + 0.5) / factor;
        iter->y2 = (iter->y2 + 0.5) / factor;
    }

    roi_pooling(net_output[1], detections, input_shape_.dims[2] / factor, input_shape_.dims[3] / factor);

    //通道需要的数据
    channel_data->factor = factor;
    channel_data->img_h = input_shape_.dims[2] / factor;
    channel_data->img_w = input_shape_.dims[3] / factor;

    if (channel_data->net_ouput.data.size()!=0 || channel_data->net_ouput.shape.size()!=0) {
        channel_data->net_ouput.data.clear();
        channel_data->net_ouput.shape.clear();
    }

    if (channel_data->need_feature) {
        channel_data->net_ouput.data = net_output[1].data;
        channel_data->net_ouput.shape = net_output[1].shape;
    }

    return true;
};
bool Alg_Module_Burst_Into_Ban_Detection::forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::map<std::string, std::shared_ptr<InputOutput>>& output)
{
    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }


    std::vector<Result_item_burst_into_ban_detection> detections;
    this->detect_person(channel_name, input, detections);

    std::shared_ptr<Channel_data_Burst_Into_Ban_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Burst_Into_Ban_Detection>(this->get_channal_data(channel_name));
    if (channel_data->width==0 || channel_data->height==0) {
        int width=0;                            //不同类型图片下，获取的图片宽度
        int height=0;                           //不同类型图片下，获取的图片高度
        if(input["image"]->data_type==InputOutput::Type::Image_t){
            auto input_image=input["image"]->data.image;
            if(input_image==nullptr){
                throw Alg_Module_Exception("Error:\t image error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
            }
            width=input_image->get_width();
            height=input_image->get_height();
        }
        else
        {
            throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建

        }
        channel_data->width = width;
        channel_data->height = height;
        channel_data->init_buffer(channel_data->width, channel_data->height);
    }


    //整理检测数据
    auto forward_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    forward_output->data.detect.resize(detections.size());
    auto& forward_result = forward_output->data.detect;
    for (int i = 0; i < detections.size(); i++)                                             //循环填充结果数据
    {
        // std::cout << detections[i].str() << std::endl;
        forward_result[i].x1         = detections[i].x1;
        forward_result[i].y1         = detections[i].y1;
        forward_result[i].x2         = detections[i].x2;
        forward_result[i].y2         = detections[i].y2;
        forward_result[i].score      = detections[i].score;
        forward_result[i].class_id   = detections[i].class_id;

        forward_result[i].tag        = detections[i].tag;
        forward_result[i].region_idx = detections[i].region_idx;
        forward_result[i].new_obj    = detections[i].new_obj;
        forward_result[i].temp_idx   = detections[i].temp_idx;
        forward_result[i].feature    = detections[i].feature;
        forward_result[i].contour    = detections[i].contour;
        forward_result[i].new_obj    = detections[i].new_obj;
    }
    output.clear();
    output["result"] = forward_output;
    output["image"] = input["image"];

    return true;
};
bool Alg_Module_Burst_Into_Ban_Detection::filter(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::map<std::string, std::shared_ptr<InputOutput>>& output)
{
    //检查是否包含需要的数据
    if (input.find("result") == input.end()) {
        throw Alg_Module_Exception("Error:\t can't find \"result\" in filter.input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    std::shared_ptr<Channel_data_Burst_Into_Ban_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Burst_Into_Ban_Detection>(this->get_channal_data(channel_name));
    std::shared_ptr<Module_cfg_Burst_Into_Ban_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Burst_Into_Ban_Detection>(this->get_module_cfg());             //获取模型配置，派生的模型配置文件类指针需要手动转换为子类

    //获取检测结果
    auto& forward_result = input["result"]->data.detect;

    //没有检测结果
    if (forward_result.size() <= 0)
    {   
        auto detect_res = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
        output["result"] = detect_res;
        return true;
    }
    
    //没有边界设置
    if (channel_data->boundarys_.size() <= 0) 
    {
        auto detect_res = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
        output["result"] = detect_res;
        return true;
    }

    //转换到内部处理方法
    std::vector<Result_item_burst_into_ban_detection> detect_res_;
    detect_res_.resize(forward_result.size());
    for (int i = 0; i < forward_result.size(); i++)
    {
        detect_res_[i].x1         = forward_result[i].x1;
        detect_res_[i].y1         = forward_result[i].y1;
        detect_res_[i].x2         = forward_result[i].x2;
        detect_res_[i].y2         = forward_result[i].y2;
        detect_res_[i].score      = forward_result[i].score;
        detect_res_[i].class_id   = forward_result[i].class_id;
        detect_res_[i].tag        = forward_result[i].tag;
        detect_res_[i].region_idx = forward_result[i].region_idx;
        detect_res_[i].new_obj    = forward_result[i].new_obj;
        detect_res_[i].temp_idx   = forward_result[i].temp_idx;
        detect_res_[i].feature    = forward_result[i].feature;
        detect_res_[i].contour    = forward_result[i].contour;
        detect_res_[i].new_obj    = forward_result[i].new_obj;

        // if(detect_res[i].class_id == 0){
        //     cout<<"行人置信度: "<<detect_res[i].score<<endl;
        // }
    }

    // 在这里进行车内人员的判断 2024.09.04 yan
    std::vector<Result_item_burst_into_ban_detection> detect_res = channel_data->remove_person_in_car(detect_res_, this->remove_person_iou_thres);

    //重置各个边界在原始图像上的掩码
    for (auto iter=channel_data->boundarys_.begin(); iter!=channel_data->boundarys_.end(); iter++) {
        int mask_idx = iter->first;
        channel_data->grid_temp[mask_idx] = cv::Mat::zeros(channel_data->grid_height, channel_data->grid_width, CV_16UC1);
    }

    //行人和非机动车去重，两者同时出现的时候去掉行人，保留机动车
    std::vector<int>remove_man_i;
    remove_man_i.resize(detect_res.size(),0);
//    int remove_man_i[detect_res.size()];
    int remove_man_index = 0;
    for (int j=0; j<detect_res.size(); j++)
    {
        Result_item_burst_into_ban_detection& item_j = detect_res[j];

        if (item_j.class_id != 0) continue;

        float iou_item = 0.0;
        for (int k=0; k<detect_res.size(); k++)
        {
            Result_item_burst_into_ban_detection& item_k=detect_res[k];
            if (item_k.class_id == 1 || item_k.class_id == 3)
            {
                //计算iou
                int max_x = std::max(item_j.x1,item_k.x1);  // 找出左上角坐标哪个大
                int min_x = std::min(item_j.x2,item_k.x2);  // 找出右上角坐标哪个小
                int max_y = std::max(item_j.y1,item_k.y1);
                int min_y = std::min(item_j.y2,item_k.y2);
                float over_area = (min_x - max_x) * (min_y - max_y);  // 计算重叠面积
                float area_a = (item_j.x2 - item_j.x1) * (item_j.y2 - item_j.y1);
                float area_b = (item_k.x2 - item_k.x1) * (item_k.y2 - item_k.y1);
                if (min_x<=max_x || min_y<=max_y) // 如果没有重叠
                    iou_item = 0.0;
                else iou_item = over_area / (area_a + area_b - over_area);
                //cout<<"iou: "<<iou_item<<endl;
                if (iou_item >= channel_data->man_non_motor_thresh_iou)
                {
                    remove_man_i[j] = 1;
                    remove_man_index++;
                    break;
                }
            }
        }
    }

    //去掉重合的行人和非机动车
    std::vector<Result_item_burst_into_ban_detection> detect_res_new;
    for (int h=0; h<detect_res.size(); h++)
    {
        if (remove_man_i[h] != 1) {
            detect_res_new.push_back(detect_res[h]);
        }
    }
    detect_res = detect_res_new;

    //在这里引入交警与非交警的分支 resnet18对二轮车子图进行分类
    //子图resize为384*384 结果id=0为非交警id=1为交警
    //2024.02.26 yan

    //检查是否开启交警判定
    int is_filter_police;
    bool load_res = true;
    load_res &= module_cfg->get_int("is_filter_police", is_filter_police);
    if(is_filter_police == 1) // 开启状态
    { 
        bool is_exist_motor = false;
        std::vector<int> remove_police_i;
        int remove_police_index = 0;
        remove_police_i.resize(detect_res.size(),0);

        if (input.find("image") == input.end())
                {
            throw Alg_Module_Exception("Error:\t can't find \"result\" in filter.input",this->node_name,Alg_Module_Exception::Stage::filter);
            return false;
        }

#ifdef _OPENMP
#pragma omp parallel for            
#endif
        for(int i=0;i<detect_res.size();i++)
        {
            if (detect_res[i].class_id == 1 || detect_res[i].class_id == 3)// 当前检测类别中存在二轮车
            {
                //cout<<"检测到非机动车闯入"<<endl;
                //检查是否包含图片

                bool is_P = this->is_police_forward(input, detect_res[i].x1,  detect_res[i].x2, detect_res[i].y1, detect_res[i].y2);
                
                //如果是交警 记录
                if(is_P)
                {
                    remove_police_i[i] = 1;
                    continue; 
                }
                //如果不是交警 直接跳过
                
            }else{
                continue;
            }
        }
        for(int i=0;i<remove_police_i.size();i++){
            if(remove_police_i[i]>0)
                remove_police_index++;

        }

        //去掉带有交警的结果
        std::vector<Result_item_burst_into_ban_detection> detect_removed_police;
        for (int h=0; h<detect_res.size(); h++)
        {
            if (remove_police_i[h] != 1) {
                detect_removed_police.push_back(detect_res[h]);
            }
        }
        detect_res = detect_removed_police;
    }
    //以上为新增交警判定功能 2024.2.26 yan

    std::vector<Result_item_burst_into_ban_detection> detections_engineering_worker;
    // this->detect_engineering_worker(channel_name, handle, input_image, detections_engineering_worker);

    detections_engineering_worker.resize(detect_res.size());
    for (int i = 0; i < detect_res.size(); ++i) {
        detections_engineering_worker[i].x1        = detect_res[i].x1;
        detections_engineering_worker[i].y1        = detect_res[i].y1;
        detections_engineering_worker[i].x2        = detect_res[i].x2;
        detections_engineering_worker[i].y2        = detect_res[i].y2;
        detections_engineering_worker[i].score     = detect_res[i].score;
        detections_engineering_worker[i].class_id  = detect_res[i].class_id;
    }

    this->classify_engineering_worker(channel_name, input, detections_engineering_worker);

    detect_res = detections_engineering_worker;



    //行人和非机动车得分阈值过滤
    std::vector<Result_item_burst_into_ban_detection> detect_new;
    for (int y=0; y<detect_res.size(); y++)
    {
        if (detect_res[y].class_id==0) {
            if(detect_res[y].score>channel_data->man_thresh_score)
                detect_new.push_back(detect_res[y]);
        }
        else if (detect_res[y].class_id==1) {
            if(detect_res[y].score>channel_data->non_motor_thresh_score)
                detect_new.push_back(detect_res[y]);
        }
        else if (detect_res[y].class_id==3) {
            if (detect_res[y].score>channel_data->non_motor_thresh_score)
                detect_new.push_back(detect_res[y]);
        }
        else {
            detect_new.push_back(detect_res[y]);
        }
    }
    detect_res = detect_new;

    channel_data->result_orig = detect_res;

    for (int i=0; i<detect_res.size(); i++)
    {
        Result_item_burst_into_ban_detection& item=detect_res[i];
        //行人
        if (item.class_id==0)
        {
            //cout<<"man on road"<<endl;
            /*
            float x=(item.x1+item.x2)/2;
            float y=item.y2;
            float w=(item.x2-item.x1)*0.8;
            float h=w*0.5;
            y+=h*0.3;
            item.x1=x-w*0.5;
            item.x2=x+w*0.5;
            item.y1=y-h*0.5;
            item.y2=y+h*0.5;
            */
        }
        //非机动车
        else if (item.class_id==1||item.class_id==3)
        {
            //cout<<"motor on road"<<endl;
            float x=(item.x1+item.x2)/2;
            float y=item.y2;
            float w=(item.x2-item.x1)*0.7;
            float h=w*0.5;
            y+=h*0.3;
            item.x1=x-w*0.5;
            item.x2=x+w*0.5;
            item.y1=y-h*0.5;
            item.y2=y+h*0.5;
        }
        //机动车
        else if (item.class_id==2||item.class_id==4||item.class_id==5)
        {
            float x=(item.x1+item.x2)/2;
            float y=item.y2;
            float w=(item.x2-item.x1)*0.9;
            float h=w*0.5;
            y+=h*0.3;
            item.x1=x-w*0.5;
            item.x2=x+w*0.5;
            item.y1=y-h*0.5;
            item.y2=y+h*0.5;

        }
    }

    channel_data->region_check(detect_res);
    channel_data->count_add();
    channel_data->count_check();

    std::vector<Result_item_burst_into_ban_detection> result;
    result = channel_data->get_result(detect_res);
    for (int i=0; i<result.size(); i++)
    {
        result[i].tag = channel_data->decode_tag(result[i]);
    }
    if (channel_data->need_remove_duplicate)
    {
        roi_pooling(channel_data->net_ouput, result, channel_data->img_h, channel_data->img_w);
        result = channel_data->remover.process(result);
    }

    // 统计行人闯入事件
    Result_Detect event_person_burst_into_ban;
    event_person_burst_into_ban.class_id = 0;
    event_person_burst_into_ban.tag = "行人 闯入 ";
    float worker_cof_sum = 0.0;
    int person_num = 0;
    for (int i = 0; i < result.size(); i++) {
        if (result[i].class_id == 0 && person_num == 0) {
            // std::cout << "行人" << std::endl;
            event_person_burst_into_ban.x1 = std::max(result[i].x1-10, (float)0);
            event_person_burst_into_ban.y1 = std::max(result[i].y1-10, (float)0);
            event_person_burst_into_ban.x2 = std::min(result[i].x2+10, channel_data->img_w-1);
            event_person_burst_into_ban.y2 = std::min(result[i].y2+10, channel_data->img_h-1);
            event_person_burst_into_ban.region_idx = result[i].region_idx;
            event_person_burst_into_ban.score = result[i].score;
            worker_cof_sum += result[i].feature[0];
            person_num += 1;
            continue;
        }
        if (result[i].class_id == 0 && person_num > 0) {
            // std::cout << "行人" << std::endl;
            event_person_burst_into_ban.x1 = std::max(std::min(result[i].x1-10, event_person_burst_into_ban.x1), (float)0);
            event_person_burst_into_ban.y1 = std::max(std::min(result[i].y1-10, event_person_burst_into_ban.y1), (float)0);
            event_person_burst_into_ban.x2 = std::min(std::max(result[i].x2+10, event_person_burst_into_ban.x2), channel_data->img_w-1);
            event_person_burst_into_ban.y2 = std::min(std::max(result[i].y2+10, event_person_burst_into_ban.y2), channel_data->img_h-1);
            event_person_burst_into_ban.score = event_person_burst_into_ban.score/person_num + result[i].score/(person_num+1);
            worker_cof_sum += result[i].feature[0];
            person_num += 1;
            continue;
        }
    }
    event_person_burst_into_ban.feature = std::vector<float>{worker_cof_sum / person_num};

    //整理检测数据
    auto filter_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    auto &filter_result = filter_output->data.detect;
    for (int i=0; i<result.size(); i++)
    {
        if (std::isnan(result[i].score)) continue;

        if (result[i].class_id == 1 || result[i].class_id == 3) {
            Result_Detect res;
            res.x1         = result[i].x1;
            res.y1         = result[i].y1;
            res.x2         = result[i].x2;
            res.y2         = result[i].y2;
            res.score      = result[i].score;
            res.class_id   = result[i].class_id;
            res.tag        = result[i].tag;
            res.region_idx = result[i].region_idx;
            filter_result.push_back(res);
        }
    }

    float engineering_worker_thresh;
    bool load_res_ = true;
    load_res_ &= module_cfg->get_float("engineering_worker_thresh", engineering_worker_thresh);
    if (load_res_ == false) {
        throw Alg_Module_Exception("Error:\t load module param failed",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }


    for (int i=0; i<result.size(); i++)
    {
        if (std::isnan(result[i].score)) continue;

        if(result[i].class_id == 0 && event_person_burst_into_ban.feature[0] > engineering_worker_thresh){
            break;
        }

        
        if (result[i].class_id == 0 && person_num > 0) {
            Ext_Result res_person_num;
            res_person_num.score = event_person_burst_into_ban.score;
            res_person_num.class_id = person_num;
            res_person_num.tag = "行人数量";
            event_person_burst_into_ban.ext_result["person_num"] = res_person_num;

            Ext_Result res_worker_conf;
            res_worker_conf.score = event_person_burst_into_ban.feature[0];
            res_worker_conf.class_id = -1;
            res_worker_conf.tag = "施工工人置信度";
            event_person_burst_into_ban.ext_result["worker_conf"] = res_worker_conf;

            filter_result.push_back(event_person_burst_into_ban);
            break;
        }
    }
/*
    for (int i=0; i<forward_result.size(); i++)
    {
        // if (std::isnan(forward_result[i].score)) continue;

        if (forward_result[i].class_id == 0) {
            // std::cout << "行人" << std::endl;
            Result_Detect res;
            res.x1         = forward_result[i].x1;
            res.y1         = forward_result[i].y1;
            res.x2         = forward_result[i].x2;
            res.y2         = forward_result[i].y2;
            res.score      = forward_result[i].score;
            res.class_id   = -1;
            res.tag        = forward_result[i].tag;
            res.region_idx = forward_result[i].region_idx;
            filter_result.push_back(res);
        }
    }*/

    output.clear();
    output["result"] = filter_output;
    return true;
};
bool Alg_Module_Burst_Into_Ban_Detection::display(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::map<std::string, std::shared_ptr<InputOutput>>& filter_output)
{
    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t can't find \"image\" in display.input",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }
    if (filter_output.find("result") == input.end()) {
        throw Alg_Module_Exception("Error:\t can't find \"result\" in display.filter_output",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    std::shared_ptr<Module_cfg_Burst_Into_Ban_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Burst_Into_Ban_Detection>(this->get_module_cfg());             //获取模型配置，派生的模型配置文件类指针需要手动转换为子类

    //加载目标框相关参数
    bool load_res = true;
    int box_color_blue;
    int box_color_green;
    int box_color_red;
    int box_thickness;
    load_res &= module_cfg->get_int("box_color_blue", box_color_blue);
    load_res &= module_cfg->get_int("box_color_green", box_color_green);
    load_res &= module_cfg->get_int("box_color_red", box_color_red);
    load_res &= module_cfg->get_int("box_thickness", box_thickness);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t somethine wrong when load param in display",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    //获取图片
    cv::Mat image;
    if (input["image"]->data_type == InputOutput::Type::Image_t) {
        image=input["image"]->data.image->get_image();
    }
    else {
        //暂时不支持其他类型的图像
        std::cout << "Error:\t image input type error" << std::endl;
        throw Alg_Module_Exception("image input type error",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    //将图片放置到 Result_Detect.res_images 中
    std::vector<Result_Detect> &results = filter_output["result"]->data.detect;
    
    if (results.size() == 0) return true;

    int person_idx = -1; //行人闯入事件的索引
    cv::Mat image_copy_person = image.clone();
    for (int i = 0; i < results.size(); i++)
    {
        if (results[i].class_id == -1) {
            //要素
            int x = results[i].x1;
            int y = results[i].y1;
            int w = results[i].x2 - results[i].x1;
            int h = results[i].y2 - results[i].y1;
            
            if (this->debug == 1 || this->debug == 11 || this->debug == 12) {
                cv::Rect box(x, y, w, h);
                cv::rectangle(image_copy_person, box, cv::Scalar(0, 0, 255), 1);     
            }
            if (this->debug == 1 || this->debug == 12) {
                cv::Point point(x, y);
                cv::putText(image_copy_person, std::to_string(results[i].score), point, 1, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_8);
            } 
        }

        if (results[i].class_id == 0) {
            //行人
            int x = results[i].x1;
            int y = results[i].y1;
            int w = results[i].x2 - results[i].x1;
            int h = results[i].y2 - results[i].y1;
            cv::Rect box(x, y, w, h);
            cv::rectangle(image_copy_person, box, cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);
            person_idx = i;
        } 
        
        if (results[i].class_id == 1 || results[i].class_id == 3) {
            //自行车, 摩托车
            cv::Mat image_copy = image.clone();
            int x = results[i].x1;
            int y = results[i].y1;
            int w = results[i].x2 - results[i].x1;
            int h = results[i].y2 - results[i].y1;
            cv::Rect box(x, y, w, h);
            cv::rectangle(image_copy, box, cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);
            std::pair<std::string,cv::Mat> res_image = {"image", image_copy};
            results[i].res_images.insert(res_image);
        }
    }

    if (person_idx >= 0) {
        std::pair<std::string,cv::Mat> res_image = {"image", image_copy_person};
        results[person_idx].res_images.insert(res_image);
    }

    //删除作为元素的事件结果
    for (auto itr = results.begin(); itr != results.end(); ) {
        if ((*itr).class_id == -1) {
            itr = results.erase(itr);
        } else {
            itr++;
        }
    }

    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_Burst_Into_Ban_Detection::load_module_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_Burst_Into_Ban_Detection>(this->node_name);
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Model_cfg_base> Alg_Module_Burst_Into_Ban_Detection::load_model_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_Burst_Into_Ban_Detection>();
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Channel_cfg_base> Alg_Module_Burst_Into_Ban_Detection::load_channel_cfg_(std::string channel_name, std::string cfg_path)
{
    auto res = std::make_shared<Channel_cfg_Burst_Into_Ban_Detection>(channel_name);
    if (access(cfg_path.c_str(), F_OK) != 0) {
        //文件不存在
        throw Alg_Module_Exception("Error:\t channel cfg " + cfg_path + " is not exist", this->node_name, Alg_Module_Exception::Stage::load_channel);
    }
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Channel_data_base> Alg_Module_Burst_Into_Ban_Detection::init_channal_data_(std::string channel_name)
{
    auto res = std::make_shared<Channel_data_Burst_Into_Ban_Detection>(channel_name);

    //根据通道配置参数重置通道数据
    std::shared_ptr<Channel_cfg_Burst_Into_Ban_Detection> channel_cfg = std::dynamic_pointer_cast<Channel_cfg_Burst_Into_Ban_Detection>(this->get_channel_cfg(channel_name));
    res->set_boundarys(channel_cfg->copy_bounary());

    //根据模块配置参数重置通道数据
    std::shared_ptr<Module_cfg_Burst_Into_Ban_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Burst_Into_Ban_Detection>(this->get_module_cfg());

    int min_interval;
    int max_interval;
    float accept_sim_thres;
    float trigger_sim_thres;
    float iou_thres;
    float man_non_motor_thresh_iou;
    float man_thresh_score;
    float non_motor_thresh_score;
    float check_sensitivity_thres;

    module_cfg->get_int("min_interval", min_interval);
    module_cfg->get_int("max_interval", max_interval);
    module_cfg->get_float("accept_sim_thres", accept_sim_thres);
    module_cfg->get_float("trigger_sim_thres", trigger_sim_thres);
    module_cfg->get_float("iou_thres", iou_thres);
    module_cfg->get_float("man_non_motor_thresh_iou", man_non_motor_thresh_iou);
    module_cfg->get_float("man_thresh_score", man_thresh_score);
    module_cfg->get_float("non_motor_thresh_score", non_motor_thresh_score);
    module_cfg->get_float("check_sensitivity_thres", check_sensitivity_thres);

    res->remover.set_trigger_sim_thres(trigger_sim_thres);
    res->remover.set_min_interval(min_interval);
    res->remover.set_max_interval(max_interval);
    res->remover.set_accept_sim_thres(accept_sim_thres);
    res->remover.set_iou_thres(iou_thres);
    res->man_non_motor_thresh_iou = man_non_motor_thresh_iou;
    res->man_thresh_score = man_thresh_score;
    res->non_motor_thresh_score = non_motor_thresh_score;
    res->check_sensitivity_thres = check_sensitivity_thres;

    return res;
};

extern "C" Alg_Module_Base *create()        //外部调用的构造函数
{
    return new Alg_Module_Burst_Into_Ban_Detection();                     //返回当前算法模块子类的指针
};
extern "C" void destory(Alg_Module_Base *p) //外部调用的析构函数
{
    delete p;
};

