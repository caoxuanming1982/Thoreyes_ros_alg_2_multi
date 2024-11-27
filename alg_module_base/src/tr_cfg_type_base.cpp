#include "tr_cfg_type_base.h"
#include <mutex>
Channel_data_base::Channel_data_base(std::string channel_name)
{
    this->channel_name = channel_name;
};
Channel_data_base::~Channel_data_base(){

};

Channel_cfg_base::Channel_cfg_base(std::string channel_name)
{
    this->channel_name = channel_name;
};
Channel_cfg_base::~Channel_cfg_base(){

};
int Channel_cfg_base::from_file(std::string cfg_path)
{
    if (access(cfg_path.c_str(), F_OK) != 0)
        return -1;
    tinyxml2::XMLDocument document;
    std::fstream file;
    file.open(cfg_path, std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return from_string(buffer.str());
};
std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> Channel_cfg_base::get_boundary(std::string name){
    std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> result;
    if(name=="")
        return boundary;
    for(int i=0;i<this->boundary.size();i++){
        if(boundary[i].first==name){
            result.push_back(boundary[i]);
        }        
    }
    return result;
};

int Channel_cfg_base::from_string(std::string cfg_str)
{
    int res=0;
    if(ext_type==Ext_Type::EXTEND){
        res=this->Channel_cfg_base::from_string_(cfg_str);
    }
    if(res<0)
        return res;
    if(typeid(*this).name()!="Channel_cfg_base")
        res=this->from_string_(cfg_str);
    return res;

};

int Channel_cfg_base::from_string_(std::string cfg_str)
{
    boundary.clear();
    tinyxml2::XMLDocument document;
    document.Parse(cfg_str.c_str());
    auto root = document.RootElement();
    auto ptr = root->FirstChildElement("boundary");
    if (ptr != nullptr)
    {
        std::string boundary_string = ptr->GetText();
        std::vector<std::string> lines;
        supersplit(boundary_string, lines, "\n");
        for (int i = 0; i < lines.size(); i++)
        {
            std::string &line = lines[i];
            auto pos = line.find("\r");
            if (pos != std::string::npos)
            {
                line = line.substr(0, pos);
            }
            std::vector<std::string> items;
            supersplit(line, items, ":");
            if (items.size() <= 1)
                continue;
            std::string boundary_name = items[0];

            std::vector<std::string> items1;
            supersplit(items[1], items1, ";");
            std::vector<std::pair<float, float>> points;
            for (int i = 0; i < items1.size(); i++)
            {
                std::vector<std::string> xy_item;
                supersplit(items1[i], xy_item, ",");
                float x = stringToNum<float>(xy_item[0]);
                float y = stringToNum<float>(xy_item[1]);
                points.push_back(std::make_pair(x, y));
            }
            boundary.push_back(std::make_pair(boundary_name, points));
        }
    }

    return 0;
};

std::string Channel_cfg_base::to_string(){
    std::string res="boundary:";
    for(auto iter=this->boundary.begin();iter!=this->boundary.end();iter++){
        res+="\t"+iter->first+"\n";
        for(int i=0;i<iter->second.size();i++){
            res+="\t\t"+ Num2string<float>(iter->second[i].first)+","+Num2string<float>(iter->second[i].second)+"\n";
        }
    }
    return res;
};

Model_cfg_base::Model_cfg_base(){

};
Model_cfg_base::~Model_cfg_base(){

};

int Model_cfg_base::from_string(std::string cfg_str)
{
    int res=0;
    if(ext_type==Ext_Type::EXTEND){
        res=this->Model_cfg_base::from_string_(cfg_str);
    }

    if(res<0)
        return res;
    if(typeid(*this).name()!="Model_cfg_base")
        res=this->from_string_(cfg_str);
    return res;

};

std::vector<Shape_t> string2shape_vector(std::string str){
    str=remove_space(str);
    std::vector<std::string> strings;
    supersplit(str,strings,";");
    std::vector<Shape_t> res;
    for(int i=0;i<strings.size();i++){
        Shape_t shape;
        std::vector<std::string> items;
        supersplit(strings[i],items,",");
        int cnt=0;
        for(int j=0;j<items.size();j++){
            if(items[j]!=""){
                shape.dims[cnt]=stringToNum<int>(items[j]);
                cnt+=1;
            }
        }
        shape.num_dims=cnt;
        res.push_back(shape);
    }
    return res;
}

int Model_cfg_base::from_string_(std::string cfg_str)
{
    tinyxml2::XMLDocument document;
    document.Parse(cfg_str.c_str());
    auto root = document.RootElement();
    auto ptr = root->FirstChildElement("resource_usage");
    if (ptr != nullptr)
    {
        auto ptr1 = ptr->FirstChildElement("memory");
        if (ptr1 != nullptr)
        {
            this->mem_require_mbyte = stringToNum<int>(ptr1->GetText());
        }
        ptr1 = ptr->FirstChildElement("tpu_memory");
        if (ptr1 != nullptr)
        {
            this->tpu_mem_require_mbyte = stringToNum<int>(ptr1->GetText());
        }
        ptr1 = ptr->FirstChildElement("cpu_util");
        if (ptr1 != nullptr)
        {
            this->cpu_util_require = stringToNum<int>(ptr1->GetText());
        }
        ptr1 = ptr->FirstChildElement("tpu_util");
        if (ptr1 != nullptr)
        {
            this->tpu_util_require = stringToNum<int>(ptr1->GetText());
        }
    }
        ptr = root->FirstChildElement("input_shapes");
        if (ptr != nullptr)
        {
            this->input_shapes=string2shape_vector(ptr->GetText());
        
   
        }

    return 0;
}
int Model_cfg_base::from_file(std::string cfg_path)
{
    if (access(cfg_path.c_str(), F_OK) != 0)
        return -1;
    tinyxml2::XMLDocument document;
    std::fstream file;
    file.open(cfg_path, std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return from_string(buffer.str());
};
std::string Model_cfg_base::to_string()
{
    std::string res;
    res += "mem_require_mbyte:" + Num2string<int>(this->mem_require_mbyte);
    res += "\ttpu_mem_require_mbyte:" + Num2string<int>(this->tpu_mem_require_mbyte);
    res += "\tcpu_util_require:" + Num2string<int>(this->cpu_util_require);
    res += "\ttpu_util_require:" + Num2string<int>(this->tpu_util_require);
    return res;
};

Module_cfg_base::Module_cfg_base(std::string module_name)
{
    this->module_name = module_name;
    this->input_output_cfg=std::make_shared<InputOutput_cfg>();
    this->publish_cfg=std::make_shared<Publish_cfg>();
    this->post_process_cfg=std::make_shared<Post_process_cfg_base>();
};
Module_cfg_base::~Module_cfg_base(){

};
int Module_cfg_base::from_string(std::string cfg_str)
{
    int res=0;
    if(ext_type==Ext_Type::EXTEND){
        res=this->Module_cfg_base::from_string_(cfg_str);
    }
    if(res<0)
        return res;
    if(typeid(*this).name()!="Module_cfg_base")
        res=this->from_string_(cfg_str);
    return res;
};

std::string Input_cfg_item::to_string(){
		return "type:"+InputOutput::type2string(data_type)+"\t required_from_module:"+required_from_module+"\t required_output_name:"+required_from_module_output_name;
	};

Input_cfg_item::Input_cfg_item(){
		required_from_module="";
	    required_from_module_output_name="";
        data_type=InputOutput::Type::UNKNOWN;

	};
Input_cfg_item::~Input_cfg_item(){
	};


int Module_cfg_base::from_string_(std::string cfg_str)
{
    cfg_float.clear();
    cfg_int.clear();
    cfg_string.clear();
    input_output_cfg->input_cfgs.clear();
    input_output_cfg->output_cfgs.clear();
    publish_cfg->filter_publish_cfg.clear();
    publish_cfg->raw_publish_cfg.clear();

    tinyxml2::XMLDocument document;
    document.Parse(cfg_str.c_str());
    auto root = document.RootElement();

    auto ptr = root->FirstChildElement("module_name");
    if (ptr != nullptr)
    {
        if(ptr->GetText()!=""){
            this->module_name=ptr->GetText();
        }
    }

    ptr = root->FirstChildElement("float");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            float value = stringToNum<float>(ptr->GetText());
            cfg_float[param_name]= value;
            ptr = ptr->NextSiblingElement();
        }
    }
    ptr = root->FirstChildElement("int");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            float value = stringToNum<int>(ptr->GetText());
            cfg_int[param_name]= value;
            ptr = ptr->NextSiblingElement();
        }
    }

    ptr = root->FirstChildElement("string");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            std::string value = ptr->GetText();
            cfg_string[param_name]= value;
            ptr = ptr->NextSiblingElement();
        }
    }

    ptr = root->FirstChildElement("int_vector");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            std::string value = ptr->GetText();
            std::vector<std::string> value_vector;
            supersplit(value,value_vector,",");
            std::vector<int> res;
            for(int i=0;i<value_vector.size();i++){
                res.push_back(stringToNum<int>(value_vector[i]));
            }
            cfg_int_vector[param_name]= res;

            ptr = ptr->NextSiblingElement();
        }
    }

    ptr = root->FirstChildElement("float_vector");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            std::string value = ptr->GetText();
            std::vector<std::string> value_vector;
            supersplit(value,value_vector,",");
            std::vector<float> res;
            for(int i=0;i<value_vector.size();i++){
                res.push_back(stringToNum<float>(value_vector[i]));
            }
            cfg_float_vector[param_name]= res;

            ptr = ptr->NextSiblingElement();
        }
    }
    ptr = root->FirstChildElement("input_cfg");
    int res=0;
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {

            ptr = ptr->NextSiblingElement();
        }
    }

    ptr = root->FirstChildElement("input_cfg");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            Input_cfg_item item;
            std::string param_name(ptr->Name());{
            auto attr_ptr=ptr->FindAttribute("type");
            if(attr_ptr!=nullptr){
                item.data_type=InputOutput::string2type(attr_ptr->Value());
                if(item.data_type==InputOutput::Type::UNKNOWN){
                    std::cout<<"Error: \t input param "<<param_name<<" is wrong current value is "<<attr_ptr->Value() <<std::endl;      
                    res=-1;              
                    ptr = ptr->NextSiblingElement();
                    continue;
                }
            }

            }
            {
            auto attr_ptr=ptr->FindAttribute("from_module");
            if(attr_ptr!=nullptr){
                item.required_from_module=attr_ptr->Value();
            }

            }
            {
            auto attr_ptr=ptr->FindAttribute("from_module_output_name");
            if(attr_ptr!=nullptr){
                item.required_from_module_output_name=attr_ptr->Value();                
            }

            }
            ptr = ptr->NextSiblingElement();
            this->input_output_cfg->input_cfgs[param_name]=item;
        }
    }
    else{
        std::cout<<"Warning: \t no input cfg"<<std::endl;
    }

    ptr = root->FirstChildElement("output_cfg");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());

            auto attr_ptr=ptr->FindAttribute("type");

            if(attr_ptr!=nullptr){
                auto data_type=InputOutput::string2type(attr_ptr->Value());
                if(data_type==InputOutput::Type::UNKNOWN){
                    std::cout<<"Error: \t input param "<<param_name<<" is wrong current value is "<<attr_ptr->Value() <<std::endl;                    
                    res=-1;
                }
                else{
                    this->input_output_cfg->output_cfgs[param_name]=data_type;
                }
            }
            ptr = ptr->NextSiblingElement();

        }
    }
    else{
        std::cout<<"Warning: \t no output cfg"<<std::endl;
    }

    ptr = root->FirstChildElement("publish_cfg_raw");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            Publish_cfg_item item;
            item.output_result_name=param_name;
            {
                auto attr_ptr=ptr->FindAttribute("topic_name");
                if(attr_ptr!=nullptr){
                    item.topic_name=attr_ptr->Value();
                }

            }
            {
                auto attr_ptr=ptr->FindAttribute("need_publish");
                if(attr_ptr!=nullptr){
                    item.need_publish=attr_ptr->BoolValue();
                }
                else{
                    item.need_publish=false;
                }

            }
            this->publish_cfg->raw_publish_cfg.push_back(item);
            ptr = ptr->NextSiblingElement();

        }
    }
    else{
        std::cout<<"Warning: \t no raw publish cfg"<<std::endl;
    }

    ptr = root->FirstChildElement("publish_cfg_filter");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string param_name(ptr->Name());
            Publish_cfg_item item;
            item.output_result_name=param_name;
            {
                auto attr_ptr=ptr->FindAttribute("topic_name");
                if(attr_ptr!=nullptr){
                    item.topic_name=attr_ptr->Value();
                }

            }
            {
                auto attr_ptr=ptr->FindAttribute("need_publish");
                if(attr_ptr!=nullptr){
                    item.need_publish=attr_ptr->BoolValue();
                }
                else{
                    item.need_publish=false;
                }

            }
            this->publish_cfg->filter_publish_cfg.push_back(item);
            ptr = ptr->NextSiblingElement();

        }
    }
    else{
        std::cout<<"Warning: \t no filter publish cfg"<<std::endl;
    }

    ptr = root->FirstChildElement("post_process_cfg");
    if (ptr != nullptr)
    {
        ptr = ptr->FirstChildElement();
        while (ptr != nullptr)
        {
            std::string post_module_name(ptr->Name());
            Post_process_cfg_item item(post_module_name);
            auto ptr1=ptr->FirstChildElement("input_map");
            if (ptr1 != nullptr)
            {
                ptr1=ptr1->FirstChildElement();
                while (ptr1 != nullptr)
                {
                    auto ptr3=ptr1->FindAttribute("output_name");
                    auto ptr4=ptr1->FindAttribute("module_input_name");
                    if(ptr3!=nullptr&&ptr4!=nullptr){
                        item.map_input[ptr4->Value()]=ptr3->Value();
                    }
                    ptr1 = ptr1->NextSiblingElement();

                }
            }

            auto ptr2=ptr->FirstChildElement("output_map");
            if (ptr2 != nullptr)
            {
                ptr2=ptr2->FirstChildElement();
                while (ptr2 != nullptr)
                {
                    auto ptr3=ptr2->FindAttribute("module_output_name");
                    auto ptr4=ptr2->FindAttribute("output_name");
                    if(ptr3!=nullptr&&ptr4!=nullptr){
                        item.map_output[ptr3->Value()]=ptr4->Value();
                    }
                    ptr2 = ptr2->NextSiblingElement();

                }

            }
            ptr = ptr->NextSiblingElement();
            post_process_cfg->post_process_cfgs.push_back(item);
        }
    }   
    else{
        std::cout<<"info: \t no post process module cfg"<<std::endl;
    }

    return res;
};

int Module_cfg_base::from_file(std::string cfg_path)
{
    if (access(cfg_path.c_str(), F_OK) != 0)
        return -1;

    std::fstream file;
    file.open(cfg_path, std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return from_string(buffer.str());
};

std::shared_ptr<InputOutput_cfg> Module_cfg_base::get_input_output_cfg(){
    return this->input_output_cfg;
};

bool Module_cfg_base::get_float(std::string element_name, float &res)
{
    if (cfg_float.find(element_name) == cfg_float.end())
        return false;
    res = cfg_float[element_name];
    return true;
};
bool Module_cfg_base::get_int(std::string element_name, int &res)
{
    if (cfg_int.find(element_name) == cfg_int.end())
        return false;
    res = cfg_int[element_name];
    return true;
};
bool Module_cfg_base::get_string(std::string element_name, std::string &res)
{
    if (cfg_string.find(element_name) == cfg_string.end())
        return false;
    res = cfg_string[element_name];
    return true;
};
bool Module_cfg_base::get_float_vector(std::string element_name,std::vector<float>& res){
    if (cfg_float_vector.find(element_name) == cfg_float_vector.end())
        return false;
    res = cfg_float_vector[element_name];
    return true;

};
bool Module_cfg_base::get_int_vector(std::string element_name,std::vector<int>& res){
    if (cfg_int_vector.find(element_name) == cfg_int_vector.end())
        return false;
    res = cfg_int_vector[element_name];
    return true;
    
};
std::shared_ptr<Publish_cfg> Module_cfg_base::get_publish_cfg(){
    return this->publish_cfg;
};

std::shared_ptr<Post_process_cfg_base> Module_cfg_base::get_post_process_cfg(){
    return this->post_process_cfg;

};



std::string Module_cfg_base::to_string()
{
    std::string res;
    res += "Module name:"+this->module_name+"\n";
    res += "float parameter:\n";
    for (auto iter = this->cfg_float.begin(); iter != cfg_float.end(); iter++)
    {
        res += "\t" + iter->first + ":" + Num2string<float>(iter->second) + "\n";
    }

    res += "int parameter:\n";
    for (auto iter = this->cfg_int.begin(); iter != cfg_int.end(); iter++)
    {
        res += "\t" + iter->first + ":" + Num2string<int>(iter->second) + "\n";
    }

    res += "string parameter:\n";
    for (auto iter = this->cfg_string.begin(); iter != cfg_string.end(); iter++)
    {
        res += "\t" + iter->first + ":" + iter->second + "\n";
    }

    res += "int vector parameter:\n";
    for (auto iter = this->cfg_int_vector.begin(); iter != cfg_int_vector.end(); iter++)
    {
        res += "\t" + iter->first + ":";
        for(int i=0;i<iter->second.size();i++){
            res+=Num2string<int>(iter->second[i])+",";
        }
        
        res += "\n";
    }

    res += "float vector parameter:\n";
    for (auto iter = this->cfg_float_vector.begin(); iter != cfg_float_vector.end(); iter++)
    {
        res += "\t" + iter->first + ":";
        for(int i=0;i<iter->second.size();i++){
            res+=Num2string<int>(iter->second[i])+",";
        }
        
        res += "\n";
    }
    res+=input_output_cfg->to_string()+"\n";
    res+=publish_cfg->to_string()+"\n";
    res+=post_process_cfg->to_string()+"\n";
    return res;
};
std::string Module_cfg_base::get_module_name(){
    return this->module_name;
};

