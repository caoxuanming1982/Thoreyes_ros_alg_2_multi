#ifndef __PUBLISH_CFG_BASE_H__
#define __PUBLISH_CFG_BASE_H__

#include <iostream>
#include<map>
#include<vector>

struct Publish_cfg_item{
    bool need_publish=false;
    std::string topic_name;    
    std::string output_result_name;
};

struct Publish_cfg{
    std::vector<Publish_cfg_item> raw_publish_cfg;
    std::vector<Publish_cfg_item> filter_publish_cfg;
    std::string to_string(){
        std::string res="publish_cfg\n";
        res+="\traw_publish_cfg\n";
        for(auto iter=raw_publish_cfg.begin();iter!=raw_publish_cfg.end();iter++){
            if(iter->need_publish){
                res+="*";
            }
            res+="\t\t"+iter->output_result_name+" to \t"+iter->topic_name+"\n";
        }
        res+="\tfilter_publish_cfg\n";
        return res;
    };
};


#endif