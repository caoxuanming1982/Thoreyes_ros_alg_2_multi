#ifndef __POST_PROCESS_CFG_BASE_H__
#define __POST_PROCESS_CFG_BASE_H__

#include<vector>
#include<chrono>
#include <shared_mutex>
#include<fstream>
#include<sstream>
#include "common.h"
#include <map>
#include "tinyxml2.h"
#include <shared_mutex>
#include<fstream>
#include<sstream>
#include<unistd.h>

class Post_process_cfg_item{
public:
    std::string module_name;
    std::map<std::string,std::string> map_input;
    std::map<std::string,std::string> map_output;
    Post_process_cfg_item(string name){
        module_name=name;
    }
	std::string to_string(){
        std::string res="\t\t"+module_name+"\n";
        res+="\t\tmap_input:\n";
        for(auto iter=map_input.begin();iter!=map_input.end();iter++){
            res+="\t\t\t"+iter->first+"\t"+iter->second+"\n";
        }

        res+="\t\tmap_output:\n";
        for(auto iter=map_output.begin();iter!=map_output.end();iter++){
            res+="\t\t\t"+iter->first+"\t"+iter->second+"\n";
        }
        return res;
    }
};


class Post_process_cfg_base{
public:
    std::vector<Post_process_cfg_item> post_process_cfgs;
	std::string to_string(){
		std::string res="post_process_cfg:\n";
        for(int i=0;i<post_process_cfgs.size();i++){
            res+=post_process_cfgs[i].to_string();
        }
		return res;
	}


};


#endif