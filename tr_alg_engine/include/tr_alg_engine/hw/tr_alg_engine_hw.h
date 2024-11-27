#ifndef __TR_ALG_ENGINE_HW_H__
#define __TR_ALG_ENGINE_HW_H__

#include "tr_alg_engine/tr_alg_engine.h"
#include <iostream>

class Tr_Alg_Engine_module_hw:public Tr_Alg_Engine_module{
public:
    Tr_Alg_Engine_module_hw();

    virtual ~Tr_Alg_Engine_module_hw();
    virtual bool init_(std::string submodule_dir,std::string requirement_dir);

};

#endif