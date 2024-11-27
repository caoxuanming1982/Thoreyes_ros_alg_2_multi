#ifndef __TR_ALG_ENGINE_BM_H__
#define __TR_ALG_ENGINE_BM_H__

#include "tr_alg_engine/tr_alg_engine.h"
#include <iostream>

class Tr_Alg_Engine_module_bm:public Tr_Alg_Engine_module{
public:
    Tr_Alg_Engine_module_bm();

    virtual ~Tr_Alg_Engine_module_bm();
    virtual bool init_(std::string submodule_dir,std::string requirement_dir);

};

#endif