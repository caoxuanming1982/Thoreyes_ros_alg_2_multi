#include "common.h"
#include <iostream>


template <class Type>  
std::string Num2string(Type Num)
{
	std::ostringstream oss;
	oss<<Num;
	std::string str(oss.str());
	return str;
};
template <class Type>  
Type stringToNum(const std::string& str)  
{  
    std::istringstream iss(str);  
    Type num;  
    iss >> num;  
    return num;      
} 

template std::string Num2string<int>(int Num);
template std::string Num2string<float>(float Num );
template std::string Num2string<double>(double Num);
template std::string Num2string<long>(long Num);

template int stringToNum<int>(const std::string& str);
template float stringToNum<float>(const std::string& str);
template long stringToNum<long>(const std::string& str);
template double stringToNum<double>(const std::string& str);


long get_time(){
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec*1000+tv.tv_usec/1000;
};

void supersplit(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    size_t len = s.length();
    pos2 = s.find(c);
    pos1 = 0;
//    cout<<s<<endl;
    while(std::string::npos != pos2)
    {
        v.emplace_back(s.substr(pos1, pos2-pos1));
 
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != len)
        v.emplace_back(s.substr(pos1));
};


