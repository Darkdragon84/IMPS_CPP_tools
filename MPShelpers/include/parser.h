#ifndef PARSER_H
#define PARSER_H

#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

#include "Defs.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::string;

class parser
{
    public:
        parser() = default;
        parser(int argc, char** argv){parse(argc,argv);};
        virtual ~parser() = default;

        void parse(int argc, char** argv);
        void PrintValues() const;
        void AddValue(const string& key, const string& value);
        template<typename T> bool GetValue(T& val, const string& name, bool abrt=false) const;
        template<typename T> bool GetValue(std::vector<T>& val, const string& name, bool abrt=false) const;
        template<typename T> bool GetValue(std::vector<std::vector<T> >& vec, const string& name, bool abrt=false) const;

    protected:
        inline bool is_cfg_file(string arg);
        void parse_from_file(const string& filename);
        void parse_argument(const string& arg);
        std::map<string,string> values_;

        template<typename T> T sconv(const string& str) const;
        template<typename T> void extract_vector(std::vector<T>& vec, string tmp) const;
    private:
};

template<> bool parser::sconv<bool>(const string& str) const;
template<> int parser::sconv<int>(const string& str) const;
template<> unsigned int parser::sconv<unsigned int>(const string& str) const;
template<> unsigned long long parser::sconv<unsigned long long>(const string& str) const;
template<> double parser::sconv<double>(const string& str) const;
template<> size_t parser::sconv<size_t>(const string& str) const;
template<> string parser::sconv<string>(const string& str) const;

template<typename T>
bool
parser::GetValue(T& val, const string& name, bool abrt) const
{
    auto it=values_.find(name); /// look for parameter "name"
    bool found=(it!=values_.end()); /// see if found
    if (found) val=sconv<T>(it->second); /// if found, write value to external variable val
    else
    {
        if (abrt) throw std::invalid_argument(string("argument \"")+name+string("\" not found"));
        else cout<<"argument \""<<name<<"\" not found, using "<<name<<"="<<val<<endl;
    } /// if not found, warn and do nothing to val
    return found;
}


template<typename T>
void
parser::extract_vector(std::vector<T>& vec, string tmp) const
{
    vec.clear();
    size_t len = tmp.size();
    if (tmp.compare(0,1,"[")!=0 || tmp.compare(tmp.size()-1,1,"]")!=0) throw std::invalid_argument("argument must begin with a [ and end in a ]");
    tmp = tmp.substr(1,len-2);

    size_t pos1=0,pos2=0;
    while(pos2!=string::npos)
    {
        pos2 = tmp.find(",",pos1);
        vec.push_back(sconv<T>(tmp.substr(pos1,pos2-pos1)));
        pos1 = pos2 + 1;
    }
}

template<typename T>
bool
parser::GetValue(std::vector<std::vector<T> >& vec, const string& name, bool abrt) const
{
    string tmp;

    auto it=values_.find(name); // look for parameter "name"
    bool found=(it!=values_.end()); // see if found
    if (found)
    {
        vec.clear();
        std::vector<T> subvec;

        string tmp(it->second);
        size_t len = tmp.size();
        if (tmp.compare(0,1,"[")!=0 || tmp.compare(tmp.size()-1,1,"]")!=0) throw std::invalid_argument("argument must begin with a [ and end in a ]");
        tmp = tmp.substr(1,len-2);

        size_t pos1=0,pos2=0;
        while(pos1<len-1)
        {
            subvec.clear();
            pos2 = tmp.find("]",pos1);
            extract_vector(subvec,tmp.substr(pos1,pos2-pos1+1));
            vec.push_back(subvec);
            pos1 = pos2 + 2;
        }
    }
    else
    {
        if (abrt) throw std::invalid_argument(string("argument \"")+name+string("\" not found"));
        else cout<<"argument \""<<name<<"\" not found"<<endl;
    } // if not found, warn and do nothing to val
    return found;
}

template<typename T>
bool
parser::GetValue(std::vector<T>& vec, const string& name, bool abrt) const
{
    string tmp;
    auto it=values_.find(name); // look for parameter "name"
    bool found=(it!=values_.end()); // see if found
    if (found) extract_vector(vec,it->second);
    else
    {
        if (abrt) throw std::invalid_argument(string("argument \"")+name+string("\" not found"));
        else cout<<"argument \""<<name<<"\" not found"<<endl;
    } // if not found, warn and do nothing to val
    return found;
}




#endif // PARSER_H
