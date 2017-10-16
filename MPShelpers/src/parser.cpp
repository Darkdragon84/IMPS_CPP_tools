#include "../include/parser.h"


void parser::parse(int argc, char** argv)
{
    std::vector<string> vfiles;
    std::vector<string> vargs;
    string tmp;
    this->parse_from_file("config.cfg");//always use std config file
    if (argc>1)
    {
        for (int i=1;i<argc;++i)
        {
            tmp=string(argv[i]);
            if (is_cfg_file(tmp)) vfiles.push_back(tmp);
            else vargs.push_back(tmp);
        }
        for (const auto& fit : vfiles)this->parse_from_file(fit);
        for (const auto& far : vargs)this->parse_argument(far);
    }
}

void parser::parse_argument(const string& arg)
{
    size_t epos = arg.find("=",2); // determine position of equal sign (check below if even present)
    bool use = arg.substr(0,2)=="--" && epos!=string::npos && epos>2; // check if argument starts with --, if there is a = and if there at least one character between -- and =

    if (use) values_[arg.substr(2,epos-2)]=arg.substr(epos+1,arg.length()); // if suitable, use stuff between -- and = as key and everything after as target
    else cout<<"ignoring argument "<<arg<<endl;
}

void parser::parse_from_file(const string& filename)
{
    std::ifstream file(filename);
    string line;
    if (file.good())
    {
        while(getline(file,line)) parse_argument(line);
    }
    else cerr<<"file "<<filename<<" not found"<<endl;
}

inline bool parser::is_cfg_file(string arg)
{
    return arg.substr(arg.length()-4,arg.length()-1)==".cfg";
}

void parser::AddValue(const string& key, const string& value)
{
    auto res = values_.emplace(key,value);
    if (!res.second) cerr<<key<<" = "<<res.first->second<<" already present, "<<value<<" not inserted"<<endl;
}

void parser::PrintValues() const
{
    for (const auto it : values_) cout<<it.first<<": "<<it.second<<endl;
}

template<> bool parser::sconv<bool>(const string& str) const {return stoi(str);}
template<> int parser::sconv<int>(const string& str) const {return stoi(str);}
template<> unsigned int parser::sconv<unsigned int>(const string& str) const {return stoul(str);}
template<> unsigned long long parser::sconv<unsigned long long>(const string& str) const {return stoul(str);}
template<> double parser::sconv<double>(const string& str) const {return stod(str);}
template<> size_t parser::sconv<size_t>(const string& str) const {return stoul(str);}
template<> string parser::sconv<string>(const string& str) const {return string(str);}


