#include "../include/helpers.h"

const std::vector<unsigned int> num2ditvec(unsigned int x, unsigned int d, unsigned int N)
{
    assert(x<std::pow(d,N));
    std::vector<unsigned int> v;
    v.reserve(N);
    unsigned int div;
    for (unsigned int i=N-1; i>0; --i)
    {
        div=std::pow(d,i);
        v.push_back(x/div);
        x=x%div;
    }
    v.push_back(x);
    return v;
}

unsigned int ditvec2num(const std::vector<unsigned int>& vec,unsigned int d)
{
    unsigned int x=0,i=0;
    std::vector<unsigned int>::const_reverse_iterator it;
    for (it=vec.rbegin(); it!=vec.rend(); ++it)
    {
        x+=(*it)*std::pow(d,i++);
    }
    return x;
}


unsigned int gcd(unsigned int u, unsigned int v)
{
    while ( v != 0) {
        unsigned int r = u % v;
        u = v;
        v = r;
    }
    return u;
}

std::ostream& operator<<(std::ostream& os, const arma::span& sp)
{
    os<<"("<<sp.a<<"-"<<sp.b<<")";
    return os;
}

bool
all(const std::vector<bool>& X)
{
    bool out = true;
    /// as soon as we encounter a false entry, we can return false
    for (const auto& it : X) if (!it) return false;
    return out;
}

bool
any(const std::vector<bool>& X)
{
    bool out = false;
    /// as soon as we encounter a true entry, we can return true
    for (const auto& it : X) if (it) return true;
    return out;
}

/// for these, take a look at
/// http://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/stat.h.html
bool
FileExist(const std::string& filename) /// checks whether filename is
{
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}

bool
RegFileExist(const std::string& filename)
{
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0 && S_ISREG(buf.st_mode));
}

bool
FolderExist(const std::string& foldername, bool create)
{
    struct stat buf;
    bool exist = stat(foldername.c_str(), &buf) == 0 && S_ISDIR(buf.st_mode);
    /// if the folder doesn't exist and we would like to create it, do so
//    if (create && !exist) exist = (mkdir(foldername.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == 0); /// S_IRWXU, S_IRWXG, S_IRWXO set read/write/execute&search permissions for owner,group and others
    if (create && !exist) exist = (mkdir(foldername) == 0); /// S_IRWXU, S_IRWXG, S_IRWXO set read/write/execute&search permissions for owner,group and others
    return exist;
}


fileparts::fileparts(const std::string& filename)
{
//    cout<<"filename: "<<filename<<endl;
    size_t idot = filename.find_last_of('.');
    size_t islash = filename.find_last_of('/');

    if (islash == std::string::npos) /// without folder
    {
        if (idot == std::string::npos) this->name = filename; /// file name without folder and ending
        else /// file name with ending but without folder
        {
            this->name = filename.substr(0,idot);
            this->ext = filename.substr(idot+1);
        }
    }
    else /// with folder
    {
        this->path = filename.substr(0,islash);
        if (idot == std::string::npos) this->name = filename.substr(islash+1); /// file name with folder but without ending
        else /// file name with folder and ending
        {
            this->name = filename.substr(islash+1,idot-islash-1);
            this->ext = filename.substr(idot+1);
        }

    }
}


std::string
Fullpath(const std::string& filename, const std::string& ending, const std::string& folder)
{
   return folder+"/"+filename+"."+ending;
}

bool mkdir(const std::string& path)
{
    return (mkdir(path.c_str(),S_IRWXU | S_IRWXG | S_IRWXO) == 0);
}

std::string
GetUniquePath(const std::string& path)
{
    std::string tmpname(path);
    unsigned int ct=1;
    while (FileExist(tmpname)) tmpname = path+"_"+std::to_string(ct++);
    return tmpname;
}

std::string
GetUniqueFileName(const std::string& filename, const std::string& ending, const std::string& folder)
{
//    std::string tmpname = folder+"/"+filename+"."+ending;
    std::string tmpname = Fullpath(filename,ending,folder);
    unsigned int ct=1;
    while (RegFileExist(tmpname)) tmpname = folder+"/"+filename+"_"+std::to_string(ct++)+"."+ending;
    return tmpname;
}
