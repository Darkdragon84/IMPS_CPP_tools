#ifndef HELPERS_H_
#define HELPERS_H_

#include <vector>
#include <ostream>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <armadillo>

#include "Defs.h"

const std::vector<unsigned int> num2ditvec(unsigned int x, unsigned int d, unsigned int N);
unsigned int ditvec2num(const std::vector<unsigned int>& vec,unsigned int d);
unsigned int gcd(unsigned int u, unsigned int v); /// greatest common divisor
std::ostream& operator<<(std::ostream& os, const arma::span& sp);

bool all(const std::vector<bool>& X);
bool any(const std::vector<bool>& X);

bool FileExist(const std::string& filename);
bool RegFileExist(const std::string& filename);
bool FolderExist(const std::string& filename, bool create=false);


typedef struct fileparts
{
    fileparts(const std::string& filename);
    std::string path;
    std::string name;
    std::string ext;
} fileparts;

std::string Fullpath(const std::string& filename, const std::string& ending="bin", const std::string& folder=".");
std::string GetUniquePath(const std::string& path);
std::string GetUniqueFileName(const std::string& filename, const std::string& ending="bin", const std::string& folder=".");
bool mkdir(const std::string& path); /// overload standard mkdir() function from <sys/stat.h>
#endif // HELPERS_H_
