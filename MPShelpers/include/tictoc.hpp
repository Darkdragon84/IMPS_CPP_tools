#ifndef TICTOC_H
#define TICTOC_H

#include <sys/time.h>
#include <ctime>

using std::cout;
using std::cerr;
using std::endl;

class tictoc
{
    public:
        tictoc():running_(false),etime_(0) {}
        void tic()
        {
            if(running_) cerr<<"already running"<<endl;
            else
            {
                etime_=0;
                gettimeofday(&tvs_,NULL);
                running_=true;
            }
        }
        double toc(bool verbose=false)
        {
            if (running_)
            {
                gettimeofday(&tve_,NULL);
                etime_=(tve_.tv_sec - tvs_.tv_sec) + (double)(tve_.tv_usec - tvs_.tv_usec)/(double)1e6;
                running_=false;
                if (verbose)
                {
                    cout<<"elapsed time: ";
                    if (this->etime_ > 60)cout<<this->etime_/60.<<" min."<<endl;
                    else cout<<this->etime_<<" s."<<endl;
                }
                return etime_;
            }
            else
            {
                cerr<<"tictoc not running!"<<endl;
                return 0;
            }
        }
        virtual ~tictoc() {}
    protected:
    private:
        bool running_;
        double etime_;
        struct timeval tvs_,tve_;
};

#endif // TICTOC_H
