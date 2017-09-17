#ifdef SYMS
#define _USE_SYMMETRIES_
#endif // SYMS

#ifdef VERBOSE
#define DOUT(str) do { std::cout << str << std::endl; } while( false )
#else
#define DOUT(str) do { } while ( false )
#endif
