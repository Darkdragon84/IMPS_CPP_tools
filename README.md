# IMPS_CPP_tools
C++ Implementations of IMPS algorithms with **abelian** symmetries, for calculating ground states 
and elementary excitations for infinite one-dimensional quantum lattices.

## Prerequisites
- armadillo library for fast matrix wrappers and linear algebra http://arma.sourceforge.net/
- ARPACK library for iterative eigensolvers https://github.com/opencollab/arpack-ng  
beware of this [bug](https://forge.scilab.org/index.php/p/arpack-ng/issues/1315/), for which a workaround is available [here](https://gist.github.com/Darkdragon84/6728023)

## Installation
Installation is CMake based. First, build the helper libraries in MPShelpers, then VUMPS and Excitations
  
