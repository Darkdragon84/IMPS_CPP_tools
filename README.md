# IMPS_CPP_tools
C++ Implementations of IMPS algorithms with **abelian** symmetries, for calculating ground states 
and elementary excitations for infinite one-dimensional quantum lattices. For now, only a certain collection of popular models with nearest neighbor interactions are implemented and supported. The possibility to define general user-defined models and MPO implementations are features to come.

When using this code, please cite the following open access articles</br>
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145</br>
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.235155

## Prerequisites
- armadillo library for fast matrix wrappers and linear algebra http://arma.sourceforge.net/
- ARPACK library for iterative eigensolvers https://github.com/opencollab/arpack-ng  
beware of this [bug](https://forge.scilab.org/index.php/p/arpack-ng/issues/1315/), for which a workaround is available [here](https://gist.github.com/Darkdragon84/6728023#file-dneupd2-f)

## Installation
Installation is CMake based. First, build the helper libraries in MPShelpers, then VUMPS and Excitations
  
