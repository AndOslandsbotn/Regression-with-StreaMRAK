# Regression-with-StreaMRAK
This repository contains an implementation of the kernel method
StreaMRAK <a id="1">[1]</a>, which is built on the kernel solver FALKON <a id="1">[2]</a> as a
base solver.  

### Install
To install the code, clone the repository and install the packages in the requirements.txt file
using pip install -r requirements.txt from the root of your repository

### Run
To run the kernel methods on a toy example run the following two scripts
 - main_train_streamrak.py 
 - main_pred_streamrak.py
 - main_train_falkon.py 
 - main_pred_falkon.py

#### References
<a id="1">[1]</a> 
A. Oslandsbotn, Z. Kereta, V. Naumova, Y. Freund and A. Cloninger (2022). 
StreaMRAK a streaming multi-resolution adaptive kernel algorithm
Applied Mathematics and Computation, vol 426, p. 127112
https://doi.org/10.1016/j.amc.2022.127112

<a id="1">[2]</a> 
A. Rudi, L. Carratino and L. Rosasco (2017). 
FALKON: An Optimal Large Scale Kernel Method. 
Advances in Neural Information Processing Systems, vol 30
https://proceedings.neurips.cc/paper/2017/file/05546b0e38ab9175cd905eebcc6ebb76-Paper.pdf
