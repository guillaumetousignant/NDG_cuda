# NDG_cuda

[![GitHub license](https://img.shields.io/github/license/guillaumetousignant/NDG_cuda.svg)](https://github.com/guillaumetousignant/NDG_cuda/blob/master/LICENSE) [![GitHub release](https://img.shields.io/github/release-pre/guillaumetousignant/NDG_cuda.svg)](https://GitHub.com/guillaumetousignant/NDG_cuda/releases/) [![Documentation Status](https://readthedocs.org/projects/another-path-tracer/badge/?version=latest)](https://ndg-cuda.readthedocs.io/en/latest/?badge=latest) [![build-ubuntu Actions Status](https://github.com/guillaumetousignant/NDG_cuda/workflows/Ubuntu/badge.svg)](https://github.com/guillaumetousignant/NDG_cuda/actions) [![build-windows Actions Status](https://github.com/guillaumetousignant/NDG_cuda/workflows/Windows/badge.svg)](https://github.com/guillaumetousignant/NDG_cuda/actions)

Installation:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build .
cmake --install .
```

sm_61 is GTX 10X0, sm_60 is Tesla P100, sm_70 is Tesla V100

Dependencies:

- HDF5
- CGNS
- VTK

To build without cgns and HDF5, add `-DUSE_CGNS=OFF` to cmake call.  

To build tests, add `-DBUILD_TESTING=ON` to cmake call. Then:

```bash
ctest
```

To generate documentation, add `-DBUILD_DOC=ON` to cmake call. Additional dependencies:

- Doxygen
- dot (graphviz)
- Sphinx
- Breathe (pip: breathe)
- Read the Docs Sphinx Theme (pip: sphinx_rtd_theme)

To include in a CMake project:

```bash
find_package(SEM-CUDA 1.0.0 REQUIRED)
target_link_libraries(example SEM-CUDA::SEM-CUDA)
```
