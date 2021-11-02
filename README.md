# cuda_test

[![GitHub license](https://img.shields.io/github/license/guillaumetousignant/NDG_cuda.svg)](https://github.com/guillaumetousignant/NDG_cuda/blob/master/LICENSE) [![GitHub release](https://img.shields.io/github/release-pre/guillaumetousignant/NDG_cuda.svg)](https://GitHub.com/guillaumetousignant/NDG_cuda/releases/) [![Documentation Status](https://readthedocs.org/projects/another-path-tracer/badge/?version=latest)](https://ndg-cuda.readthedocs.io/en/latest/?badge=latest) [![build-ubuntu Actions Status](https://github.com/guillaumetousignant/NDG_cuda/workflows/Ubuntu/badge.svg)](https://github.com/guillaumetousignant/NDG_cuda/actions) [![build-windows Actions Status](https://github.com/guillaumetousignant/NDG_cuda/workflows/Windows/badge.svg)](https://github.com/guillaumetousignant/NDG_cuda/actions) [![build-macos Actions Status](https://github.com/guillaumetousignant/NDG_cuda/workflows/macOS/badge.svg)](https://github.com/guillaumetousignant/NDG_cuda/actions)

Installation:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build .
cmake --install .
```

sm_61 is GTX 10X0, sm_60 is Tesla P100, sm_70 is Tesla V100
