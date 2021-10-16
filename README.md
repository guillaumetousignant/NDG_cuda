# cuda_test

Installation:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build .
cmake --install .
```

sm_61 is GTX 10X0, sm_60 is Tesla P100, sm_70 is Tesla V100
