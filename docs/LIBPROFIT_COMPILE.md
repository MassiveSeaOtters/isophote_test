
# Compile `libprofit` on MacOSX with OpenMP support 

```
brew install libomp llvm fftw gsl
```

```
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \                                                     ─╯
-DOpenMP_CXX_FLAGS=-fopenmp=lomp \
-DOpenMP_CXX_LIB_NAMES="libomp" \
-DOpenMP_libomp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
-DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /opt/homebrew/opt/libomp/lib/libomp.dylib -I/opt/homebrew/opt/libomp/include" \
-DOpenMP_CXX_LIB_NAMES="libomp"
```
