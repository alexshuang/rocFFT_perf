#!/bin/sh

SRC=$1
BASENAME=${SRC%%.*}
OUT=${BASENAME}.py
cp $SRC $OUT

sed -i 's/^__global__ void\s*\(\w*\)(/def \1(grid, threads, empty, stream, /g' $OUT
sed -i 's/^__device__\s*/def /g' $OUT
sed -i 's/hipLaunchKernelGGL.*(\(.*\)<.*>.*/\1(/g' $OUT
sed -i '/^template\s*</d' $OUT
sed -i '/^\s*{\s*$/d' $OUT
sed -i '/^\s*}\s*$/d' $OUT
sed -i 's/sizeof(float)/4/g' $OUT
sed -i 's/sizeof(int)/4/g' $OUT
sed -i 's/sizeof(double)/8/g' $OUT
sed -i 's/hipStream_t\s*//g' $OUT
sed -i 's/\<void\>\s*\**\s*//g' $OUT
sed -i 's/\<int\>\s*\**\s*//g' $OUT
sed -i 's/\<const\>\s*\**\s*//g' $OUT
sed -i 's/(size_t)\s*//g' $OUT
sed -i 's/\<size_t\>\s*\**\s*//g' $OUT
sed -i 's/lib_make_vector2<T>//g' $OUT
sed -i 's/\<T\>\s*\**\s*//g' $OUT
sed -i 's/\<float\>\s*\**\s*//g' $OUT
sed -i 's/\<double\>\s*\**\s*//g' $OUT
sed -i 's/else if\s*(\(.*\))/elif \1:/g' $OUT
sed -i 's/if\s*(\(.*\))/if \1:/g' $OUT
sed -i 's/^\(\s*\)else\(\s*\)$/\1else:\2/g' $OUT
sed -i 's/\<rocfft_status_success\>/0/g' $OUT
sed -i 's/data->node->//g' $OUT
sed -i 's/\<dim3(\(.*\))/\1/g' $OUT
sed -i 's/\<dim3\>\s*\(\w*\)(\(.*\))/\1 = (\2)/g' $OUT
sed -i 's/\<real_type_t\><T*>\s*\**\s*//g' $OUT
sed -i 's/\.x//g' $OUT
sed -i 's/\.y//g' $OUT
sed -i 's;//;#//;g' $OUT
sed -i 's/devKernArg\.data() + 1 \* KERN_ARGS_ARRAY_WIDTH/inStride/g' $OUT
sed -i 's/devKernArg\.data() + 2 \* KERN_ARGS_ARRAY_WIDTH/outStride/g' $OUT
sed -i 's/devKernArg\.data(),/length,/g' $OUT
