TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#nvcc -std=c++11 -c -o lasso_fista.cu.o lasso_fista.cu.cc -I$TF_INC -I$TF_INC/external/nsync/public -I$TF_LIB -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED 

max=2

for i in {1..15}

do
	echo $i
	nvcc -std=c++11 -c -o "lasso_fista_Layer"$i".cu.o" "lasso_fista_Layer"$i".cu.cc" -I$TF_INC -I$TF_INC/external/nsync/public -I$TF_LIB -D GOOGLE_CUDA=1  -L /usr/local/cuda-9.1/lib64/  -x cu -Xcompiler -fPIC   -D_MWAITXINTRIN_H_INCLUDED --expt-relaxed-constexpr

	g++ -std=c++11 -shared -o "lasso_fista_Layer"$i".so" "lasso_fista_Layer"$i".cc" "lasso_fista_Layer"$i".cu.o"  -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0  -I$TF_INC -I$TF_INC/external/nsync/public   -L $TF_LIB -L /usr/local/cuda-9.1/lib64/ -ltensorflow_framework -O2 

done

