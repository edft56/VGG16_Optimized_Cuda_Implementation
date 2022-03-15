TF_CFLAGS = `python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"`
TF_LFLAGS = `python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"`
# might need to also include current environment include directory. ex. CONDA_INCLUDE = -I//home/username/miniconda3/envs/env_name/include

all: vgg16_TF

vgg16_TF: vgg16.o
	g++-10 -std=c++14 -shared vgg16_custom.cc -o vgg16_custom.so -fPIC vgg16.o $(TF_CFLAGS) $(TF_LFLAGS) -O2

vgg16.o:
	nvcc -ccbin=g++-10 --compiler-options='-std=c++14 -O2 -fPIC' -arch=sm_61 -Xptxas="-v -dlcm=ca" -c vgg16.cu

clean:
	rm -f *.o vgg16_custom.so
	
	



	

