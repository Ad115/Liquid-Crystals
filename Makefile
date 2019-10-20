.PHONY: compile test clean

compile:
	nvcc gpu_main.cu -std=c++11 -arch=sm_75 --expt-extended-lambda -o gpu_main.out -I.

run: compile
	./gpu_main.out

compile_cpu:
	g++ cpu_main.cpp -std=c++11  -o PartiCuditas.out

run_cpu: compile_cpu
	./PartiCuditas.out

test: compile # "It compiles" is a good first test
	nvcc tests_main.cu -std=c++11 -arch=sm_75 --expt-extended-lambda -o tests.out -I. && \
	./tests.out

test_pcuditas:
	nvcc pcuditas_test.cu -std=c++11 -arch=sm_75 --expt-extended-lambda -o tests.out -I. && \
	./tests.out

clean:
	rm *.out *.o *.xyz --verbose -f