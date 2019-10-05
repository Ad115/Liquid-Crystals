.PHONY: compile test clean

compile:
	nvcc main_gpu.cu -std=c++11 -arch=sm_75 --expt-extended-lambda -o main_gpu.out -I.

run: compile
	./main_gpu.out

compile_cpu:
	g++ main_cpu.cpp -std=c++11  -o PartiCuditas.out

run_cpu: compile_cpu
	./PartiCuditas.out

test: compile # "It compiles" is a good first test
	nvcc tests.cu -std=c++11 -arch=sm_75 --expt-extended-lambda -o tests.out -I. && \
	./tests.out

clean:
	rm *.out *.o --verbose -f