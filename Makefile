.PHONY: compile

compile:
	nvcc main_gpu.cu -std=c++11 -arch=sm_75 --expt-extended-lambda -o main_gpu.out