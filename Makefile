.PHONY: compile test clean


run: compile
	./pcuditas_main.out

compile:
	nvcc pcuditas_main.cu -std=c++11 -arch=sm_50 --expt-extended-lambda -o pcuditas_main.out -I.

test: compile # "It compiles" is a good first test
	nvcc pcuditas_test.cu -std=c++11 -arch=sm_50 --expt-extended-lambda -o tests.out -I. && \
	./tests.out

clean:
	rm *.out *.o *.xyz --verbose -f