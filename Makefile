.PHONY: compile test clean

COMPILE_FLAGS=-std=c++11 -arch=sm_50 --expt-extended-lambda -I.

run: compile
	./pcuditas_main.out

compile:
	nvcc pcuditas_main.cu -std=c++11 -arch=sm_50 --expt-extended-lambda -o pcuditas_main.out -I.

test: compile # "It compiles" is a good first test
	nvcc pcuditas_test.cu -std=c++11 -arch=sm_50 --expt-extended-lambda -o tests.out -I. && \
	./tests.out

compile_for_benchmark:
	nvcc examples/simplified.cu ${COMPILE_FLAGS}

compile_examples:
	nvcc examples/random_walk.cu ${COMPILE_FLAGS} -I. -o example_random_walk.out
	nvcc examples/simplified.cu ${COMPILE_FLAGS} -I. -o example_simplified.out
	#nvcc examples/no_interactions.cu ${COMPILE_FLAGS} -I. -o example_no_interactions.out

clean:
	rm *.out *.o *.xyz --verbose -f