DIMENSIONS = 3 ## 2D or 3D?

CC=gcc
LDLIBS = -lm -g -Ofast -D DIMENSIONS=$(DIMENSIONS)#-pg
CFLAGS = -I. -g -Ofast -D DIMENSIONS=$(DIMENSIONS) #-pg

OUT = main.out
DEPS = simulation.h particle_system.h random.h configuration.h string_utils.h args_parse.h
OBJ = simulation.o particle_system.o random.o configuration.o string_utils.o args_parse.o


$(OBJ): %.o: %.c %.h

main: main.c $(OBJ) $(DEPS)
	cc $^ -o $(OUT) $(LDLIBS)

# Para limpiar los archivos de salida
.PHONY: clean
clean:
	rm -f *.o $(OUT)