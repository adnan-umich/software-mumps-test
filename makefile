CC      = mpicc
CFLAGS  = -O3 -fopenmp

MUMPS_DIR     = /sw/spack/opt/pkgs/mumps/5.9.1-gcc-14.3.1-25vt
SCALAPACK_DIR = /sw/spack/opt/pkgs/netlib-scalapack/2.2.3-gcc-14.3.1-qtfq

INCLUDES = \
	-I$(MUMPS_DIR)/include \
	-I$(SCALAPACK_DIR)/include

LDFLAGS = \
	-L$(MUMPS_DIR)/lib \
	-L$(SCALAPACK_DIR)/lib64

LIBS = \
	-ldmumps \
	-lmumps_common \
	-lpord \
	-lscalapack \
	-lopenblas \
	-lgfortran \
	-lm \
	-fopenmp

TARGET = mumps_hpc
SRC    = mumps_hpc.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

run: $(TARGET)
	OMP_NUM_THREADS=1 mpirun -np 4 ./$(TARGET) 100 100 100

clean:
	rm -f $(TARGET)

.PHONY: all run clean
