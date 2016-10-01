NVCC = nvcc
LIB = -lm
CFLAGS = -DITERATIONSPERTHREAD="1000"
EXEC = \
	time_test_montecarlo_cpu time_test_montecarlo_gpu\
	benchmark_clock_gettime error_rate_gpu

SRCS_common = main.c

%.o: %.cu
	$(NVCC) -c $^ -o $@

default: montecarlopi.o
	$(NVCC) $^ $(SRCS_common) -o time_test_montecarlo_cpu -DMONTECARLO_CPU $(LIB)
	$(NVCC) $^ $(SRCS_common) -o time_test_montecarlo_gpu -DMONTECARLO_GPU $(LIB)
	$(NVCC) $^ $(SRCS_common) -o benchmark_clock_gettime -DBENCHMARK $(LIB)
	$(NVCC) $(CFLAGS) $^ $(SRCS_common) -o error_rate_gpu -DERRORRATE $(LIB)

.PHONY: clean default

check:
	time ./time_test_montecarlo_cpu
	time ./time_test_montecarlo_gpu

gencsv:
	printf ",Monte Carlo with CPU,Monte Carlo with GPU\n" > result_clock_gettime.csv
	for i in `seq 2000000 4000000 40000000`; do \
		printf "%d," $$i;\
		./benchmark_clock_gettime $$i; \
	done >> result_clock_gettime.csv

errorrate:
	printf ",Monte Carlo with GPU\n" > result_clock_gettime.csv
	./error_rate_gpu >> result_clock_gettime.csv

clean:
	rm -f $(EXEC) *.o result_clock_gettime.csv
