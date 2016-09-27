NVCC = nvcc
EXEC = \
	time_test_montecarlo_cpu time_test_montecarlo_gpu\
	benchmark_clock_gettime

all: 
	$(NVCC) montecarlopi.cu -o time_test_montecarlo_cpu -DMONTECARLO_CPU
	$(NVCC) montecarlopi.cu -o time_test_montecarlo_gpu -DMONTECARLO_GPU
	$(NVCC) montecarlopi.cu -o benchmark_clock_gettime -DBENCHMARK

check:
	time ./time_test_montecarlo_cpu
	time ./time_test_montecarlo_gpu

gencsv:
	printf ",Monte Carlo with CPU,Monte Carlo with GPU\n" > result_clock_gettime.csv
	for i in `seq 2000000 4000000 40000000`; do \
		printf "%d," $$i;\
		./benchmark_clock_gettime $$i; \
	done >> result_clock_gettime.csv

clean:
	rm -f $(EXEC) result_clock_gettime.csv
