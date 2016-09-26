CC = gcc
CFLAGS = -O0 -std=gnu99 -Wall -fopenmp -mavx
EXECUTABLE = \
	time_test_baseline time_test_openmp_2 time_test_openmp_4 \
	time_test_avx time_test_avxunroll \
	time_test_leibniz time_test_montecarlo time_test_montecarlo_pthread\
	benchmark_clock_gettime

default: computepi.o
	$(CC) $(CFLAGS) computepi.o time_test.c -DBASELINE -o time_test_baseline -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DOPENMP_2 -o time_test_openmp_2 -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DOPENMP_4 -o time_test_openmp_4 -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DAVX -o time_test_avx -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DAVXUNROLL -o time_test_avxunroll -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DLEIBNIZ -o time_test_leibniz -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DMONTECARLO -o time_test_montecarlo -lm -lpthread
	$(CC) $(CFLAGS) computepi.o time_test.c -DMONTECARLO_4 -o time_test_montecarlo_pthread -lm -lpthread
	$(CC) $(CFLAGS) computepi.o benchmark_clock_gettime.c -o benchmark_clock_gettime -lm -lpthread

.PHONY: clean default

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

check: default
	time ./time_test_baseline
	time ./time_test_openmp_2
	time ./time_test_openmp_4
	time ./time_test_avx
	time ./time_test_avxunroll
	time ./time_test_leibniz
	time ./time_test_montecarlo
	time ./time_test_montecarlo_pthread

gencsv: default
	printf ",baseline,openMP_2,openMP_4,avx,avx+unroll,Leibniz,Monte Carlo,Monte Carlo+pthread\n" > result_clock_gettime.csv
	for i in `seq 100 5000 25000`; do \
		printf "%d," $$i;\
		./benchmark_clock_gettime $$i; \
	done >> result_clock_gettime.csv

clean:
	rm -f $(EXECUTABLE) *.o *.s result_clock_gettime.csv
