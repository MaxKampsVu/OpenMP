
# Better to just add the pin binary to your $PATH
#PIN_ROOT=~/projects/pin/

.PHONY: run_with_tracing traces clean
MCPS_TRACER_ROOT=/home/spolstra/projects/mcps-tracer

# Take number of threads from environment var P, use 2 as default
P ?= 2

CURR_DIR=$(notdir $(basename $(shell pwd)))
PRJ=$(CURR_DIR)
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))
ANNOT_LIB=../lib/libannotation.a

# Use Homebrew's LLVM Clang instead of system Clang
CC=$(shell brew --prefix llvm)/bin/clang

# Get OpenMP paths from Homebrew
LIBOMP_PATH=$(shell brew --prefix libomp)
LLVM_PATH=$(shell brew --prefix llvm)

INCLUDES=-I../include -I$(LIBOMP_PATH)/include
ifndef DEBUG
CFLAGS=-O2 -std=gnu99 -Xpreprocessor -fopenmp -I$(LIBOMP_PATH)/include
LDFLAGS=-L../lib -lannotation -L$(LIBOMP_PATH)/lib -lomp -lm
else
CFLAGS=-O0 -g3 -std=gnu99 -fsanitize=address -Xpreprocessor -fopenmp -I$(LIBOMP_PATH)/include
LDFLAGS=-L../lib -lm -fsanitize=address -L$(LIBOMP_PATH)/lib -lomp
endif

all: $(PRJ)

$(ANNOT_LIB):
	make -C ../lib

$(PRJ): $(OBJ) $(ANNOT_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

run_with_tracing: all
	pin -t $(MCPS_TRACER_ROOT)/obj-intel64/tool/tracer.so -- ./$(PRJ) -i 1 -p $(P)

traces: run_with_tracing
	$(MCPS_TRACER_ROOT)/scripts/trace_converter.py address_log memory_trace.trf

clean:
	-rm -f $(OBJ) $(PRJ) address_log.* memory_trace.trf
	-make -C ../lib clean
