.PHONY: run_with_tracing traces clean

# Make sure that PIN_ROOT is set to your pin installation and pin is in your PATH.

MCPS_TRACER_ROOT=/home/spolstra/projects/mcps-tracer

CURR_DIR=$(notdir $(basename $(shell pwd)))
PRJ=$(CURR_DIR)
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))
ANNOT_LIB=../lib/libannotation.a

CC=gcc
INCLUDES=-I../include
ifndef DEBUG
CFLAGS=-O2 -std=gnu99 -lm -fopenmp
LDFLAGS=-L../lib -lannotation -lm -fopenmp
else
CFLAGS=-O0 -g3 -std=gnu99 -fsanitize=address -fopenmp
LDFLAGS=-L../lib -lm -fsanitize=address -fopenmp
endif

all: $(PRJ)

$(ANNOT_LIB):
	make -C ../lib

$(PRJ): $(OBJ) $(ANNOT_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

run: all
	./$(PRJ)

run_with_tracing: all
	pin -t $(MCPS_TRACER_ROOT)/tool/obj-intel64/tracer.so -- ./$(PRJ)

memory_trace.trf: address_log.00
	$(MCPS_TRACER_ROOT)/scripts/trace_converter.py address_log $@

clean:
	-rm -f $(OBJ) $(PRJ) address_log.* memory_trace.trf
	-make -C ../lib clean
