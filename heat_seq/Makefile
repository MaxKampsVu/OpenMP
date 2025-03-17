.PHONY: run_with_tracing traces clean

# Better to just add the pin binary to your $PATH
#PIN_ROOT=~/projects/pin/

MCPS_TRACER_ROOT=/home/spolstra/projects/mcps-tracer

CURR_DIR=$(notdir $(basename $(shell pwd)))
PRJ=$(CURR_DIR)
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))
ANNOT_LIB=../lib/libannotation.a

CC=gcc
INCLUDES=-I../include
ifndef DEBUG
CFLAGS=-O2 -std=gnu99 -lm
LDFLAGS=-L../lib -lannotation -lm
else
CFLAGS=-O0 -g3 -std=gnu99 -fsanitize=address
LDFLAGS=-L../lib -lm -fsanitize=address
endif

all: $(PRJ)

$(ANNOT_LIB):
	make -C ../lib

$(PRJ): $(OBJ) $(ANNOT_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

run: all
	./$(PRJ) -i 1

run_with_tracing: all
	pin -t $(MCPS_TRACER_ROOT)/tool/obj-intel64/tracer.so -- ./$(PRJ) -i 1

traces: run_with_tracing
	$(MCPS_TRACER_ROOT)/scripts/trace_converter.py address_log memory_trace.trf

clean:
	-rm -f $(OBJ) $(PRJ) address_log.* memory_trace.trf
	-make -C ../lib clean
