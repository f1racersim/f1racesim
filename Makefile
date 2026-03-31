.PHONY: all clean run

CC := gcc
MUJOCO_DIR := $(CURDIR)/.local/mujoco-3.6.0
CFLAGS := -O2 -I$(MUJOCO_DIR)/include
LDFLAGS := -L$(MUJOCO_DIR)/lib -Wl,-rpath,$(MUJOCO_DIR)/lib
LDLIBS := -lmujoco -lglfw -lm -lros3 -lmirage -lquicksand

all: v22_sim

v22_sim: main2.o
	$(CC) $^ $(LDFLAGS) $(LDLIBS) -o $@

main2.o: main2.c
	$(CC) $(CFLAGS) -c $< -o $@

run: v22_sim
	./v22_sim testdata/mcqueen.xml

clean:
	rm -f main2.o v22_sim
