.PHONY: all run

all:
	gcc -O2 main2.c -I$HOME/.mujoco/mujoco-3.1.3/include -L$HOME/.mujoco/mujoco-3.1.3/lib -lmujoco -lglfw -lm -lros3 -lmirage -lquicksand -o v22_sim

run:
	./v22_sim testdata/mcqueen.xml
