#!/bin/bash -x

echo "Construyendo el proyecto"
make clean && make

echo "Ejecutando el proyecto"
time mpirun --hostfile maquinas.txt ./dist/programa 1 /media/compartida/test.png