# Image-Manager
***
##### Integrantes:
* Victor Araya
* Humberto Román
* Israel Ramirez
***
##### Aplicativo que usa la libreria opencv, openmpi y posix threads para el tratamiento de imagenes.
***
Para esta aplicación se usó el lenguaje `c++`, junto con las librerías de `openmpi`, `opencv` y `pthreads` de posix.

Requiere tener instalado y configurado `openmpi`

Requiere tener instalado la libreria `libopencv-dev` usando `sudo apt-get install libopencv-dev`en Ubuntu 20.04.

Esta aplicación requiere el uso exclusivo de sistemas operativos basados en `Linux` debido al uso de la libreria `pthreads` de posix.

Es necesario configurar el archivo `.../img-mang/maquinas.txt` con las máquinas que se disponen para el uso del programa y cambiar la ruta de salida de las imagenes guardadas en la variagle global al inicio del documento `pathDest` a la ruta donde quiere que se guarde la imagen, por defecto la ruta es `/media/compartida/`. También, es posible cambiar la cantidad de `Threads` utilizados, en la variable global `NUMTHREADS`, por defecto `NUMTHREADS = 2`.

Además, se debe ejecutar `.../img-mang$ make clean && make` para la compilación del aplicativo.

***
# Importante: 
##### Es posible que se encuentren problemas con el archivo de maquinas.txt, la solucioón a dicho problema es ejecutar la línea de codigo de la siguiente forma: `.../img-mang$ mpirun -host <host1>,<host2>,<host3>,...,<hostn> ./dist/programa <opción> <path de la imagen>`
***

El aplicativo cuenta con 3 opciones para el tratamiento de imagenes:

* 1.- Difuminado de imagenes, utilizando difuminado gaussian. Para su uso, utilizar la siguiente línea `.../img-mang$ mpirun --hostfile maquinas.txt ./dist/programa 1 <path de la imagen>`

* 2.- Escala de grises. Para su uso, utilizar la siguiente línea `.../img-mang$ mpirun --hostfile maquinas.txt ./dist/programa 2 <path de la imagen>`

* 3.- Escalado de imagen 2x. Para su uso, utilizar la siguiente línea `.../img-mang$ mpirun --hostfile maquinas.txt ./dist/programa 3 <path de la imagen>`

La lógica empleada para el desarollo de esta aplicación es la siguiente:

Teniendo en cuenta una imagen de tamaño `L * A`, donde L es el largo de la imagen y A es el alto de la imagen, se divide la L entre la cantidad total de procesadores (p) usados con `openMPI`, obteniendo p-trozos de la imagen, donde cada trozo tendrá un tamaño de `(L/p) * A`. Cada trozo será subdivido a lo alto en la cantidad de threads configurada (t), por lo tanto, se procesaran trozos de la imagen original del tamaño de `(L/p) * (A/t)`. 

***
##### Ejemplos:
###### Imagen Original
![img](img-mang/examples/test.png "Original")
###### Difuminado gaussiano
![img](img-mang/examples/programa_1.png "Difuminado gaussiano")
###### Escalado de grises
![img](img-mang/examples/programa_2.png "Escala de grises")
###### Escalado 2x
![img](img-mang/examples/programa_3.png "Escalado 2x")
