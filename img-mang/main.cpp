#include <iostream>
#include <cstdlib>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core.hpp>
#include <mpi.h>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;
/** Variables globales **/
float kernel[5][5];

/** Funciones **/
/**
 * Funcion que guarda la imagen según un formato programa_operacion_yyyymmddHHMMss
 * @param image imagen a guardar
 * @param operation nombre de la operacion
*/
void saveImage(Mat image, string operation);

/** Operacion 1 Difuminado de imagenes **/

/**
 * Funcion que obtiene un kernel de tamaño 5 x 5 y lo guarda en la variable global kernel[][]
*/
void getKernel();

/**
 * Funcion que difumina un pixel con el metodo de gauss
 * @param centerx Punto x del punto central
 * @param centery Punto y del punto central
 * @param image Imagen a la cual se aplica la difuminación
 * @param minx Minimo valor del eje x 
 * @param miny Minimo valor del eje y
 * @param maxx Maximo valor del eje x
 * @param maxy Maximo valor del eje y
 * @param channel Canal al que se realiza la operación "R, G o B"
 * @return Devuelve el valor para el punto central (centerx, centery)
*/
float gauss(int centerx,int centery, Mat image,int minx, int miny, int maxx, int maxy, int channel);

/** Operacion 2 Escalado de grises **/

/**
 * Funcion transorma una imagen en RGB a escala de grises
 * @param src Imagen original a la que se hace la transformación
 * @param dst Imagen destino donde se guarda la transformación
 * @param minx Valor minimo del eje x
 * @param miny Valor minimo del eje y
 * @param maxx Valor maximo del eje x
 * @param maxy Valor maximo del eje y
*/
void RGB2GRAYS(Mat src, Mat dst,int minx, int miny, int maxx, int maxy);

/** Operación 3 Escalado de imagen **/

/**
 * Este programa realiza 3 operaciones con respecto al tratamiento de imagenes
 * @param argc Cantidad de argumentos
 * @param argv Arreglo de argumentos
 * @return resultado exitoso o fallido de la operacion
*/
int main(int argc, char** argv ){
    if(argc > 2){
        int myrank;
        int tag = 0;
        int procesadores;
        int rangeMin, rangeMax;
        Mat img,newimg;
        
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &procesadores);
        
        if(myrank == 0){
            string path = argv[2];
            img = imread(path,1);
            newimg = imread(path,1);
            
        }
        
        if(*argv[1]== '1'){
            if(myrank == 0){
                getKernel();
                for(int i = 0 ; i < img.rows ; i++){
                    for(int j = 0 ; j < img.cols ; j++){
                        for(int k = 0; k < 3; k++){
                            newimg.at<Vec3b>(i,j)[k] = gauss(j,i,img,0,0,img.cols,img.rows,k);
                        }
                    }
                }
                saveImage(newimg,"1");
            }
        }
        else if(*argv[1]== '2'){
            if(myrank == 0){
                RGB2GRAYS(img, newimg, 0, 0, img.cols, img.rows);
                saveImage(newimg,"2");
            }
        }
        else if(*argv[1]== '3'){
            //saveImage(newimg,"3");
        }
        else{
            cout<<"La opcion ingresada no es valida..."<<endl;
            return EXIT_FAILURE;
        }
        
        MPI_Finalize();
    }
    else{
        cout<<"No se ingresaron lo argumentos <opcion> <filepath>..."<<endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void saveImage(Mat image, string operation){
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);
    string date(buf);
    imwrite("/media/compartida/programa_"+operation+"_"+date+".png", image);
}
float gauss(int centerx,int centery, Mat image,int minx, int miny, int maxx, int maxy, int channel){
    float gaussBlur = 0;
    for(int x = 0; x<5; x++){
        if(centerx+x-2>=minx && centerx+x-2<maxx){
            for(int y = 0; y<5; y++){
                if(centery+y-2>=miny && centery+y-2 <maxy){
                    gaussBlur += image.at<Vec3b>(centery+y-2,centerx+x-2)[channel]*kernel[y][x];
                }
            }
        }
        
    }
    return gaussBlur;
}

void getKernel(){
    for(int i = 0; i<5; i++){
        for(int j = 0; j<5; j++){
            float expo = exp(-1*((pow(i-2,2)+pow(j-2,2))/(2*pow(1.5,2))));
            kernel[i][j]=expo/(2*3.1416*pow(1.5,2));
        }
    }
}

void RGB2GRAYS(Mat src, Mat dst,int minx, int miny, int maxx, int maxy){
    for(int x = minx; x < maxx; x++){
        for(int y = miny; y < maxy; y++){
            float promedio = (src.at<Vec3b>(y,x)[0] + src.at<Vec3b>(y,x)[1] + src.at<Vec3b>(y,x)[2])/3;
            dst.at<Vec3b>(y,x)[0] = promedio;
            dst.at<Vec3b>(y,x)[1] = promedio;
            dst.at<Vec3b>(y,x)[2] = promedio;
        }
    }
}