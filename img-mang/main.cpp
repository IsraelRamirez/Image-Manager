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
int rangeMin, rangeMax;
/** Funciones **/

/**
 * Funcion que guarda la imagen según un formato programa_operacion_yyyymmddHHMMss
 * @param image imagen a guardar
 * @param operation nombre de la operacion
*/
void saveImage(Mat image, string operation);

/**
 * Función que copia un segmento de una imagen para pegarla en otra imagen
 * @param src Imagen a la que se le realiza la copia
 * @param dst Imagen destino donde se guarda la copia
 * @param minx Inicio de la dimension x de la copia
 * @param miny Inicio de la dimension y de la copia
 * @param maxx Final de la dimension x de la copia
 * @param maxy Final de la dimension y de la copia
*/
void copyTo(Mat src, Mat dst, int minx, int miny, int maxx, int maxy);

/**
 * Función que recibe un segmento, o una imagen y la mezcla en la imagen destino, segun el proceso y la cantidad de procesadores es la posicion donde se implanta
 * @param src Segmento de la imagen total
 * @param dst Imagen total
 * @param proceso Proceso actual
 * @param procesadores Cantidad de procesos totales
*/
void join(Mat src, Mat dst,int proceso, int procesadores);

/**
 * Función que recibe un segmento, o una imagen y la mezcla en la imagen destino, segun el proceso y la cantidad de procesadores es la posicion donde se implanta
 * Nota: Esta función es específica para la opción 3
 * @param src Segmento de la imagen total
 * @param dst Imagen total
 * @param proceso Proceso actual
 * @param procesadores Cantidad de procesos totales
*/
void anotherJoin(Mat src, Mat dst,int proceso, int procesadores);

/**
 * Función que envia una imagen a un destinatario
 * @param imgToSend Es la imagen a enviar
 * @param dst Es el rango del destinatario a enviar
*/
void sendMsg(Mat imgToSend, int dst);

/**
 * Función que recibe la imagen de un destinatario
 * @param imgToRecv Es donde se guardará la imagen recebida
 * @param src Es el destinatario que envia la imagen
*/
void recvMsg(Mat &imgToRecv,int src);

/** Operacion 1 Difuminado de imagenes **/

/**
 * Funcion que obtiene un kernel de tamaño 5 x 5 y lo guarda en la variable global kernel[][]
*/
void getKernel();

/**
 * Funcion que difumina una imagen con el metodo de gauss
 * @param src Imagen a la cual se aplica la difuminación
 * @param dst Imagen destino, donde se guarda la imagen difuminada
 * @param maxx Maximo valor del eje x
 * @param maxy Maximo valor del eje y
*/
void gauss(Mat src, Mat dst, int maxx, int maxy);

/** Operacion 2 Escalado de grises **/

/**
 * Funcion transorma una imagen en RGB a escala de grises
 * @param src Imagen original a la que se hace la transformación
 * @param dst Imagen destino donde se guarda la transformación
 * @param maxx Valor maximo del eje x
 * @param maxy Valor maximo del eje y
*/
void RGB2GRAYS(Mat src, Mat dst, int maxx, int maxy);

/** Operación 3 Escalado de imagen **/

/**
 * Función que escala una imagen
 * @param src Imagen original a ser escalada
 * @param dst Imagen donde se guardara la imagen
 * @param scale porcentage de escalado
*/
Mat scaleIMG(Mat src, float scale);

/**
 * Función que realiza una interpolación bilineal
 * @param c00 pixel de esquina superior-izquierdo
 * @param c10 pixel de esquina inferior-izquierdo
 * @param c01 pixel de esquina superior-derecho
 * @param c11 pixel de esquina inferior-derecho
 * @return Devuelve el valor de interpolar un pixel con los 4 pixels de su subdivisión
*/
float Blerp(float c00, float c10, float c01, float c11, float x, float y);

/**
 * Función de interpolación lineal y = y0 + (x-x0) * m
 * @param s Es el valor de un pixel
 * @param e Es el valor de un pixel adyacente
 * @param t es la distancia entre ambos pixeles
 * @return Devuelve el valor de interpolar el color de un pixel segun un cierto ponderado de distancia
*/
float Lerp(float s, float e, float t);

/**
 * Este programa realiza 3 operaciones con respecto al tratamiento de imagenes
 * @param argc Cantidad de argumentos
 * @param argv Arreglo de argumentos
 * @return resultado exitoso o fallido de la operacion
*/
int main(int argc, char** argv ){
    if(argc > 2){
        int myrank, procesadores;
        Mat img, imgsplit;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &procesadores);

        string option(argv[1]);

        if(myrank == 0){
            string path = argv[2];
            img = imread(path, 1);

            int diferencia = img.cols / procesadores, agregado = 0;
            if(option == "1" || option == "2"){agregado = 2;}
            int mintemp = 0, maxtemp = diferencia;

            Mat tmpimgsplit(Size(diferencia+agregado, img.rows), img.type());
            imgsplit = tmpimgsplit.clone();
            copyTo(img, imgsplit, 0, 0, diferencia + agregado, img.rows);

            for(int p = 1; p < procesadores; p++){
                mintemp = (diferencia * p) - agregado;
                maxtemp = (diferencia * (p + 1)) + agregado;
                if(p+1 == procesadores){
                    maxtemp = img.cols;
                }
                int diference = maxtemp - mintemp;
                Mat imgToSend(Size(diference, img.rows), img.type());
                copyTo(img, imgToSend, mintemp, 0, maxtemp, img.rows);
                sendMsg(imgToSend, p);
            }
        }
        else{
            recvMsg(imgsplit,0);
        }
        if(option == "1" || option == "2"){

            Mat newimg = imgsplit.clone();
            if(option == "1"){
                getKernel();
                gauss(imgsplit, newimg, imgsplit.cols, imgsplit.rows);
            }
            else if(option == "2"){
                RGB2GRAYS(imgsplit, newimg, imgsplit.cols, imgsplit.rows);
            }

            if(myrank == 0){
                join(newimg, img, 0, procesadores);
                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recvMsg(imgtmpjoin, p);
                    join(imgtmpjoin, img, p, procesadores);
                }
                saveImage(img, option);
            }
            else{
                sendMsg(newimg, 0);
            }
        }
        
        else if(option == "3"){
            Mat tmpnewimg = scaleIMG(imgsplit, 2.0);
            if(myrank == 0){
                Mat newimg(img.rows*2, img.cols*2, CV_8UC3);
                anotherJoin(tmpnewimg, newimg, 0, procesadores);
                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recvMsg(imgtmpjoin, p);
                    anotherJoin(imgtmpjoin, newimg, p, procesadores);
                }
                saveImage(newimg, option);
            }
            else{
                sendMsg(tmpnewimg, 0);
            }
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

void copyTo(Mat src, Mat dst, int minx, int miny, int maxx, int maxy){
    for(int x = 0; x<maxx-minx; x++){
        for(int y = 0; y<maxy-miny; y++){
            dst.at<Vec3b>(y,x)[0] = src.at<Vec3b>(y,x+minx)[0];
            dst.at<Vec3b>(y,x)[1] = src.at<Vec3b>(y,x+minx)[1];
            dst.at<Vec3b>(y,x)[2] = src.at<Vec3b>(y,x+minx)[2];
        }
    }
}

void join(Mat src, Mat dst,int proceso, int procesadores){
    int diferencia = (dst.cols/procesadores)*proceso;
    int agregadoInicio = 0, agregadoFinal = 0;
    if(proceso !=0){
        agregadoInicio = 2;
    }
    if(proceso == procesadores-1){
        agregadoFinal = -2;
    }
    for(int x=0; x<src.cols+agregadoFinal; x++){
        for(int y = 0; y<src.rows; y++){
            dst.at<Vec3b>(y,diferencia+x)[0] = src.at<Vec3b>(y,x+agregadoInicio)[0];
            dst.at<Vec3b>(y,diferencia+x)[1] = src.at<Vec3b>(y,x+agregadoInicio)[1];
            dst.at<Vec3b>(y,diferencia+x)[2] = src.at<Vec3b>(y,x+agregadoInicio)[2];
        }
    }
}

void anotherJoin(Mat src, Mat dst,int proceso, int procesadores){
    int diferencia = (dst.cols/procesadores)*proceso;
    
    for(int x=0; x<src.cols; x++){
        for(int y = 0; y<src.rows; y++){
            dst.at<Vec3b>(y,diferencia+x)[0] = src.at<Vec3b>(y,x)[0];
            dst.at<Vec3b>(y,diferencia+x)[1] = src.at<Vec3b>(y,x)[1];
            dst.at<Vec3b>(y,diferencia+x)[2] = src.at<Vec3b>(y,x)[2];
        }
    }
}

void sendMsg(Mat imgToSend, int dst){
    size_t total, elemsize;
    int sizes[3];

    sizes[2] = imgToSend.elemSize();
    Size s = imgToSend.size();
    sizes[0] = s.height;
    sizes[1] = s.width;
    MPI_Send( sizes, 3, MPI_INT,dst,0,MPI_COMM_WORLD);
    MPI_Send( imgToSend.data, sizes[0]*sizes[1]*3, MPI_CHAR,dst,1, MPI_COMM_WORLD);
}

void recvMsg(Mat &imgToRecv,int src){
    MPI_Status estado;
    size_t total, elemsize;
    int sizes[3];
    MPI_Recv( sizes,3, MPI_INT,src,0, MPI_COMM_WORLD, &estado);
    imgToRecv.create(sizes[0], sizes[1], CV_8UC3);
    MPI_Recv( imgToRecv.data, sizes[0] * sizes[1] * 3, MPI_CHAR, src, 1, MPI_COMM_WORLD, &estado);
}

void gauss(Mat src, Mat dst, int maxx, int maxy){
    for(int x = 0; x < maxx; x++){
        for(int y = 0; y < maxy; y++){
            for(int c = 0; c < 3; c++){
                float sumGauss = 0;
                for(int kx = -2;kx < 3; kx++){
                    for(int ky = -2; ky < 3; ky++){
                        if(kx+x >= 0 && kx+x < maxx){
                            if(ky+y >= 0 && ky+y < maxy){
                                sumGauss += src.at<Vec3b>(y+ky,x+kx)[c] * kernel[ky+2][kx+2];
                            }
                            else{
                                sumGauss += src.at<Vec3b>(y,x+kx)[c] * kernel[ky+2][kx+2];
                            }
                        }
                        else{
                            if(ky+y >= 0 && ky+y < maxy){
                                sumGauss += src.at<Vec3b>(y+ky,x)[c] * kernel[ky+2][kx+2];
                            }
                            else{
                                sumGauss += src.at<Vec3b>(y,x)[c] * kernel[ky+2][kx+2];
                            }
                        }
                    }
                }
                dst.at<Vec3b>(y,x)[c] = sumGauss;
            }
        }
    }
}

void getKernel(){
    for(int i = 0; i<5; i++){
        for(int j = 0; j<5; j++){
            float expo = exp(-1*((pow(i-2,2)+pow(j-2,2))/(2*pow(1.5,2))));
            kernel[i][j]=expo/(2*3.1416*pow(1.5,2));
        }
    }
}

void RGB2GRAYS(Mat src, Mat dst, int maxx, int maxy){
    for(int x = 0; x < maxx; x++){
        for(int y = 0; y < maxy; y++){
            float promedio = (src.at<Vec3b>(y,x)[0] + src.at<Vec3b>(y,x)[1] + src.at<Vec3b>(y,x)[2])/3;
            dst.at<Vec3b>(y,x)[0] = promedio;
            dst.at<Vec3b>(y,x)[1] = promedio;
            dst.at<Vec3b>(y,x)[2] = promedio;
        }
    }
}

Mat scaleIMG(Mat src, float scale){
    int newcols = src.cols * scale;
    int newrows = src.rows * scale;
    Mat dst(newrows, newcols, CV_8UC3);
    for(int x = 0; x < newcols; x++){
        for(int y = 0; y < newrows; y++){
            float gx = ((float)(x) / newcols) * (src.cols - 1);
            float gy = ((float)(y) / newrows) * (src.rows - 1);

            int gxi = (int) gx;
            int gyi = (int) gy;
            int red = Blerp(src.at<Vec3b>(gyi, gxi)[0], src.at<Vec3b>(gyi + 1, gxi)[0], src.at<Vec3b>(gyi, gxi + 1)[0], src.at<Vec3b>(gyi + 1, gxi + 1)[0], gx - gxi, gy - gyi);
            int green = Blerp(src.at<Vec3b>(gyi, gxi)[1], src.at<Vec3b>(gyi + 1, gxi)[1], src.at<Vec3b>(gyi, gxi + 1)[1], src.at<Vec3b>(gyi + 1, gxi + 1)[1], gx - gxi, gy - gyi);
            int blue = Blerp(src.at<Vec3b>(gyi, gxi)[2], src.at<Vec3b>(gyi + 1, gxi)[2], src.at<Vec3b>(gyi, gxi + 1)[2], src.at<Vec3b>(gyi + 1, gxi + 1)[2], gx - gxi, gy - gyi);
            
            dst.at<Vec3b>(y, x)[0] = red;
            dst.at<Vec3b>(y, x)[1] = green;
            dst.at<Vec3b>(y, x)[2] = blue;
        }
    }
    return dst;
}

float Blerp(float c00, float c10, float c01, float c11, float x, float y){
    return Lerp(Lerp(c00, c10, x), Lerp(c01, c11, x), y);
}


float Lerp(float s, float e, float t){
    return s + (e - s) * t;
}