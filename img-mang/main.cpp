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
#include <pthread.h>

using namespace cv;
using namespace std;

/** Variables globales **/

string pathDest = "/media/compartida/";
double kernel[5][5];
float dev;
int rangeMin, rangeMax;
Mat imgsplit, newimg;
int NUMTHREADS = 2;

/**
 * Argumentos del hilo ejecutado
 * @param id Id del hilo ejecutado
 * @param option Opción elegida para la ejecución del programa
*/
struct t_data{
    int id;
    string option;
};

/** Funciones **/

/**
 * Participantes del grupo de la asignatura 
*/
void participantes();

/**
 * Función para inicializar un hilo de trabajo
 * @param t_arg Argumentos/datos de un hilo
*/
void *init(void *t_arg);

/**
 * Función que ejecuta la opcion ingresada
 * @param option Opción ingresada por el cliente
 * @param indica el hilo que ejecuta la función
*/
void selectorOpcion(string option, int thisthread);

/**
 * Función que ejecuta la difuminación gaussiana
 * @param indica el hilo que ejecuta la función
*/
void option1(int thisthread);

/**
 * Función que ejecuta el escalado de grises
 * @param indica el hilo que ejecuta la función
*/
void option2(int thisthread);

/**
 * Función que ejecuta el escalado de imagen a un 2x
 * @param indica el hilo que ejecuta la función
*/
void option3(int thisthread);

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
 * Función que calcula el exponente de Gauss
 * @param x posicion del punto "x"
 * @param mu Distancia desde el punto x al centro
 * @param sigma Desviación estandar
 * @return Devuelve el exponente de gauss
*/
double gaussian (double x, double mu, double sigma);

/**
 * Funcion que obtiene un kernel de tamaño 5 x 5 y lo guarda en la variable global kernel[][]
*/
void getKernel();


/**
 * Funcion que difumina una imagen con el metodo de gauss
 * @param src Imagen a la cual se aplica la difuminación
 * @param dst Imagen destino, donde se guarda la imagen difuminada
 * @param minx El mínimo valor de x que se tomará
 * @param miny El mínimo valor de y que se tomará
 * @param maxx Maximo valor del eje x
 * @param maxy Maximo valor del eje y
*/
void gauss(Mat src, Mat dst, int minx, int miny, int maxx, int maxy);

/** Operacion 2 Escalado de grises **/

/**
 * Funcion transorma una imagen en RGB a escala de grises
 * @param src Imagen original a la que se hace la transformación
 * @param dst Imagen destino donde se guarda la transformación
 * @param minx El mínimo valor de x que se tomará
 * @param miny El mínimo valor de y que se tomará
 * @param maxx Valor maximo del eje x
 * @param maxy Valor maximo del eje y
*/
void RGB2GRAYS(Mat src, Mat dst, int minx, int miny,int maxx, int maxy);

/** Operación 3 Escalado de imagen **/

/**
 * Función que escala una imagen
 * @param src Imagen original a ser escalada
 * @param dst Imagen donde se guardara la imagen
 * @param minx El mínimo valor de x que se tomará
 * @param miny El mínimo valor de y que se tomará
 * @param maxx Valor maximo del eje x
 * @param maxy Valor maximo del eje y
*/
void scaleIMG(Mat src, Mat dst, int minx, int miny,int maxx, int maxy);

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
    if(NUMTHREADS < 1){
        cout<< "La cantidad de hilos a utilizar debe ser mínimo 1..."<<endl;
        return EXIT_FAILURE;
    }
    if(argc > 2){
        int myrank, procesadores;
        Mat img;

        pthread_attr_t attr;
        void *status;

        /** Inicialización de MPI **/
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &procesadores);

        /** Variables usadas para el manejo de hilos **/
        pthread_t threads[NUMTHREADS]; // Lista de los hilos ejecutados
        struct t_data t_d[NUMTHREADS]; // Lista con los datos de los hilos

        string option(argv[1]);

        //Separción de la imagenes a la largo de esta
        if(myrank == 0){
            string path = argv[2];
            img = imread(path, -1);

            int diferencia = img.cols / procesadores, agregado = 0;
            if(option == "1" || option == "2"){agregado = 2;}
            int mintemp = 0, maxtemp = diferencia;

            imgsplit.create( img.rows, diferencia+agregado, CV_8UC4);
            copyTo(img, imgsplit, 0, 0, diferencia + agregado, img.rows);
            
            for(int p = 1; p < procesadores; p++){
                mintemp = (diferencia * p) - agregado;
                maxtemp = (diferencia * (p + 1)) + agregado;
                if(p+1 == procesadores){
                    maxtemp = img.cols;
                }
                int diference = maxtemp - mintemp;
                Mat imgToSend(Size(diference, img.rows), CV_8UC4);
                copyTo(img, imgToSend, mintemp, 0, maxtemp, img.rows);
                
                sendMsg(imgToSend, p);
            }
        }
        // Recepción de las imagenes en los distintos procesadores
        else{
            recvMsg(imgsplit,0);
            
        }
        // Creación del medio final donde se guardará la imagen procesada
        if(option == "1" || option == "2"){
            
            if(option == "1"){
                dev = 0.99;
                getKernel();
            }
            
            newimg = imgsplit.clone();
        }
        else if(option == "3"){
            int newcols = imgsplit.cols * 1.13;
            int newrows = imgsplit.rows * 1.13;
            newimg.create(newrows, newcols, CV_8UC4);
        }
        
        // Declaración para hilos "joineables"
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        // Inicialización de los hilos
        for(int i = 0; i < NUMTHREADS; i++){
            t_d[i].id = i;
            t_d[i].option = option;

            int rc = pthread_create(&threads[i], NULL, init, (void *)&t_d[i]);
            if(rc){
                cout << "Error:unable to create thread," << myrank<< endl;
            }

        }

        // Destrucción de los atributos
        pthread_attr_destroy(&attr);
        // Join de los hilos que han terminado sus funciones
        for(int i = 0; i < NUMTHREADS; i++) {
            int rc = pthread_join(threads[i], &status);
            if (rc) {
                cout << "Error:unable to join," << rc << endl;
            }
        }

        // Merge de las imagenes segmentadas y procesadas para la opcion 1 y 2
        if(option == "1" || option == "2"){
            
            if(myrank == 0){
                Mat finalimg(img.rows, img.cols, CV_8UC4);
                join(newimg, finalimg, 0, procesadores);

                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recvMsg(imgtmpjoin, p);
                    join(imgtmpjoin, finalimg, p, procesadores);
                }
                
                    
                saveImage(finalimg, option);
            }
            // Envio de las imagenes al procesador maestro
            else{
                sendMsg(newimg, 0);
            }
        }
        // Merge de las imagenes segmentadas y procesadas para la opcion 3
        else if(option == "3"){

            if(myrank == 0){

                Mat tmpnewimg(img.rows*1.13, img.cols*1.13, CV_8UC4);
                anotherJoin(newimg, tmpnewimg, 0, procesadores);

                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recvMsg(imgtmpjoin, p);
                    anotherJoin(imgtmpjoin, tmpnewimg, p, procesadores);
                }

                saveImage(tmpnewimg, option);

            }
            // Envio de las imagenes al procesador maestro
            else{
                sendMsg(newimg, 0);
            }
        }
        else{
            cout<<"La opcion ingresada no es valida..."<<endl;
            return EXIT_FAILURE;
        }
        if(myrank == 0)
            participantes();
        MPI_Finalize();
    }
    else{
        cout<<"No se ingresaron lo argumentos <opcion> <filepath>..."<<endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void participantes(){
    cout<<endl<<"@===Participantes====@"<<endl;
    cout<<"@===Israel Ramirez===@"<<endl;
    cout<<"@===Humberto Roman===@"<<endl;
    cout<<"@===Victor Araya=====@"<<endl;
    cout<<"@===Participantes====@"<<endl;

}

void *init(void *t_arg){
    struct t_data *my_data = (struct t_data *) t_arg;

    selectorOpcion(my_data->option, my_data->id);

    pthread_exit(NULL);
}

void selectorOpcion(string option, int thisthread){
    if(option == "1"){ option1(thisthread); }
    else if(option == "2"){ option2(thisthread); }
    else if(option == "3"){ option3(thisthread); }
}

void option1(int thisthread){
    

    int diferencia = imgsplit.rows / NUMTHREADS;
    int minx = 0;
    int maxx = imgsplit.cols;
    int miny = diferencia * thisthread;
    int maxy = diferencia * (thisthread + 1);

    if(thisthread + 1 == NUMTHREADS){ maxy = imgsplit.rows; }
    // Se hace 4 veces para mejor difuminación en imagenes de alta resolución
    gauss(imgsplit, newimg, minx, miny, maxx, maxy);
    gauss(newimg, newimg, minx, miny, maxx, maxy);
    gauss(newimg, newimg, minx, miny, maxx, maxy);
    gauss(newimg, newimg, minx, miny, maxx, maxy);
}

void option2(int thisthread){
    
    int diferencia = imgsplit.rows / NUMTHREADS;
    int minx = 0;
    int maxx = imgsplit.cols;
    int miny = diferencia * thisthread;
    int maxy = diferencia * (thisthread + 1);

    if(thisthread + 1 == NUMTHREADS){ maxy = imgsplit.rows; }

    RGB2GRAYS(imgsplit, newimg, minx, miny, maxx, maxy);
}

void option3(int thisthread){
    int newcols = imgsplit.cols * 1.13;
    int newrows = imgsplit.rows * 1.13;

    int diferencia = newrows / NUMTHREADS;
    int minx = 0;
    int maxx = newcols;
    int miny = diferencia * thisthread;
    int maxy = diferencia * (thisthread + 1);

    if(thisthread + 1 == NUMTHREADS){ maxy = newrows; }

    scaleIMG(imgsplit, newimg, minx, miny, maxx, maxy);
}

void saveImage(Mat image, string operation){
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);
    string date(buf);
    imwrite(pathDest+"programa_"+operation+"_"+date+".png", image);
}

void copyTo(Mat src, Mat dst, int minx, int miny, int maxx, int maxy){
    for(int x = 0; x<maxx-minx; x++){
        for(int y = 0; y<maxy-miny; y++){
            dst.at<Vec4b>(y,x)[0] = src.at<Vec3b>(y,x+minx)[0];
            dst.at<Vec4b>(y,x)[1] = src.at<Vec3b>(y,x+minx)[1];
            dst.at<Vec4b>(y,x)[2] = src.at<Vec3b>(y,x+minx)[2];
            dst.at<Vec4b>(y,x)[3] = 255;
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
            dst.at<Vec4b>(y,diferencia+x)[0] = src.at<Vec4b>(y,x+agregadoInicio)[0];
            dst.at<Vec4b>(y,diferencia+x)[1] = src.at<Vec4b>(y,x+agregadoInicio)[1];
            dst.at<Vec4b>(y,diferencia+x)[2] = src.at<Vec4b>(y,x+agregadoInicio)[2];
            dst.at<Vec4b>(y,diferencia+x)[3] = src.at<Vec4b>(y,x+agregadoInicio)[3];
        }
    }
}

void anotherJoin(Mat src, Mat dst,int proceso, int procesadores){
    int diferencia = (dst.cols/procesadores)*proceso;
    
    for(int x=0; x<src.cols; x++){
        for(int y = 0; y<src.rows; y++){
            dst.at<Vec4b>(y,diferencia+x)[0] = src.at<Vec4b>(y,x)[0];
            dst.at<Vec4b>(y,diferencia+x)[1] = src.at<Vec4b>(y,x)[1];
            dst.at<Vec4b>(y,diferencia+x)[2] = src.at<Vec4b>(y,x)[2];
            dst.at<Vec4b>(y,diferencia+x)[3] = src.at<Vec4b>(y,x)[3];
        }
    }
}

void sendMsg(Mat imgToSend, int dst){
    int sizes[3];

    sizes[2] = imgToSend.elemSize();
    Size s = imgToSend.size();
    sizes[0] = s.height;
    sizes[1] = s.width;
    MPI_Send( sizes, 3, MPI_INT,dst,0,MPI_COMM_WORLD);
    MPI_Send( imgToSend.data, sizes[0]*sizes[1]*4, MPI_CHAR,dst,1, MPI_COMM_WORLD);
}

void recvMsg(Mat &imgToRecv,int src){
    MPI_Status estado;
    
    int sizes[3];
    MPI_Recv( sizes,3, MPI_INT,src,0, MPI_COMM_WORLD, &estado);
    imgToRecv.create(sizes[0], sizes[1], CV_8UC4);
    MPI_Recv( imgToRecv.data, sizes[0] * sizes[1] * 4, MPI_CHAR, src, 1, MPI_COMM_WORLD, &estado);
}

void gauss(Mat src, Mat dst, int minx, int miny,int maxx, int maxy){
    for(int x = minx; x < maxx; x++){
        for(int y = miny; y < maxy; y++){
            for(int c = 0; c < 3; c++){
                float sumGauss = 0;
                for(int kx = -2;kx < 3; kx++){
                    for(int ky = -2; ky < 3; ky++){
                        if(kx+x >= 0 && kx+x < maxx){
                            if(ky+y >= 0 && ky+y < src.rows){
                                sumGauss += ((float)(src.at<Vec4b>(y+ky,x+kx)[c]) * kernel[ky+2][kx+2]);
                            }
                            else{
                                sumGauss += ((float)(src.at<Vec4b>(y,x+kx)[c]) * kernel[ky+2][kx+2]);
                            }
                        }
                        else{
                            if(ky+y >= 0 && ky+y < src.rows){
                                sumGauss += ((float)(src.at<Vec4b>(y+ky,x)[c]) * kernel[ky+2][kx+2]);
                            }
                            else{
                                sumGauss += ((float)(src.at<Vec4b>(y,x)[c]) * kernel[ky+2][kx+2]);
                            }
                        }
                    }
                }
                dst.at<Vec4b>(y,x)[c] = sumGauss;
            }
            dst.at<Vec4b>(y,x)[3] = 255;
        }
    }
}

double gaussian (double x, double mu, double sigma) {
     return exp( -(((x-mu)/(sigma))*((x-mu)/(sigma)))/2.0 );
}

void getKernel(){
    for(int i = 0; i<5; i++){
        for(int j = 0; j<5; j++){
            float expo = exp(((-pow(i-2,2)-pow(j-2,2))/(2*pow(dev,2))));
            kernel[i][j]=expo/(2*3.1416*pow(dev,2));
        }
    }
}

void RGB2GRAYS(Mat src, Mat dst, int minx, int miny,int maxx, int maxy){
    for(int x = minx; x < maxx; x++){
        for(int y = miny; y < maxy; y++){
            float promedio = (src.at<Vec4b>(y,x)[0] + src.at<Vec4b>(y,x)[1] + src.at<Vec4b>(y,x)[2])/3;
            dst.at<Vec4b>(y,x)[0] = promedio;
            dst.at<Vec4b>(y,x)[1] = promedio;
            dst.at<Vec4b>(y,x)[2] = promedio;
            dst.at<Vec4b>(y,x)[3] = 255;
        }
    }
}

void scaleIMG(Mat src, Mat dst, int minx, int miny,int maxx, int maxy){
    
    for(int x = minx; x < maxx; x++){
        for(int y = miny; y < maxy; y++){
            float gx = ((float)(x) / dst.cols) * (src.cols - 1);
            float gy = ((float)(y) / dst.rows) * (src.rows - 1);

            int gxi = (int) gx;
            int gyi = (int) gy;
            int red = Blerp(src.at<Vec4b>(gyi, gxi)[0], src.at<Vec4b>(gyi + 1, gxi)[0], src.at<Vec4b>(gyi, gxi + 1)[0], src.at<Vec4b>(gyi + 1, gxi + 1)[0], gx - gxi, gy - gyi);
            int green = Blerp(src.at<Vec4b>(gyi, gxi)[1], src.at<Vec4b>(gyi + 1, gxi)[1], src.at<Vec4b>(gyi, gxi + 1)[1], src.at<Vec4b>(gyi + 1, gxi + 1)[1], gx - gxi, gy - gyi);
            int blue = Blerp(src.at<Vec4b>(gyi, gxi)[2], src.at<Vec4b>(gyi + 1, gxi)[2], src.at<Vec4b>(gyi, gxi + 1)[2], src.at<Vec4b>(gyi + 1, gxi + 1)[2], gx - gxi, gy - gyi);
            
            dst.at<Vec4b>(y, x)[0] = red;
            dst.at<Vec4b>(y, x)[1] = green;
            dst.at<Vec4b>(y, x)[2] = blue;
            dst.at<Vec4b>(y, x)[3] = 255;
        }
    }
}

float Blerp(float c00, float c10, float c01, float c11, float x, float y){
    return Lerp(Lerp(c00, c10, x), Lerp(c01, c11, x), y);
}


float Lerp(float s, float e, float t){
    return s + (e - s) * t;
}