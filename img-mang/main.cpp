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

using namespace cv;
using namespace std;

void saveImage(Mat image, string operation);

int main(int argc, char** argv )
{
    
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
        }
        
        if(*argv[1]== '1'){
            if(myrank==0){
                
                saveImage(img,"1");
                
            }
        }
        else if(*argv[1]== '2'){
            
        }
        else if(*argv[1]== '3'){
        }
        else{
            
        
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
    cout<<"date:"<<date<<endl;
    imwrite("/media/compartida/programa_"+operation+"_"+date+".png", image);
}