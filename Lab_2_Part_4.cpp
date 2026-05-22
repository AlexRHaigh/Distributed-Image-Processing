#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

/*
parallelRange

The provided parallel range function
*/
void parallelRange(int globalstart, int globalstop, int irank, int nproc, int& localstart, int& localstop, int& localcount)
{
    int nrows = globalstop - globalstart + 1;
    int divisor = nrows / nproc;
    int remainder = nrows % nproc;
    int offset;
    if (irank < remainder) offset = irank;
    else offset = remainder;

    localstart = irank * divisor + globalstart + offset;
    localstop = localstart + divisor - 1;
    if (remainder > irank) localstop += 1;
    localcount = localstop - localstart + 1;
}

/*
serialImageConvolution

This Code is used to serially perform an image convolution. Its code it identical to the code that was created in Part 1
*/

void serialImageConvolution(cv::Mat& inputImage, cv::Mat& outputImage,const vector<double> convolutionKernel)
{
    int kernelDimensions = (int)sqrt(convolutionKernel.size());//obtain kernel dimension
    int distanceToEdge = (int)floor(kernelDimensions/2.0);//find distance from center to edge of kernel

    for (int iy = 0; iy < inputImage.rows; iy++)//for every pixel in the image
    {
        for (int ix = 0; ix < inputImage.cols; ix++)
        {
            double kernelBSum = 0;//initiate all used varaibles at the beginning of each pixel
            double kernelGSum = 0;
            double kernelRSum = 0;
            int blur_pixel_count = 0;
            int usedKernelAmounts = 0;

            for (int i = -distanceToEdge; i <= distanceToEdge; i++){//for the whole kernel parameters
                for (int j = -distanceToEdge; j <= distanceToEdge; j++){
                    if (iy + i >= 0 && ix + j >= 0 && iy + i < inputImage.rows && ix + j < inputImage.cols){//makes it so we dont access pixels that dont exist
                        kernelBSum += inputImage.at<cv::Vec3b>(iy + i,ix + j)[0]*convolutionKernel[blur_pixel_count];//adds to the sum using kernel entry and colour at pixel
                        kernelGSum += inputImage.at<cv::Vec3b>(iy + i,ix + j)[1]*convolutionKernel[blur_pixel_count];
                        kernelRSum += inputImage.at<cv::Vec3b>(iy + i,ix + j)[2]*convolutionKernel[blur_pixel_count];
                        usedKernelAmounts += fabs(convolutionKernel[blur_pixel_count]);//add to the used kernel amounts
                    }
                    blur_pixel_count++;//add one to the blur pixel count for mean calculation
                }
            }

            //cout << "KernelBSum,kernelGSum,kernelRSum = " << kernelBSum << "," << kernelGSum << "," << kernelRSum << endl;

            kernelBSum = kernelBSum/usedKernelAmounts;//obtains the mean of the sum based on how many kernel entries were used
            kernelGSum = kernelGSum/usedKernelAmounts;
            kernelRSum = kernelRSum/usedKernelAmounts;

            outputImage.at<cv::Vec3b>(iy,ix)[0] = (int)floor(kernelBSum);//sets the value for each channel in the output image
            outputImage.at<cv::Vec3b>(iy,ix)[1] = (int)floor(kernelGSum);
            outputImage.at<cv::Vec3b>(iy,ix)[2] = (int)floor(kernelRSum);
        }
    }
}


int main(int argc, char** argv)
{
    int rank, nproc;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // Get the number of processors
    
    MPI_Datatype Vec3b,Row_Pixels; //create the datatypes we will use

    vector<int> displs(nproc, 0); //create the displs and count vectors
    vector<int> count(nproc, 0);

    int localstart, localstop, localcount, scatterCount, amountRows, haloRows, kernelDimensions, kernelSize; //create integers that we will need

    int size[2];//create the array that will store the size of the image
    vector<double> convolutionKernel;//create the vector that will store the convolution kernel
    double end_timeparallel,start_timeparallel;
    cv::Mat image,recvImage,processed_image1;//create the images

    if (rank == 0){

        string filename = argv[1];
        image = cv::imread(filename,1);   // Read the file

        processed_image1 = cv::Mat(image.rows,image.cols, image.type());//create the processed image

        string kernelFileName = argv[2];//read the kernel file name
        fstream kernelFile;

        kernelFile.open(kernelFileName,ios::in);//open the kernel file

        string textLine;
        while (getline(kernelFile, textLine)) {//obtain the number that is inputted as a portion of the convolution kernel
            stringstream ss(textLine);
            string cell;

            while (getline(ss, cell, ',')) {
                int value;
                istringstream(cell) >> value;
                convolutionKernel.push_back(value*1.0);
            }
        }
        size[0] = image.rows;//fill the size array
        size[1] = image.cols;

        kernelSize = convolutionKernel.size();//define the kernel size
    }

    MPI_Bcast(&kernelSize, 1, MPI_INT, 0, MPI_COMM_WORLD);//broadcast the kernel size to all processors
    if (rank != 0){
        convolutionKernel.resize(kernelSize);//resize the convolution kernel vector to be equal to the kernel size
    }
    MPI_Bcast(&convolutionKernel.at(0), kernelSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);//broadcast the kernel to all processors
    MPI_Bcast(size, 2, MPI_INT, 0, MPI_COMM_WORLD);//broadcast the image size to all processors

    if (rank == 0){
        int kernelDimensions = (int)sqrt(convolutionKernel.size());//determine kernel dimensions
        haloRows = (int)floor(kernelDimensions/2.0);//determine amount of halo rows that are needed based on our kernel

        serialImageConvolution(image, processed_image1, convolutionKernel);//run the serial blur image function

        double start_timeparallel = MPI_Wtime();
        parallelRange(0, size[0], 0, nproc, localstart, localstop, localcount);
        recvImage = cv::Mat(localcount+haloRows, size[1], CV_8UC3);//call parallel range to get the size of the recieved image on rank 0

        count.resize(nproc);//resize count and displs vectors
        displs.resize(nproc);

        count[0] = localcount + haloRows;//add halo rows to the amount of rows that will be recieved on rank 0
        displs[0] = 0;//no displacement for the portion that goes to rank 1
        for (int i = 1; i < nproc; i++){
            parallelRange(0, image.rows, i, nproc, localstart, localstop, localcount);
            if (i == (nproc - 1)){
                count[i] = localcount + haloRows;//initiate count and displacement based on halo rows
                displs[i] = localstart - haloRows;
            }
            else{
                count[i] = localcount + 2*haloRows;//for every other processor there will be 2 sets of halo rows
                displs[i] = localstart - haloRows;                    
            }
        }
        scatterCount = count[0];//determine the amount of data that will be recieved by this processor
    }
    else{
        int kernelDimensions = (int)sqrt(convolutionKernel.size());//obtain kernel dimenstions
        haloRows = floor(kernelDimensions/2.0);//obtain amount of halo rows
        parallelRange(0 , size[0], rank , nproc, localstart, localstop , localcount);
        if (rank == nproc - 1){
            scatterCount = localcount + haloRows;//determine the amount of data that will be recieved by this processor
            recvImage = cv::Mat(localcount+haloRows, size[1], CV_8UC3);//create image that will recieve data
        }
        else{
            scatterCount = localcount + 2*haloRows;//determine the amount of data that will be recieved by this processor
            recvImage = cv::Mat(localcount+2*haloRows, size[1], CV_8UC3);//create image that will recieve data
        }
        image = cv::Mat(1, 1, CV_8UC3);//create proxy image for memory reasons
    }


    MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &Vec3b);//Define a MPI_Type_contiguous derived type for cv::Vec3b
    MPI_Type_commit(&Vec3b);

    MPI_Type_contiguous(size[1],Vec3b,&Row_Pixels);//Define a MPI_Type_contiguous derived type for row of pixels
    MPI_Type_commit(&Row_Pixels);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(image.data, count.data(), displs.data(), Row_Pixels, recvImage.data, scatterCount, Row_Pixels, 0, MPI_COMM_WORLD);//scatter image rows to each processor

    std::string recvImageTitle = "Recived Image on Rank#";//create image title for recieved portion
    std::string rankNum = to_string(rank);
    std::string finalTitle = recvImageTitle + rankNum;

    cv::namedWindow(finalTitle, cv::WINDOW_AUTOSIZE );	// Create a window for display.
    cv::imshow(finalTitle, recvImage);                  // Show our image inside it.
    //cv::waitKey(10000);//wait 10 seconds

    cv::Mat proxyImage(recvImage.rows, recvImage.cols, CV_8UC3);//create a proxy image to store updated image

    kernelDimensions = (int)sqrt(convolutionKernel.size());//similar as above obtain kernel dimensions and halo rows
    haloRows = (int)floor(kernelDimensions/2.0);

    int frontDisplacement = 0;//these displacements are so that halo rows dont have the convolution kernel applied to them
    int backDisplacement = 0;

    if (rank == 0){
        backDisplacement = haloRows;//there are different displacement depending on it you have the first or the last portion
        frontDisplacement = 0;
    }
    else if (rank == (nproc - 1)){
        frontDisplacement = haloRows;
        backDisplacement = 0;
    }
    else{
        frontDisplacement = haloRows;
        backDisplacement = haloRows;
    }

    for (int iy = 0 + frontDisplacement; iy < recvImage.rows - backDisplacement; iy++)//apply the serial portion that uses the convolation kernel on the image piece
    {
        for (int ix = 0; ix < recvImage.cols; ix++)
        {
            double kernelBSum = 0;
            double kernelGSum = 0;
            double kernelRSum = 0;
            int blur_pixel_count = 0;
            int usedKernelAmounts = 0;

            for (int i = -haloRows; i <= haloRows; i++){
                for (int j = -haloRows; j <= haloRows; j++){
                    if (iy + i >= 0 && ix + j >= 0 && iy + i < recvImage.rows && ix + j < recvImage.cols){
                        kernelBSum += recvImage.at<cv::Vec3b>(iy + i,ix + j)[0]*convolutionKernel[blur_pixel_count];
                        kernelGSum += recvImage.at<cv::Vec3b>(iy + i,ix + j)[1]*convolutionKernel[blur_pixel_count];
                        kernelRSum += recvImage.at<cv::Vec3b>(iy + i,ix + j)[2]*convolutionKernel[blur_pixel_count];
                        usedKernelAmounts += fabs(convolutionKernel[blur_pixel_count]);
                    }
                    blur_pixel_count++;
                }
            }

            kernelBSum = kernelBSum/usedKernelAmounts;
            kernelGSum = kernelGSum/usedKernelAmounts;
            kernelRSum = kernelRSum/usedKernelAmounts;

            proxyImage.at<cv::Vec3b>(iy,ix)[0] = (int)floor(kernelBSum);
            proxyImage.at<cv::Vec3b>(iy,ix)[1] = (int)floor(kernelGSum);
            proxyImage.at<cv::Vec3b>(iy,ix)[2] = (int)floor(kernelRSum);
        }
    }

    recvImageTitle = "Modified Image Segment on Rank#";//create proper title for the modified image segment
    rankNum = to_string(rank);
    string secondTitle = recvImageTitle + rankNum;

    cv::namedWindow(secondTitle, cv::WINDOW_AUTOSIZE );	// Create a window for display.
    cv::imshow(secondTitle, proxyImage);                // Show our image inside it.
    //cv::waitKey(50000);//wait 10 seconds

    cv::Mat finalImage;//create final image

    int firstNonHaloEntry = 0;//create a variable to hold the first non halo entry

    if (rank == 0){
        for (int i = 0; i < nproc; i++)
        {
            parallelRange(0 , size[0], i , nproc, localstart, localstop , localcount);
            count[i] = localcount;//determine recieved counts and displacements for each rank
            displs[i] = localstart; 
        }
        finalImage = cv::Mat(size[0], size[1], CV_8UC3);//create final image
    }
    else{
        firstNonHaloEntry+=haloRows;//determine first entry that is not a halo row 
        finalImage = cv::Mat(1, 1, CV_8UC3);//create proxy final image
        count.resize(nproc);//resize count and displs vectors
        displs.resize(nproc);
    }

    parallelRange(0 , size[0], rank , nproc, localstart, localstop , localcount);//obtain the local count on each node

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(&proxyImage.at<cv::Mat3b>(firstNonHaloEntry,0), localcount, Row_Pixels, finalImage.data, count.data(), displs.data(), Row_Pixels, 0, MPI_COMM_WORLD);//gather the data onto rank 0

    if (rank == 0){
        double end_timeparallel = MPI_Wtime();

        cout << "Parallel Time = " << end_timeparallel-start_timeparallel << endl;

        cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE );	// Create a window for display.
        cv::imshow("Original Image", image);                   				// Show our image inside it.
        
        cv::namedWindow("Serial Blurred Image", cv::WINDOW_AUTOSIZE );	// Create display window.
        cv::imshow("Serial Blurred Image", processed_image1);      // Show our image inside it.

        cv::namedWindow("Parallel Blurred Image", cv::WINDOW_AUTOSIZE );	// Create display window.
        cv::imshow("Parallel Blurred Image", finalImage);      // Show our image inside it.

        string serialname = "Serial_final_Image_" + std::to_string(rank) + ".jpg";
        cv::imwrite(serialname, processed_image1);

        string parallelname = "Parallel_final_Image_" + std::to_string(rank) + ".jpg";
        cv::imwrite(parallelname, finalImage);

        int success = 0;//variable to store if the parallel image is the same as the serial one
        int flag = 0;//used to see which row isnt the same (used this for debugging)

        if ((processed_image1.cols == finalImage.cols) && (processed_image1.rows == finalImage.rows)){//if image dimensions are identical
            for (int i = 0; i < finalImage.rows - haloRows; i++){
                for (int j = 0; j < finalImage.cols; j++){
                    for (int k = 0; k < 3; k++){
                        if (processed_image1.at<cv::Vec3b>(i,j)[k] != finalImage.at<cv::Vec3b>(i,j)[k]){//if at any point the pixel value is not correct
                            success = 1;//set both success and flag off
                            flag = 1;
                        }
                    }
                }
                if (flag){//points to row that has the error
                    cout << "Serial is NOT equal to parallel at row#" << i << endl;
                    flag = 0;
                }
            }
        }
        else{
            success = 1;//failure when rows and cols not equal
        }

        if (success){//print success or error messages afterwards
            cout << "The Serial Image and the Parallel Image are NOT identical" << endl;
        }
        else{
            cout << "The Serial Image and the Parallel Image are identical" << endl;
        }
        
        //cv::waitKey(50000);//wait 50 seconds
    }
    MPI_Type_free(&Vec3b);//free up datatypes
    MPI_Type_free(&Row_Pixels);
    MPI_Finalize();//finalize program
    return 0;//exit
}