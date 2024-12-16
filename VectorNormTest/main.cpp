#include <iostream>
#include <vector>
#include <cmath>

#include "Testers/CorrectnessTester.h"
#include "Testers/DataBaseErrorCheck.h"
#include "Testers/KNNtester.h"
#include "Utils/ReadDatFile.h"

#define NORM_PARAMS MAX_RESIDUAL,VECT_TYPE
int main(int argc, char* argv[])
{
    // DataBaseErrorCheck<float> dbec("data_set_train.dat");
    // KNNtester<VECT_TYPE> knnTester("dpedia.raw");
    // std::vector<std::vector<VECT_TYPE>> test_set = knnTester.splitTestTrain();
    // test_set = std::vector<std::vector<VECT_TYPE>>(test_set.begin(),test_set.begin()+10);
    // // knnTester.print_vectors();
    // auto results  = knnTester.getRecallStatistics<SingleSqrt>(test_set,5);
    // // readDatFile("data_set_train.dat",test_set);
    // std::cout<<"recall: "<<results["recall"]<<", precision: "<<results["precision"]<<std::endl;
    // std::cout<<"tp: "<<results["tp"]<<", fp: "<<results["fp"]<<std::endl;

    CorrectnessTester<NORM_PARAMS,64,512,SingleSqrt> ::test_vectors();
    CorrectnessTester<NORM_PARAMS,64,512,SingleRsqrt> ::test_vectors();
    CorrectnessTester<NORM_PARAMS,64,512,SingleScalar> ::test_vectors();
    // auto results = dbec.getErrorStatistics<SingleRsqrt>();
    // std::cout<<"mean error: "<<results["mean"]<<std::endl;
    // std::cout<<"error std: "<<results["std"]<<std::endl;
    // float ones[] = {1,1,1,1}; 

}


