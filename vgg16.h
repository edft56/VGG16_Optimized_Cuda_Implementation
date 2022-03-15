#pragma once

const int BATCH_SIZE = 32;
const int NUM_CLASSES = 1000;
const int INPUT_SIZE = BATCH_SIZE * 224 * 224 * 3;
const int WEIGHT_SIZE = 182475520; //transformed weight size
const int WEIGHT_SIZE_ORGNL = 138344128;
const int BIAS_SIZE = 13416;


void vgg16_main_infer(  float*  input_buffer,    //input
                        float*  weights,         //input
                        float*  bias,            //input
                        int*    max,             //output
                        int     times_to_run     //input
                    );


void vgg16_main_normal(
                    float*  input_buffer,        //input
                    char*   truth_table_buffer,  //input
                    float*  weights,             //input/output
                    float*  weight_velocity,     //input/output
                    float*  bias,                //input/output
                    float*  bias_velocity,       //input/output
                    double* loss,                //output
                    float*  momentum,            //input
                    float*  regularization,      //input
                    float*  learning_rate,       //input
                    float*  accuracy,            //output
                    int     times_to_run,        //input
                    int     output_mode=2        //input
                );


void transform_weights_call(float* weights, float* trans_weights);
