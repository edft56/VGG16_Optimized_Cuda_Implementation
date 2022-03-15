#include "atma.cuh"
#include "btdb.cuh"
#include "matrix_multiply.cuh"
#include "misc_vgg16.cuh"

#include <iostream>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



const int BATCH_SIZE  = 32;
const int NUM_CLASSES = 1000;
const int INPUT_SIZE  = BATCH_SIZE * 224 * 224 * 3;
const int WEIGHT_SIZE = 182475520;
const int BIAS_SIZE   = 13416;



const int channels[16] = { 3, 64,  64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 25088, 4096, 4096};
const int filters [16] = {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512,  4096, 4096, 1000};

constexpr btdb_forward_data btdb_f[13]
                                    {
                                        {  3, 224, BATCH_SIZE},    //0
                                        { 64, 224, BATCH_SIZE},    //1
                                        { 64, 112, BATCH_SIZE},    //2
                                        {128, 112, BATCH_SIZE},    //3
                                        {128,  56, BATCH_SIZE},    //4
                                        {256,  56, BATCH_SIZE},    //5
                                        {256,  56, BATCH_SIZE},    //6
                                        {256,  28, BATCH_SIZE},    //7
                                        {512,  28, BATCH_SIZE},    //8
                                        {512,  28, BATCH_SIZE},    //9
                                        {512,  14, BATCH_SIZE},    //10
                                        {512,  14, BATCH_SIZE},    //11
                                        {512,  14, BATCH_SIZE}     //12
                                    };

constexpr atma_forward_data atma_f[13]
                                    {
                                        { 64, 224, BATCH_SIZE, 0},  //0
                                        { 64, 224, BATCH_SIZE, 1},  //1
                                        {128, 112, BATCH_SIZE, 0},  //2
                                        {128, 112, BATCH_SIZE, 1},  //3
                                        {256,  56, BATCH_SIZE, 0},  //4
                                        {256,  56, BATCH_SIZE, 0},  //5
                                        {256,  56, BATCH_SIZE, 1},  //6
                                        {512,  28, BATCH_SIZE, 0},  //7
                                        {512,  28, BATCH_SIZE, 0},  //8
                                        {512,  28, BATCH_SIZE, 1},  //9
                                        {512,  14, BATCH_SIZE, 0},  //10
                                        {512,  14, BATCH_SIZE, 0},  //11
                                        {512,  14, BATCH_SIZE, 1}   //12    
                                    };

constexpr mm64x64x3_nn_data mm64x64x3_nn = { 3136, 3, 64, BATCH_SIZE};

constexpr mm64x64x8_nn_data mm64x64x8_nn[12]
                                            {
                                                {    3136,  64,  64, BATCH_SIZE     },   //0
                                                {     784,  64, 128, BATCH_SIZE     },   //1
                                                {     784, 128, 128, BATCH_SIZE     },   //2
                                                {196 *  4, 128, 256, BATCH_SIZE /  4},   //3
                                                {196 *  4, 256, 256, BATCH_SIZE /  4},   //4
                                                {196 *  4, 256, 256, BATCH_SIZE /  4},   //5
                                                { 49 *  8, 256, 512, BATCH_SIZE /  8},   //6
                                                { 49 * 16, 512, 512, BATCH_SIZE / 16},   //7
                                                { 49 * 16, 512, 512, BATCH_SIZE / 16},   //8
                                                { 16 *  4, 512, 512, BATCH_SIZE /  4},   //9
                                                { 16 *  4, 512, 512, BATCH_SIZE /  4},   //10
                                                { 16 *  4, 512, 512, BATCH_SIZE /  4}    //11
                                            };

constexpr mm32x32x8_nn_data mm32x32x8_nn[3]
                                            {
                                                {BATCH_SIZE, 25088,        4096,  true},
                                                {BATCH_SIZE,  4096,        4096,  true},
                                                {BATCH_SIZE,  4096, NUM_CLASSES, false}
                                            };

constexpr softmax_loss_data soft {BATCH_SIZE, NUM_CLASSES};





constexpr btdb_back_data btdb_b[12]
                                    {
                                        //{3,224,BATCH_SIZE},     // No need to compute 3channel btdb during training
                                        { 64, 224, BATCH_SIZE},    //0
                                        { 64, 112, BATCH_SIZE},    //1
                                        {128, 112, BATCH_SIZE},    //2
                                        {128,  56, BATCH_SIZE},    //3
                                        {256,  56, BATCH_SIZE},    //4
                                        {256,  56, BATCH_SIZE},    //5
                                        {256,  28, BATCH_SIZE},    //6
                                        {512,  28, BATCH_SIZE},    //7
                                        {512,  28, BATCH_SIZE},    //8
                                        {512,  14, BATCH_SIZE},    //9
                                        {512,  14, BATCH_SIZE},    //10
                                        {512,  14, BATCH_SIZE}     //11
                                    };

constexpr atma_back_data atma_b[13]
                                    {
                                        { 64, 224, BATCH_SIZE, 0},  //0
                                        { 64, 224, BATCH_SIZE, 1},  //1
                                        {128, 112, BATCH_SIZE, 0},  //2
                                        {128, 112, BATCH_SIZE, 1},  //3
                                        {256,  56, BATCH_SIZE, 0},  //4
                                        {256,  56, BATCH_SIZE, 0},  //5
                                        {256,  56, BATCH_SIZE, 1},  //6
                                        {512,  28, BATCH_SIZE, 0},  //7
                                        {512,  28, BATCH_SIZE, 0},  //8
                                        {512,  28, BATCH_SIZE, 1},  //9
                                        {512,  14, BATCH_SIZE, 0},  //10
                                        {512,  14, BATCH_SIZE, 0},  //11
                                        {512,  14, BATCH_SIZE, 1}   //12    
                                    };

constexpr mm3ch_back_data mm3ch_back = {3, 3136, 64, BATCH_SIZE};

constexpr mm_64x64x8_nt_data mm_64x64x8_nt[12]
                                            {   //tiles,filters,channels
                                                {    3136,  64,  64, BATCH_SIZE     },   //0
                                                {     784, 128,  64, BATCH_SIZE     },   //1 
                                                {     784, 128, 128, BATCH_SIZE     },   //2 
                                                {196 *  4, 256, 128, BATCH_SIZE /  4},   //3
                                                {196 *  4, 256, 256, BATCH_SIZE /  4},   //4
                                                {196 *  4, 256, 256, BATCH_SIZE /  4},   //5
                                                { 49 *  8, 512, 256, BATCH_SIZE /  8},   //6
                                                { 49 * 16, 512, 512, BATCH_SIZE / 16},   //7
                                                { 49 * 16, 512, 512, BATCH_SIZE / 16},   //8
                                                { 16 *  4, 512, 512, BATCH_SIZE /  4},   //9
                                                { 16 *  4, 512, 512, BATCH_SIZE /  4},   //10
                                                { 16 *  4, 512, 512, BATCH_SIZE /  4}    //11
                                            };

constexpr mm_64x64x8_tn_data mm_64x64x8_tn[12]
                                            {   //channels,tiles,filters
                                                { 64, 3136,  64, BATCH_SIZE},   //0
                                                { 64,  784, 128, BATCH_SIZE},   //1
                                                {128,  784, 128, BATCH_SIZE},   //2
                                                {128,  196, 256, BATCH_SIZE},   //3
                                                {256,  196, 256, BATCH_SIZE},   //4
                                                {256,  196, 256, BATCH_SIZE},   //5
                                                {256,   49, 512, BATCH_SIZE},   //6
                                                {512,   49, 512, BATCH_SIZE},   //7
                                                {512,   49, 512, BATCH_SIZE},   //8
                                                {512,   16, 512, BATCH_SIZE},   //9
                                                {512,   16, 512, BATCH_SIZE},   //10
                                                {512,   16, 512, BATCH_SIZE}    //11
                                            };                                            

constexpr mm32x32x32_nt_data mm32x32x32_nt[3]
                                            {   //batch_size,filters,iter
                                                {BATCH_SIZE,        4096, 25088},
                                                {BATCH_SIZE,        4096,  4096},
                                                {BATCH_SIZE, NUM_CLASSES,  4096}
                                            };

constexpr mm128x128x8_tn_data mm128x128x8_tn[3]
                                            {   //iter,batch size,filters
                                                {25088, BATCH_SIZE,        4096},
                                                { 4096, BATCH_SIZE,        4096},
                                                { 4096, BATCH_SIZE, NUM_CLASSES}
                                            };

constexpr weight_update_data w_up[16]
                                    {
                                        {mm3ch_back.OUT_SIZE},
                                        {mm_64x64x8_tn[0].OUT_SIZE},
                                        {mm_64x64x8_tn[1].OUT_SIZE},
                                        {mm_64x64x8_tn[2].OUT_SIZE},
                                        {mm_64x64x8_tn[3].OUT_SIZE},
                                        {mm_64x64x8_tn[4].OUT_SIZE},
                                        {mm_64x64x8_tn[5].OUT_SIZE},
                                        {mm_64x64x8_tn[6].OUT_SIZE},
                                        {mm_64x64x8_tn[7].OUT_SIZE},
                                        {mm_64x64x8_tn[8].OUT_SIZE},
                                        {mm_64x64x8_tn[9].OUT_SIZE},
                                        {mm_64x64x8_tn[10].OUT_SIZE},
                                        {mm_64x64x8_tn[11].OUT_SIZE},
                                        {mm128x128x8_tn[0].OUT_SIZE},
                                        {mm128x128x8_tn[1].OUT_SIZE},
                                        {mm128x128x8_tn[2].OUT_SIZE},
                                    };


constexpr weight_update_data b_up[16]
                                    {
                                        {atma_b[0].DBIAS_SIZE},
                                        {atma_b[1].DBIAS_SIZE},
                                        {atma_b[2].DBIAS_SIZE},
                                        {atma_b[3].DBIAS_SIZE},
                                        {atma_b[4].DBIAS_SIZE},
                                        {atma_b[5].DBIAS_SIZE},
                                        {atma_b[6].DBIAS_SIZE},
                                        {atma_b[7].DBIAS_SIZE},
                                        {atma_b[8].DBIAS_SIZE},
                                        {atma_b[9].DBIAS_SIZE},
                                        {atma_b[10].DBIAS_SIZE},
                                        {atma_b[11].DBIAS_SIZE},
                                        {atma_b[12].DBIAS_SIZE},
                                        {mm32x32x32_nt[0].DBIAS_SIZE},
                                        {mm32x32x32_nt[1].DBIAS_SIZE},
                                        {mm32x32x32_nt[2].DBIAS_SIZE},
                                    };




constexpr find_max_2d_data find_max {NUM_CLASSES};
constexpr find_accuracy_data fa{BATCH_SIZE,NUM_CLASSES};

constexpr weight_update_data w_up_nrm[16]
                                    {
                                        {  3 *  64 * 9},
                                        { 64 *  64 * 9},
                                        { 64 * 128 * 9},
                                        {128 * 128 * 9},
                                        {128 * 256 * 9},
                                        {256 * 256 * 9},
                                        {256 * 256 * 9},
                                        {256 * 512 * 9},
                                        {512 * 512 * 9},
                                        {512 * 512 * 9},
                                        {512 * 512 * 9},
                                        {512 * 512 * 9},
                                        {512 * 512 * 9},
                                        {mm128x128x8_tn[0].OUT_SIZE},
                                        {mm128x128x8_tn[1].OUT_SIZE},
                                        {mm128x128x8_tn[2].OUT_SIZE},
                                    };



void single_mem_alloc(char** alloc_end, char** alloc_start, size_t size_in_bytes){ // ALWAYS RETURNS A 256-BYTE ALIGNED POINTER (for performance reasons)
    *alloc_end = (char*)( (reinterpret_cast<std::uintptr_t>(*alloc_start) + size_in_bytes + 255) & ( ~(0xFF) ) );
}


template<const int i>
void convolution_forward(float* d_input, float* d_U, float* d_bias, char* d_pool_relu_idx, float* d_V, float* d_scratch, float* d_output, cudaStream_t stream = 0){

    btdb_forward<   btdb_f[i].CHANNELS            , btdb_f[i].HW      , btdb_f[i].BATCH_SIZE, btdb_f[i].IN_PAD, btdb_f[i].TILES, btdb_f[i].TILES_1D,
                    btdb_f[i].CHANNELS_THREADBLOCK, btdb_f[i].BLOCKS_X, btdb_f[i].BLOCKS_Y  , btdb_f[i].BLOCKS_Z>
                    ( d_input, d_V, stream );


    mm64x64x8_nn_ns::mm64x64x8_nn_wrapper<  mm64x64x8_nn[i-1].V_DIM    , mm64x64x8_nn[i-1].ITER_DIM   , mm64x64x8_nn[i-1].U_DIM   , mm64x64x8_nn[i-1].BATCH_SIZE,
                                            mm64x64x8_nn[i-1].SMEM_U_X , mm64x64x8_nn[i-1].SMEM_U_Y   , mm64x64x8_nn[i-1].SMEM_V_X, mm64x64x8_nn[i-1].SMEM_V_Y  ,
                                            mm64x64x8_nn[i-1].SMEM_SIZE, mm64x64x8_nn[i-1].SMEM_V_SIZE, mm64x64x8_nn[i-1].REG_V_Y , mm64x64x8_nn[i-1].REG_U_X   , 
                                            mm64x64x8_nn[i-1].THREADS_X, mm64x64x8_nn[i-1].BLOCKS_GMEM, mm64x64x8_nn[i-1].BLOCKS_X, mm64x64x8_nn[i-1].BLOCKS_Y  , 
                                            mm64x64x8_nn[i-1].BLOCKS_Z>
                                            ( d_V, d_U, d_scratch, stream );


    atma_forward<   atma_f[i].FILTERS, atma_f[i].HW                 , atma_f[i].IN_STRIDE, atma_f[i].TILES_X , 
                    atma_f[i].TILES  , atma_f[i].BLOCKS_X           , atma_f[i].BLOCKS_Y , atma_f[i].BLOCKS_Z, 
                    atma_f[i].OUT_CUT, atma_f[i].FILTERS_THREADBLOCK, atma_f[i].POOL>
                ( d_scratch, d_output, d_bias, d_pool_relu_idx, stream );

}


template<const int i>
void fully_connected_forward(float* d_input, float* d_U, float* d_bias, float* d_output, char* d_relu_idx, cudaStream_t stream = 0){

    mm32x32x8_nn_ns::mm32x32x8_nn_wrapper<  mm32x32x8_nn[i].V_DIM      , mm32x32x8_nn[i].ITER_DIM, mm32x32x8_nn[i].U_DIM   , mm32x32x8_nn[i].SMEM_U_X   ,
                                            mm32x32x8_nn[i].SMEM_U_Y   , mm32x32x8_nn[i].SMEM_V_X, mm32x32x8_nn[i].SMEM_V_Y, mm32x32x8_nn[i].SMEM_V_SIZE,
                                            mm32x32x8_nn[i].SMEM_U_SIZE, mm32x32x8_nn[i].REG_V_Y , mm32x32x8_nn[i].REG_U_X , mm32x32x8_nn[i].THREADS_X  ,
                                            mm32x32x8_nn[i].BLOCKS_GMEM, mm32x32x8_nn[i].BLOCKS_X, mm32x32x8_nn[i].BLOCKS_Y, mm32x32x8_nn[i].BLOCKS_Z   , 
                                            mm32x32x8_nn[i].RELU>
                                            ( d_input, d_U, d_output, d_bias, d_relu_idx, stream );
}


void convolution3ch_forward(float* d_input, float* d_U, float* d_bias, char* d_pool_relu_idx, float* d_V, float* d_scratch, float* d_output, cudaStream_t stream = 0){

    btdb_forward<   btdb_f[0].CHANNELS, btdb_f[0].HW                  , btdb_f[0].BATCH_SIZE, btdb_f[0].IN_PAD  , btdb_f[0].TILES,
                    btdb_f[0].TILES_1D, btdb_f[0].CHANNELS_THREADBLOCK, btdb_f[0].BLOCKS_X  , btdb_f[0].BLOCKS_Y, btdb_f[0].BLOCKS_Z>
                ( d_input, d_V, stream );
    

    mm64x64x3_nn_ns::mm64x64x3_nn_wrapper<  mm64x64x3_nn.V_DIM   , mm64x64x3_nn.ITER_DIM, mm64x64x3_nn.U_DIM      , mm64x64x3_nn.BATCH_SIZE ,
                                            mm64x64x3_nn.SMEM_V_Y, mm64x64x3_nn.SMEM_U_Y, mm64x64x3_nn.SMEM_V_SIZE, mm64x64x3_nn.SMEM_U_SIZE, 
                                            mm64x64x3_nn.REG_V_Y , mm64x64x3_nn.REG_U_X , mm64x64x3_nn.THREADS_X  , mm64x64x3_nn.BLOCKS_X   ,
                                            mm64x64x3_nn.BLOCKS_Y, mm64x64x3_nn.BLOCKS_Z>
                                         ( d_V, d_U, d_scratch, stream );

    
    atma_forward<   atma_f[0].FILTERS , atma_f[0].HW      , atma_f[0].IN_STRIDE, atma_f[0].TILES_X            , atma_f[0].TILES, atma_f[0].BLOCKS_X,
                    atma_f[0].BLOCKS_Y, atma_f[0].BLOCKS_Z, atma_f[0].OUT_CUT  , atma_f[0].FILTERS_THREADBLOCK, atma_f[0].POOL>
                ( d_scratch, d_output, d_bias, d_pool_relu_idx, stream );

}


void softmax_loss(float* __restrict__ d_scores, double* __restrict__ d_loss, char* __restrict__ d_truth_table, float* __restrict__ d_grads, cudaStream_t stream = 0){

    softmax_loss_call<soft.THREADS_X, soft.NUM_CLASSES, soft.BATCH_SIZE, soft.REG_NO, soft.BLOCKS>(d_scores, d_loss, d_truth_table, d_grads, stream);
}



void conv3ch_bias_act_relu_pool_back(float* d_V, float* d_dldu, char* d_pool_relu_idx, float* d_dbias, float* d_input, float* d_output, cudaStream_t stream = 0){

    atma_back<  atma_b[0].FILTERS  , atma_b[0].IN_DIM  , atma_b[0].BATCH_SIZE, atma_b[0].TILES_1D, atma_b[0].TILES, atma_b[0].OUT_CUT, 
                atma_b[0].THREADS_X, atma_b[0].BLOCKS_X, atma_b[0].BLOCKS_Y  , atma_b[0].BLOCKS_Z, atma_b[0].POOL>
                ( d_input, d_output, d_pool_relu_idx, d_dbias, stream );

    mm3ch_back_ns::mm3ch_back_wrapper<  mm3ch_back.V_DIM      , mm3ch_back.ITER_DIM , mm3ch_back.U_DIM      , mm3ch_back.BATCH_SIZE, mm3ch_back.SMEM_U_X, 
                                        mm3ch_back.SMEM_U_Y   , mm3ch_back.SMEM_SIZE, mm3ch_back.SMEM_V_SIZE, mm3ch_back.REG_V_Y   , mm3ch_back.REG_U_X , 
                                        mm3ch_back.BLOCKS_GMEM, mm3ch_back.THREADS_X, mm3ch_back.BLOCKS_X   , mm3ch_back.BLOCKS_Y  , mm3ch_back.BLOCKS_Z, 
                                        mm3ch_back.NUM        , mm3ch_back.DEN>
                                        ( d_V, d_output, d_dldu, stream );
}

template<const int i>
void conv_bias_act_relu_pool_back(  float* d_V, float* d_U, float* d_dldu, char* d_pool_relu_idx, float* d_dbias, float* d_input, float* d_output,
                                    cudaEvent_t event_conv, cudaStream_t primary_stream = 0, cudaStream_t secondary_stream = 0){

    //d_input and d_output are used as scratch space


    atma_back<  atma_b[i].FILTERS  , atma_b[i].IN_DIM  , atma_b[i].BATCH_SIZE, atma_b[i].TILES_1D, atma_b[i].TILES, atma_b[i].OUT_CUT, 
                atma_b[i].THREADS_X, atma_b[i].BLOCKS_X, atma_b[i].BLOCKS_Y  , atma_b[i].BLOCKS_Z, atma_b[i].POOL>
                ( d_input, d_output, d_pool_relu_idx, d_dbias, primary_stream );

    cudaEventRecord ( event_conv, primary_stream );
    
    mm_64x64x8_tn_ns::mm_64x64x8_tn_wrapper<mm_64x64x8_tn[i-1].V_DIM   , mm_64x64x8_tn[i-1].ITER_DIM, mm_64x64x8_tn[i-1].U_DIM      , mm_64x64x8_tn[i-1].BATCH_SIZE, mm_64x64x8_tn[i-1].SMEM_U_X   , 
                                            mm_64x64x8_tn[i-1].SMEM_U_Y, mm_64x64x8_tn[i-1].SMEM_V_Y, mm_64x64x8_tn[i-1].SMEM_V_X   , mm_64x64x8_tn[i-1].SMEM_SIZE , mm_64x64x8_tn[i-1].SMEM_V_SIZE, 
                                            mm_64x64x8_tn[i-1].REG_V_Y , mm_64x64x8_tn[i-1].REG_U_X , mm_64x64x8_tn[i-1].BLOCKS_GMEM, mm_64x64x8_tn[i-1].THREADS_X , mm_64x64x8_tn[i-1].THREADS_Y  , 
                                            mm_64x64x8_tn[i-1].BLOCKS_X, mm_64x64x8_tn[i-1].BLOCKS_Y, mm_64x64x8_tn[i-1].BLOCKS_Z   , mm_64x64x8_tn[i-1].NUM       , mm_64x64x8_tn[i-1].DEN>
                                            ( d_V,d_output,d_dldu,primary_stream );
 

    cudaStreamWaitEvent ( secondary_stream, event_conv);
    mm_64x64x8_nt_ns::mm_64x64x8_nt_wrapper<mm_64x64x8_nt[i-1].V_DIM   , mm_64x64x8_nt[i-1].ITER_DIM, mm_64x64x8_nt[i-1].U_DIM      , mm_64x64x8_nt[i-1].BATCH_SIZE, mm_64x64x8_nt[i-1].SMEM_V_Y   , 
                                            mm_64x64x8_nt[i-1].SMEM_V_X, mm_64x64x8_nt[i-1].SMEM_U_Y, mm_64x64x8_nt[i-1].SMEM_U_X   , mm_64x64x8_nt[i-1].SMEM_SIZE , mm_64x64x8_nt[i-1].SMEM_V_SIZE, 
                                            mm_64x64x8_nt[i-1].REG_V_Y , mm_64x64x8_nt[i-1].REG_U_X , mm_64x64x8_nt[i-1].BLOCKS_GMEM, mm_64x64x8_nt[i-1].THREADS_X , mm_64x64x8_nt[i-1].THREADS_Y  ,
                                            mm_64x64x8_nt[i-1].BLOCKS_X, mm_64x64x8_nt[i-1].BLOCKS_Y, mm_64x64x8_nt[i-1].BLOCKS_Z>
                                            ( d_output,d_U,d_input,secondary_stream );
    cudaEventRecord ( event_conv, secondary_stream );


    cudaStreamWaitEvent ( primary_stream, event_conv );
    btdb_backward<  btdb_b[i-1].CHANNELS, btdb_b[i-1].OUT_DIM , btdb_b[i-1].IN_PAD  , btdb_b[i-1].TILES_1D, btdb_b[i-1].TILES, btdb_b[i-1].THREADS_X,
                    btdb_b[i-1].stride  , btdb_b[i-1].BLOCKS_X, btdb_b[i-1].BLOCKS_Y, btdb_b[i-1].BLOCKS_Z>
                ( d_input,d_output, primary_stream );

}

template<const int i, const bool RELU>
void fully_connected_act_back(float* d_V, float* d_input, float* d_dldu, char* d_pool_relu_idx, float* d_U, float* d_output, float* d_dbias, cudaStream_t stream = 0){

    if(RELU == true){
        mm128x128x8_tn_ns::relu_back_wrap<mm128x128x8_tn[i].U_DIM, mm128x128x8_tn[i].ITER_DIM>( d_input,d_pool_relu_idx, stream );
    }

    mm128x128x8_tn_ns::mm128x128x8_tn_wrapper<  mm128x128x8_tn[i].V_DIM      , mm128x128x8_tn[i].U_DIM    , mm128x128x8_tn[i].ITER_DIM, mm128x128x8_tn[i].SMEM_V_Y   , 
                                                mm128x128x8_tn[i].SMEM_V_X   , mm128x128x8_tn[i].SMEM_U_X , mm128x128x8_tn[i].SMEM_U_Y, mm128x128x8_tn[i].SMEM_U_SIZE,
                                                mm128x128x8_tn[i].SMEM_V_SIZE, mm128x128x8_tn[i].REG_V_Y  , mm128x128x8_tn[i].REG_U_X , mm128x128x8_tn[i].BLOCKS_GMEM, 
                                                mm128x128x8_tn[i].OUT_SIZE   , mm128x128x8_tn[i].THREADS_X, mm128x128x8_tn[i].BLOCKS_X, mm128x128x8_tn[i].BLOCKS_Y   , 
                                                mm128x128x8_tn[i].BLOCKS_Z   , mm128x128x8_tn[i].NUM      , mm128x128x8_tn[i].DEN>
                                            ( d_V, d_input, d_dldu, stream );

    mm32x32x32_nt_ns::mm32x32x32_nt_wrapper<mm32x32x32_nt[i].V_DIM   , mm32x32x32_nt[i].U_DIM      , mm32x32x32_nt[i].ITER_DIM   , mm32x32x32_nt[i].SMEM_U_X   , mm32x32x32_nt[i].SMEM_U_Y, 
                                            mm32x32x32_nt[i].SMEM_V_Y, mm32x32x32_nt[i].SMEM_V_X   , mm32x32x32_nt[i].SMEM_U_SIZE, mm32x32x32_nt[i].SMEM_V_SIZE, mm32x32x32_nt[i].REG_V_Y , 
                                            mm32x32x32_nt[i].REG_U_X , mm32x32x32_nt[i].BLOCKS_GMEM, mm32x32x32_nt[i].OUT_SIZE   , mm32x32x32_nt[i].BLOCKS_X   , mm32x32x32_nt[i].BLOCKS_Y, 
                                            mm32x32x32_nt[i].BLOCKS_Z, mm32x32x32_nt[i].THREADS_X>
                                            ( d_input, d_U, d_output, d_dbias, stream );
}



void update_bias(float** d_bias, float** d_dbias, float** d_bias_velocity, double* d_loss, float* d_reg, float* d_learning_rate, float* d_momentum, cudaStream_t stream = 0){
    
    weight_update_call<b_up[ 0].ARRAY_SIZE, b_up[ 0].DATA_PER_THREAD, b_up[ 0].DATA_PER_BLOCK, b_up[ 0].THREADS_X, b_up[ 0].BLOCKS>( d_bias[ 0], d_dbias[ 0], d_bias_velocity[ 0], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 1].ARRAY_SIZE, b_up[ 1].DATA_PER_THREAD, b_up[ 1].DATA_PER_BLOCK, b_up[ 1].THREADS_X, b_up[ 1].BLOCKS>( d_bias[ 1], d_dbias[ 1], d_bias_velocity[ 1], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 2].ARRAY_SIZE, b_up[ 2].DATA_PER_THREAD, b_up[ 2].DATA_PER_BLOCK, b_up[ 2].THREADS_X, b_up[ 2].BLOCKS>( d_bias[ 2], d_dbias[ 2], d_bias_velocity[ 2], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 3].ARRAY_SIZE, b_up[ 3].DATA_PER_THREAD, b_up[ 3].DATA_PER_BLOCK, b_up[ 3].THREADS_X, b_up[ 3].BLOCKS>( d_bias[ 3], d_dbias[ 3], d_bias_velocity[ 3], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 4].ARRAY_SIZE, b_up[ 4].DATA_PER_THREAD, b_up[ 4].DATA_PER_BLOCK, b_up[ 4].THREADS_X, b_up[ 4].BLOCKS>( d_bias[ 4], d_dbias[ 4], d_bias_velocity[ 4], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 5].ARRAY_SIZE, b_up[ 5].DATA_PER_THREAD, b_up[ 5].DATA_PER_BLOCK, b_up[ 5].THREADS_X, b_up[ 5].BLOCKS>( d_bias[ 5], d_dbias[ 5], d_bias_velocity[ 5], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 6].ARRAY_SIZE, b_up[ 6].DATA_PER_THREAD, b_up[ 6].DATA_PER_BLOCK, b_up[ 6].THREADS_X, b_up[ 6].BLOCKS>( d_bias[ 6], d_dbias[ 6], d_bias_velocity[ 6], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 7].ARRAY_SIZE, b_up[ 7].DATA_PER_THREAD, b_up[ 7].DATA_PER_BLOCK, b_up[ 7].THREADS_X, b_up[ 7].BLOCKS>( d_bias[ 7], d_dbias[ 7], d_bias_velocity[ 7], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 8].ARRAY_SIZE, b_up[ 8].DATA_PER_THREAD, b_up[ 8].DATA_PER_BLOCK, b_up[ 8].THREADS_X, b_up[ 8].BLOCKS>( d_bias[ 8], d_dbias[ 8], d_bias_velocity[ 8], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[ 9].ARRAY_SIZE, b_up[ 9].DATA_PER_THREAD, b_up[ 9].DATA_PER_BLOCK, b_up[ 9].THREADS_X, b_up[ 9].BLOCKS>( d_bias[ 9], d_dbias[ 9], d_bias_velocity[ 9], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[10].ARRAY_SIZE, b_up[10].DATA_PER_THREAD, b_up[10].DATA_PER_BLOCK, b_up[10].THREADS_X, b_up[10].BLOCKS>( d_bias[10], d_dbias[10], d_bias_velocity[10], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[11].ARRAY_SIZE, b_up[11].DATA_PER_THREAD, b_up[11].DATA_PER_BLOCK, b_up[11].THREADS_X, b_up[11].BLOCKS>( d_bias[11], d_dbias[11], d_bias_velocity[11], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[12].ARRAY_SIZE, b_up[12].DATA_PER_THREAD, b_up[12].DATA_PER_BLOCK, b_up[12].THREADS_X, b_up[12].BLOCKS>( d_bias[12], d_dbias[12], d_bias_velocity[12], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[13].ARRAY_SIZE, b_up[13].DATA_PER_THREAD, b_up[13].DATA_PER_BLOCK, b_up[13].THREADS_X, b_up[13].BLOCKS>( d_bias[13], d_dbias[13], d_bias_velocity[13], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[14].ARRAY_SIZE, b_up[14].DATA_PER_THREAD, b_up[14].DATA_PER_BLOCK, b_up[14].THREADS_X, b_up[14].BLOCKS>( d_bias[14], d_dbias[14], d_bias_velocity[14], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<b_up[15].ARRAY_SIZE, b_up[15].DATA_PER_THREAD, b_up[15].DATA_PER_BLOCK, b_up[15].THREADS_X, b_up[15].BLOCKS>( d_bias[15], d_dbias[15], d_bias_velocity[15], d_loss, d_reg, d_learning_rate, d_momentum, stream);
}








void copy_inp_to_gpu_infer(float* weights, float* bias, float* input, char** d_weights, char** d_bias, char* d_input){

    const int channels[16] = {3,64,64,128,128,256,256,256,512,512,512,512,512,25088,4096,4096};
    const int filters[16] = {64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096,1000};
    

    int total_size = 0;
    for(int i=0; i<16; i++){
        int size = ( (i<13) ? channels[i]*filters[i]*36 : channels[i]*filters[i] );

        gpuErrchk( cudaMemcpy(d_weights[i], &weights[total_size],  size * sizeof(float), cudaMemcpyHostToDevice) );

        total_size += size;
    }

    total_size = 0;
    for(int i=0; i<16; i++){
        int size = filters[i];

        gpuErrchk( cudaMemcpy(d_bias[i], &bias[total_size],  size * sizeof(float), cudaMemcpyHostToDevice) );
        
        total_size += size;
    }
    
    gpuErrchk( cudaMemcpy(d_input, input,  INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice) );
}




void static_mem_allocs_infer  
                        (   char** memory_pool,
                            char** d_input_tensor,
                            char** d_U,
                            char** d_bias,
                            char** d_scratch1,
                            char** d_scratch2,
                            char** d_scratch3,
                            char** d_scratch4,
                            char** d_scores,
                            char** d_max,
                            int times_to_run
                        )
{

    size_t free_mem;
    size_t total;
    gpuErrchk(cudaMemGetInfo(&free_mem,&total));

    size_t mem_to_alloc = 0.95f*free_mem;

    gpuErrchk( cudaMalloc((void **) memory_pool, mem_to_alloc) );
    
    gpuErrchk(cudaMemGetInfo(&free_mem,&total));

    char* endpoint;
    endpoint = *memory_pool;


    *d_input_tensor = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, INPUT_SIZE * sizeof(float));


    for(int i=0; i<17; i++){ //allocate gpu memory for U inputs
        if(i<1){
            d_U[i] = endpoint;
        }
        else if (i<2){
            single_mem_alloc(&d_U[i], &d_U[i-1], mm3ch_back.OUT_SIZE*sizeof(float));
        }
        else if(i<14){
            single_mem_alloc(&d_U[i], &d_U[i-1], mm_64x64x8_nt[i-2].U_SIZE*sizeof(float)); 
        }
        else if(i<16){
            single_mem_alloc(&d_U[i], &d_U[i-1], mm32x32x32_nt[i-14].U_SIZE*sizeof(float)); 
        }
        else{
            single_mem_alloc(&endpoint, &d_U[i-1], mm32x32x32_nt[i-14].U_SIZE*sizeof(float));
        }
    }
    

    for(int i=0; i<17; i++){ //allocate gpu memory for bias
        if(i<1){
            d_bias[i] = endpoint;
        }
        else if(i<14){
            single_mem_alloc(&d_bias[i], &d_bias[i-1], atma_b[i-1].DBIAS_SIZE*sizeof(float));
        }
        else if(i<16){
            single_mem_alloc(&d_bias[i], &d_bias[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_bias[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
    }



    *d_scratch2 = endpoint;
    single_mem_alloc(&endpoint, &endpoint, 36*3136*64*BATCH_SIZE * sizeof(float));

    *d_scratch1 = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, 36*3136*64*BATCH_SIZE * sizeof(float));

    *d_scratch3 = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, 36*3136*64*BATCH_SIZE * sizeof(float));

    *d_scratch4 = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, 36*3136*64*BATCH_SIZE * sizeof(char));

    *d_scores = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, NUM_CLASSES*BATCH_SIZE*times_to_run * sizeof(float));

    *d_max = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, BATCH_SIZE*times_to_run * sizeof(int));
}






void vgg16_infer(   
                    float* input_buffer,
                    float* d_input_tensor,
                    float* d_U[16],
                    float* d_bias[16],
                    float* d_scratch1,
                    float* d_scratch2,
                    float* d_scratch3,
                    char* d_scratch4,
                    float* d_scores,
                    int* d_max,
                    int times_to_run
                )
{
    cudaStream_t mem_transfer_stream;
    cudaStreamCreate(&mem_transfer_stream);
    cudaStream_t seq_stream;
    cudaStreamCreate(&seq_stream);
    

    cudaEvent_t event1;
    cudaEventCreateWithFlags ( &event1, cudaEventDisableTiming );


    for(int times=0; times<times_to_run; times++){


        //INFERENCE


        convolution3ch_forward( (float*)d_input_tensor, (float*)d_U[0], (float*)d_bias[0], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1 ,seq_stream );
        
        cudaEventRecord ( event1, seq_stream );
        
        cudaStreamWaitEvent ( mem_transfer_stream, event1);
        if(times<times_to_run-1) gpuErrchk( cudaMemcpyAsync(d_input_tensor, &input_buffer[(times+1)*INPUT_SIZE], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, mem_transfer_stream) );


        convolution_forward< 1>( (float*)d_scratch1, (float*)d_U[ 1], (float*)d_bias[ 1], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 2>( (float*)d_scratch1, (float*)d_U[ 2], (float*)d_bias[ 2], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 3>( (float*)d_scratch1, (float*)d_U[ 3], (float*)d_bias[ 3], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 4>( (float*)d_scratch1, (float*)d_U[ 4], (float*)d_bias[ 4], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 5>( (float*)d_scratch1, (float*)d_U[ 5], (float*)d_bias[ 5], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 6>( (float*)d_scratch1, (float*)d_U[ 6], (float*)d_bias[ 6], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 7>( (float*)d_scratch1, (float*)d_U[ 7], (float*)d_bias[ 7], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 8>( (float*)d_scratch1, (float*)d_U[ 8], (float*)d_bias[ 8], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward< 9>( (float*)d_scratch1, (float*)d_U[ 9], (float*)d_bias[ 9], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward<10>( (float*)d_scratch1, (float*)d_U[10], (float*)d_bias[10], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward<11>( (float*)d_scratch1, (float*)d_U[11], (float*)d_bias[11], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward<12>( (float*)d_scratch1, (float*)d_U[12], (float*)d_bias[12], d_scratch4, (float*)d_scratch3, (float*)d_scratch2, (float*)d_scratch1 ,seq_stream );

        fully_connected_forward<0>( (float*)d_scratch1, (float*)d_U[13], (float*)d_bias[13], (float*)d_scratch2                             , d_scratch4 ,seq_stream);
        fully_connected_forward<1>( (float*)d_scratch2, (float*)d_U[14], (float*)d_bias[14], (float*)d_scratch3                             , d_scratch4 ,seq_stream);
        fully_connected_forward<2>( (float*)d_scratch3, (float*)d_U[15], (float*)d_bias[15], (float*)&d_scores[times*BATCH_SIZE*NUM_CLASSES], d_scratch4 ,seq_stream);
    }

    find_max_call<find_max.REGISTERS, find_max.THREADS_X, find_max.H>( (float*)d_scores, (int*)d_max, times_to_run*BATCH_SIZE, seq_stream );

    cudaStreamDestroy (mem_transfer_stream);
    cudaStreamDestroy (seq_stream);
    cudaEventDestroy  (event1);
}


void vgg16_main_infer(  float* input_buffer,
                        float* weights,
                        float* bias,
                        int* max,
                        int times_to_run
                    )
{

    char* memory_pool;
    char* d_input_tensor;
    char* d_U[16];
    char* d_bias[16];
    char* d_scratch1;
    char* d_scratch2;
    char* d_scratch3;
    char* d_scratch4;
    char* d_scores;
    char* d_max;

    static_mem_allocs_infer  
                        (   &memory_pool,
                            &d_input_tensor,
                            d_U,
                            d_bias,
                            &d_scratch1,
                            &d_scratch2,
                            &d_scratch3,
                            &d_scratch4,
                            &d_scores,
                            &d_max,
                            times_to_run
                        );

    copy_inp_to_gpu_infer(weights, bias, input_buffer, d_U, d_bias, d_input_tensor);



    vgg16_infer(
                    input_buffer,
                    (float*)d_input_tensor,
                    (float**)d_U,
                    (float**)d_bias,
                    (float*)d_scratch1,
                    (float*)d_scratch2,
                    (float*)d_scratch3,
                    d_scratch4,
                    (float*)d_scores,
                    (int*)d_max,
                    times_to_run
                );

    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(max, d_max, times_to_run*BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost) );

    cudaFree(memory_pool);
}












template<const int i>
void convolution_forward_normal(float* d_input, float* d_U, float* d_weights, float* d_bias, char* d_pool_relu_idx, float* d_V, float* d_scratch, float* d_output, cudaStream_t stream = 0){

    btdb_forward<   btdb_f[i].CHANNELS            , btdb_f[i].HW      , btdb_f[i].BATCH_SIZE, btdb_f[i].IN_PAD, btdb_f[i].TILES, btdb_f[i].TILES_1D,
                    btdb_f[i].CHANNELS_THREADBLOCK, btdb_f[i].BLOCKS_X, btdb_f[i].BLOCKS_Y  , btdb_f[i].BLOCKS_Z>
                ( d_input, d_V, stream );

    weight_trans<mm64x64x8_nn[i-1].ITER_DIM,mm64x64x8_nn[i-1].U_DIM>(d_weights,d_U,stream);

    mm64x64x8_nn_ns::mm64x64x8_nn_wrapper<  mm64x64x8_nn[i-1].V_DIM    , mm64x64x8_nn[i-1].ITER_DIM   , mm64x64x8_nn[i-1].U_DIM   , mm64x64x8_nn[i-1].BATCH_SIZE,
                                            mm64x64x8_nn[i-1].SMEM_U_X , mm64x64x8_nn[i-1].SMEM_U_Y   , mm64x64x8_nn[i-1].SMEM_V_X, mm64x64x8_nn[i-1].SMEM_V_Y  , 
                                            mm64x64x8_nn[i-1].SMEM_SIZE, mm64x64x8_nn[i-1].SMEM_V_SIZE, mm64x64x8_nn[i-1].REG_V_Y , mm64x64x8_nn[i-1].REG_U_X   , 
                                            mm64x64x8_nn[i-1].THREADS_X, mm64x64x8_nn[i-1].BLOCKS_GMEM, mm64x64x8_nn[i-1].BLOCKS_X, mm64x64x8_nn[i-1].BLOCKS_Y  , 
                                            mm64x64x8_nn[i-1].BLOCKS_Z>
                                        ( d_V, d_U, d_scratch, stream );


    atma_forward<   atma_f[i].FILTERS , atma_f[i].HW      , atma_f[i].IN_STRIDE, atma_f[i].TILES_X            , atma_f[i].TILES, atma_f[i].BLOCKS_X,
                    atma_f[i].BLOCKS_Y, atma_f[i].BLOCKS_Z, atma_f[i].OUT_CUT  , atma_f[i].FILTERS_THREADBLOCK, atma_f[i].POOL>
                ( d_scratch, d_output, d_bias, d_pool_relu_idx, stream );

}

void convolution3ch_forward_normal(float* d_input, float* d_U, float* d_weights, float* d_bias, char* d_pool_relu_idx, float* d_V, float* d_scratch, float* d_output, cudaStream_t stream = 0){

    btdb_forward<   btdb_f[0].CHANNELS            , btdb_f[0].HW      , btdb_f[0].BATCH_SIZE, btdb_f[0].IN_PAD, btdb_f[0].TILES, btdb_f[0].TILES_1D,
                    btdb_f[0].CHANNELS_THREADBLOCK, btdb_f[0].BLOCKS_X, btdb_f[0].BLOCKS_Y  , btdb_f[0].BLOCKS_Z>
                ( d_input, d_V, stream );
    
    weight_trans<mm64x64x3_nn.ITER_DIM, mm64x64x3_nn.U_DIM>( d_weights, d_U, stream );

    mm64x64x3_nn_ns::mm64x64x3_nn_wrapper<  mm64x64x3_nn.V_DIM   , mm64x64x3_nn.ITER_DIM, mm64x64x3_nn.U_DIM      , mm64x64x3_nn.BATCH_SIZE ,
                                            mm64x64x3_nn.SMEM_V_Y, mm64x64x3_nn.SMEM_U_Y, mm64x64x3_nn.SMEM_V_SIZE, mm64x64x3_nn.SMEM_U_SIZE,
                                            mm64x64x3_nn.REG_V_Y , mm64x64x3_nn.REG_U_X , mm64x64x3_nn.THREADS_X  , mm64x64x3_nn.BLOCKS_X   ,
                                            mm64x64x3_nn.BLOCKS_Y, mm64x64x3_nn.BLOCKS_Z>
                                         ( d_V, d_U, d_scratch, stream );

    
    atma_forward<   atma_f[0].FILTERS , atma_f[0].HW      , atma_f[0].IN_STRIDE, atma_f[0].TILES_X            , atma_f[0].TILES, atma_f[0].BLOCKS_X,
                    atma_f[0].BLOCKS_Y, atma_f[0].BLOCKS_Z, atma_f[0].OUT_CUT  , atma_f[0].FILTERS_THREADBLOCK, atma_f[0].POOL>
                ( d_scratch, d_output, d_bias, d_pool_relu_idx, stream );

}

void conv3ch_bias_act_relu_pool_back_normal(float* d_V, float* d_dweights, float* d_dldu_scratch, char* d_pool_relu_idx, float* d_dbias, float* d_input, float* d_output, cudaStream_t stream = 0){

    atma_back<  atma_b[0].FILTERS  , atma_b[0].IN_DIM  , atma_b[0].BATCH_SIZE, atma_b[0].TILES_1D, atma_b[0].TILES, atma_b[0].OUT_CUT, 
                atma_b[0].THREADS_X, atma_b[0].BLOCKS_X, atma_b[0].BLOCKS_Y  , atma_b[0].BLOCKS_Z, atma_b[0].POOL>
                ( d_input, d_output, d_pool_relu_idx, d_dbias, stream );

    mm3ch_back_ns::mm3ch_back_wrapper<  mm3ch_back.V_DIM      , mm3ch_back.ITER_DIM , mm3ch_back.U_DIM      , mm3ch_back.BATCH_SIZE, mm3ch_back.SMEM_U_X, 
                                        mm3ch_back.SMEM_U_Y   , mm3ch_back.SMEM_SIZE, mm3ch_back.SMEM_V_SIZE, mm3ch_back.REG_V_Y   , mm3ch_back.REG_U_X , 
                                        mm3ch_back.BLOCKS_GMEM, mm3ch_back.THREADS_X, mm3ch_back.BLOCKS_X   , mm3ch_back.BLOCKS_Y  , mm3ch_back.BLOCKS_Z, 
                                        mm3ch_back.NUM        , mm3ch_back.DEN>
                                     (  d_V, d_output, d_dldu_scratch, stream );

    weight_trans_back_call<mm64x64x3_nn.ITER_DIM, mm64x64x3_nn.U_DIM>( d_dldu_scratch, d_dweights, stream );
}

template<const int i>
void conv_bias_act_relu_pool_back_normal(float* d_V, float* d_U, float* d_dweights, float* d_dldu_scratch, char* d_pool_relu_idx, float* d_dbias, float* d_input, float* d_output,
                                         cudaEvent_t event_conv, cudaStream_t primary_stream = 0, cudaStream_t secondary_stream = 0){

    //d_input and d_output are used as scratch space


    atma_back<  atma_b[i].FILTERS  , atma_b[i].IN_DIM  , atma_b[i].BATCH_SIZE, atma_b[i].TILES_1D, atma_b[i].TILES, atma_b[i].OUT_CUT, 
                atma_b[i].THREADS_X, atma_b[i].BLOCKS_X, atma_b[i].BLOCKS_Y  , atma_b[i].BLOCKS_Z, atma_b[i].POOL>
             ( d_input, d_output, d_pool_relu_idx, d_dbias, primary_stream );

    cudaEventRecord ( event_conv, primary_stream );
    
    mm_64x64x8_tn_ns::mm_64x64x8_tn_wrapper<mm_64x64x8_tn[i-1].V_DIM   , mm_64x64x8_tn[i-1].ITER_DIM, mm_64x64x8_tn[i-1].U_DIM      , mm_64x64x8_tn[i-1].BATCH_SIZE, mm_64x64x8_tn[i-1].SMEM_U_X   ,
                                    mm_64x64x8_tn[i-1].SMEM_U_Y, mm_64x64x8_tn[i-1].SMEM_V_Y, mm_64x64x8_tn[i-1].SMEM_V_X   , mm_64x64x8_tn[i-1].SMEM_SIZE , mm_64x64x8_tn[i-1].SMEM_V_SIZE, 
                                    mm_64x64x8_tn[i-1].REG_V_Y , mm_64x64x8_tn[i-1].REG_U_X , mm_64x64x8_tn[i-1].BLOCKS_GMEM, mm_64x64x8_tn[i-1].THREADS_X , mm_64x64x8_tn[i-1].THREADS_Y  , 
                                    mm_64x64x8_tn[i-1].BLOCKS_X, mm_64x64x8_tn[i-1].BLOCKS_Y, mm_64x64x8_tn[i-1].BLOCKS_Z   , mm_64x64x8_tn[i-1].NUM       , mm_64x64x8_tn[i-1].DEN>
                                    (d_V, d_output, d_dldu_scratch, primary_stream);
 

    cudaStreamWaitEvent ( secondary_stream, event_conv);
    mm_64x64x8_nt_ns::mm_64x64x8_nt_wrapper<mm_64x64x8_nt[i-1].V_DIM   , mm_64x64x8_nt[i-1].ITER_DIM, mm_64x64x8_nt[i-1].U_DIM      , mm_64x64x8_nt[i-1].BATCH_SIZE, mm_64x64x8_nt[i-1].SMEM_V_Y   , 
                                    mm_64x64x8_nt[i-1].SMEM_V_X, mm_64x64x8_nt[i-1].SMEM_U_Y, mm_64x64x8_nt[i-1].SMEM_U_X   , mm_64x64x8_nt[i-1].SMEM_SIZE , mm_64x64x8_nt[i-1].SMEM_V_SIZE, 
                                    mm_64x64x8_nt[i-1].REG_V_Y , mm_64x64x8_nt[i-1].REG_U_X , mm_64x64x8_nt[i-1].BLOCKS_GMEM, mm_64x64x8_nt[i-1].THREADS_X , mm_64x64x8_nt[i-1].THREADS_Y  , 
                                    mm_64x64x8_nt[i-1].BLOCKS_X, mm_64x64x8_nt[i-1].BLOCKS_Y, mm_64x64x8_nt[i-1].BLOCKS_Z>
                                   ( d_output, d_U, d_input, secondary_stream );
    cudaEventRecord ( event_conv, secondary_stream );

    cudaStreamWaitEvent ( primary_stream, event_conv);

    btdb_backward<  btdb_b[i-1].CHANNELS , btdb_b[i-1].OUT_DIM, btdb_b[i-1].IN_PAD  , btdb_b[i-1].TILES_1D, btdb_b[i-1].TILES,
                    btdb_b[i-1].THREADS_X, btdb_b[i-1].stride , btdb_b[i-1].BLOCKS_X, btdb_b[i-1].BLOCKS_Y, btdb_b[i-1].BLOCKS_Z>
                 ( d_input,d_output, primary_stream );

    weight_trans_back_call<mm64x64x8_nn[i-1].ITER_DIM, mm64x64x8_nn[i-1].U_DIM>( d_dldu_scratch, d_dweights, primary_stream );

}


void update_weights_normal(float** d_weights, float** d_dweights, float** d_velocity, double* d_loss, float* d_reg, float* d_learning_rate, float* d_momentum, cudaStream_t stream = 0){
    
    weight_update_call<w_up_nrm[ 0].ARRAY_SIZE, w_up_nrm[ 0].DATA_PER_THREAD, w_up_nrm[ 0].DATA_PER_BLOCK, w_up_nrm[ 0].THREADS_X, w_up_nrm[ 0].BLOCKS>( d_weights[ 0], d_dweights[ 0], d_velocity[ 0], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 1].ARRAY_SIZE, w_up_nrm[ 1].DATA_PER_THREAD, w_up_nrm[ 1].DATA_PER_BLOCK, w_up_nrm[ 1].THREADS_X, w_up_nrm[ 1].BLOCKS>( d_weights[ 1], d_dweights[ 1], d_velocity[ 1], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 2].ARRAY_SIZE, w_up_nrm[ 2].DATA_PER_THREAD, w_up_nrm[ 2].DATA_PER_BLOCK, w_up_nrm[ 2].THREADS_X, w_up_nrm[ 2].BLOCKS>( d_weights[ 2], d_dweights[ 2], d_velocity[ 2], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 3].ARRAY_SIZE, w_up_nrm[ 3].DATA_PER_THREAD, w_up_nrm[ 3].DATA_PER_BLOCK, w_up_nrm[ 3].THREADS_X, w_up_nrm[ 3].BLOCKS>( d_weights[ 3], d_dweights[ 3], d_velocity[ 3], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 4].ARRAY_SIZE, w_up_nrm[ 4].DATA_PER_THREAD, w_up_nrm[ 4].DATA_PER_BLOCK, w_up_nrm[ 4].THREADS_X, w_up_nrm[ 4].BLOCKS>( d_weights[ 4], d_dweights[ 4], d_velocity[ 4], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 5].ARRAY_SIZE, w_up_nrm[ 5].DATA_PER_THREAD, w_up_nrm[ 5].DATA_PER_BLOCK, w_up_nrm[ 5].THREADS_X, w_up_nrm[ 5].BLOCKS>( d_weights[ 5], d_dweights[ 5], d_velocity[ 5], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 6].ARRAY_SIZE, w_up_nrm[ 6].DATA_PER_THREAD, w_up_nrm[ 6].DATA_PER_BLOCK, w_up_nrm[ 6].THREADS_X, w_up_nrm[ 6].BLOCKS>( d_weights[ 6], d_dweights[ 6], d_velocity[ 6], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 7].ARRAY_SIZE, w_up_nrm[ 7].DATA_PER_THREAD, w_up_nrm[ 7].DATA_PER_BLOCK, w_up_nrm[ 7].THREADS_X, w_up_nrm[ 7].BLOCKS>( d_weights[ 7], d_dweights[ 7], d_velocity[ 7], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 8].ARRAY_SIZE, w_up_nrm[ 8].DATA_PER_THREAD, w_up_nrm[ 8].DATA_PER_BLOCK, w_up_nrm[ 8].THREADS_X, w_up_nrm[ 8].BLOCKS>( d_weights[ 8], d_dweights[ 8], d_velocity[ 8], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[ 9].ARRAY_SIZE, w_up_nrm[ 9].DATA_PER_THREAD, w_up_nrm[ 9].DATA_PER_BLOCK, w_up_nrm[ 9].THREADS_X, w_up_nrm[ 9].BLOCKS>( d_weights[ 9], d_dweights[ 9], d_velocity[ 9], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[10].ARRAY_SIZE, w_up_nrm[10].DATA_PER_THREAD, w_up_nrm[10].DATA_PER_BLOCK, w_up_nrm[10].THREADS_X, w_up_nrm[10].BLOCKS>( d_weights[10], d_dweights[10], d_velocity[10], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[11].ARRAY_SIZE, w_up_nrm[11].DATA_PER_THREAD, w_up_nrm[11].DATA_PER_BLOCK, w_up_nrm[11].THREADS_X, w_up_nrm[11].BLOCKS>( d_weights[11], d_dweights[11], d_velocity[11], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[12].ARRAY_SIZE, w_up_nrm[12].DATA_PER_THREAD, w_up_nrm[12].DATA_PER_BLOCK, w_up_nrm[12].THREADS_X, w_up_nrm[12].BLOCKS>( d_weights[12], d_dweights[12], d_velocity[12], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[13].ARRAY_SIZE, w_up_nrm[13].DATA_PER_THREAD, w_up_nrm[13].DATA_PER_BLOCK, w_up_nrm[13].THREADS_X, w_up_nrm[13].BLOCKS>( d_weights[13], d_dweights[13], d_velocity[13], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[14].ARRAY_SIZE, w_up_nrm[14].DATA_PER_THREAD, w_up_nrm[14].DATA_PER_BLOCK, w_up_nrm[14].THREADS_X, w_up_nrm[14].BLOCKS>( d_weights[14], d_dweights[14], d_velocity[14], d_loss, d_reg, d_learning_rate, d_momentum, stream);
    weight_update_call<w_up_nrm[15].ARRAY_SIZE, w_up_nrm[15].DATA_PER_THREAD, w_up_nrm[15].DATA_PER_BLOCK, w_up_nrm[15].THREADS_X, w_up_nrm[15].BLOCKS>( d_weights[15], d_dweights[15], d_velocity[15], d_loss, d_reg, d_learning_rate, d_momentum, stream);
}



void static_mem_allocs_normal  
                        (   char** memory_pool,
                            char** d_input_tensor,
                            char** d_truth_table,
                            char** d_V,
                            char** d_U,
                            char** d_weights,
                            char** d_dbias,
                            char** d_pool_relu_idx,
                            char** d_dweights,
                            char** d_velocity,
                            char** d_bias,
                            char** d_bias_velocity,
                            char** d_loss,
                            char** d_learning_rate,
                            char** d_regularization,
                            char** d_bias_reg,
                            char** d_momentum,
                            char** d_scratch1,
                            char** d_scratch2,
                            char** d_dldu_scratch,
                            char** d_max,
                            char** d_accuracy,
                            int times_to_run
                        )
{

    const int channels[16] = {3,64,64,128,128,256,256,256,512,512,512,512,512,25088,4096,4096};
    const int filters[16] = {64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096,1000};

    size_t free_mem;
    size_t total;
    gpuErrchk(cudaMemGetInfo(&free_mem,&total));

    size_t mem_to_alloc = 0.97f*free_mem;

    gpuErrchk( cudaMalloc((void **) memory_pool, mem_to_alloc) );
    
    gpuErrchk(cudaMemGetInfo(&free_mem,&total));

    char* endpoint;
    endpoint = *memory_pool;


    *d_input_tensor = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, INPUT_SIZE * sizeof(float));

    *d_truth_table = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, BATCH_SIZE*NUM_CLASSES * sizeof(char));

    
    for(int i=0; i<17; i++){ //allocate gpu memory for V inputs
        if(i<1){
            d_V[i] = endpoint;
        }
        else if(i<2){
            single_mem_alloc(&d_V[i], &d_V[i-1], mm3ch_back.V_SIZE*sizeof(float));
        }
        else if(i<14){
            single_mem_alloc(&d_V[i], &d_V[i-1], mm_64x64x8_tn[i-2].V_SIZE*sizeof(float));
        }
        else if(i<16){
            single_mem_alloc(&d_V[i], &d_V[i-1], mm128x128x8_tn[i-14].V_SIZE*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_V[i-1], mm128x128x8_tn[i-14].V_SIZE*sizeof(float));
        }
    }


    for(int i=0; i<14; i++){ //allocate gpu memory for transformed weights
        if(i<1){
            d_U[i] = endpoint;
        }
        else if (i<13){
            single_mem_alloc(&d_U[i], &d_U[i-1], channels[i-1]*filters[i-1]*36*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_U[i-1], channels[i-1]*filters[i-1]*36*sizeof(float));
        }
    }


    for(int i=0; i<17; i++){ //allocate gpu memory for non transformed weights
        if(i<1){
            d_weights[i] = endpoint;
        }
        else if (i<16){
            single_mem_alloc(&d_weights[i], &d_weights[i-1], channels[i-1]*filters[i-1]*((i-1<13)?9:1)*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_weights[i-1], channels[i-1]*filters[i-1]*((i-1<13)?9:1)*sizeof(float));
        }
    }


    for(int i=0; i<17; i++){ //allocate gpu memory for velocity
        if(i<1){
            d_velocity[i] = endpoint;
        }
        else if (i<16){
            single_mem_alloc(&d_velocity[i], &d_velocity[i-1], channels[i-1]*filters[i-1]*((i-1<13)?9:1)*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_velocity[i-1], channels[i-1]*filters[i-1]*((i-1<13)?9:1)*sizeof(float)); 
        }
    }

    

    for(int i=0; i<17; i++){ //allocate gpu memory for bias gradients
        if(i<1){
            d_dbias[i] = endpoint;
        }
        else if(i<14){
            single_mem_alloc(&d_dbias[i], &d_dbias[i-1], atma_b[i-1].DBIAS_SIZE*sizeof(float));
        }
        else if(i<16){
            single_mem_alloc(&d_dbias[i], &d_dbias[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_dbias[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
    }

    

    for(int i=0; i<17; i++){ //allocate gpu memory for bias
        if(i<1){
            d_bias[i] = endpoint;
        }
        else if(i<14){
            single_mem_alloc(&d_bias[i], &d_bias[i-1], atma_b[i-1].DBIAS_SIZE*sizeof(float));
        }
        else if(i<16){
            single_mem_alloc(&d_bias[i], &d_bias[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_bias[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
    }

    for(int i=0; i<17; i++){ //allocate gpu memory for bias velocity
        if(i<1){
            d_bias_velocity[i] = endpoint;
        }
        else if(i<14){
            single_mem_alloc(&d_bias_velocity[i], &d_bias_velocity[i-1], atma_b[i-1].DBIAS_SIZE*sizeof(float));
        }
        else if(i<16){
            single_mem_alloc(&d_bias_velocity[i], &d_bias_velocity[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_bias_velocity[i-1], mm32x32x32_nt[i-14].DBIAS_SIZE*sizeof(float));
        }
    }


    for(int i=0; i<17; i++){ //allocate gpu memory for pool / relu map
        if(i<1){
            d_pool_relu_idx[i] = endpoint;
        }
        else if(i<14){
            single_mem_alloc(&d_pool_relu_idx[i], &d_pool_relu_idx[i-1], atma_b[i-1].PR_IDX_SIZE * sizeof(char));
        }
        else if(i<16){
            single_mem_alloc(&d_pool_relu_idx[i], &d_pool_relu_idx[i-1], mm32x32x32_nt[i-14].PR_IDX_SIZE * sizeof(char));
        }
        else{
            single_mem_alloc(&endpoint, &d_pool_relu_idx[i-1], mm32x32x32_nt[i-14].PR_IDX_SIZE * sizeof(char));
        }
    }


    for(int i=0; i<17; i++){ //allocate gpu memory for weight gradients
        if(i<1){
            d_dweights[i] = endpoint;
        }
        else if(i<16){
            single_mem_alloc(&d_dweights[i], &d_dweights[i-1], channels[i-1]*filters[i-1]*((i-1<13)?9:1)*sizeof(float));
        }
        else{
            single_mem_alloc(&endpoint, &d_dweights[i-1], channels[i-1]*filters[i-1]*((i-1<13)?9:1)*sizeof(float));
        }
    }

    *d_loss = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, 1 * sizeof(double));

    *d_bias_reg = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, 1 * sizeof(float));

    *d_learning_rate = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, 1 * sizeof(float));

    *d_regularization = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, 1 * sizeof(float));

    *d_momentum = endpoint; 
    single_mem_alloc(&endpoint, &endpoint, 1 * sizeof(float));



    *d_scratch2 = endpoint; //these 2 will be used as scratch space
    single_mem_alloc(&endpoint, &endpoint, 36*3136*64*BATCH_SIZE * sizeof(float));


    *d_scratch1 = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, 36*3136*64*BATCH_SIZE * sizeof(float));

    *d_dldu_scratch = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, 512*512*36 * sizeof(float));

    *d_max = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, BATCH_SIZE * sizeof(int));

    *d_accuracy = endpoint;                                        
    single_mem_alloc(&endpoint, &endpoint, 1 * sizeof(float));

    gpuErrchk( cudaMemset(*d_accuracy, 0, 1*sizeof(float)) );
}



void copy_inp_to_gpu_normal(float* weights, float* weight_velocity, float* bias, float* bias_velocity, float* input, char* truth_table, float* regularization, float* momentum, float* learning_rate,
char** d_weights, char** d_weight_velocity, char** d_bias, char** d_bias_velocity, char* d_input, char* d_truth_table, char* d_regularization, char* d_momentum, char* d_learning_rate){

    const int channels[16] = {3,64,64,128,128,256,256,256,512,512,512,512,512,25088,4096,4096};
    const int filters[16] = {64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096,1000};
    

    int total_size = 0;
    for(int i=0; i<16; i++){
        int size = ( (i<13) ? channels[i]*filters[i]*9 : channels[i]*filters[i] );

        gpuErrchk( cudaMemcpy( d_weights[i], &weights[total_size],  size * sizeof(float), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_weight_velocity[i], &weight_velocity[total_size],  size * sizeof(float), cudaMemcpyHostToDevice) );

        total_size += size;
    }

    total_size = 0;
    for(int i=0; i<16; i++){
        int size = filters[i];

        gpuErrchk( cudaMemcpy(d_bias[i], &bias[total_size],  size * sizeof(float), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_bias_velocity[i], &bias_velocity[total_size],  size * sizeof(float), cudaMemcpyHostToDevice) );
        
        total_size += size;
    }
    

    gpuErrchk( cudaMemcpy(d_input, input,  INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_truth_table, truth_table,  BATCH_SIZE*NUM_CLASSES * sizeof(char), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_regularization, regularization,  1 * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_momentum, momentum,  1 * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_learning_rate, learning_rate,  1 * sizeof(float), cudaMemcpyHostToDevice) );
}




void copy_everything_to_cpu_normal(float* weights, float* weight_velocity, float* bias, float* bias_velocity,char** d_weights, char** d_weight_velocity, char** d_bias, char** d_bias_velocity){
    const int channels[16] = {3,64,64,128,128,256,256,256,512,512,512,512,512,25088,4096,4096};
    const int filters[16] = {64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096,1000};
    

    int total_size = 0;
    for(int i=0; i<16; i++){
        int size = ( (i<13) ? channels[i]*filters[i]*9 : channels[i]*filters[i] );

        gpuErrchk( cudaMemcpy(&weights[total_size], d_weights[i], size * sizeof(float), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&weight_velocity[total_size], d_weight_velocity[i], size * sizeof(float), cudaMemcpyDeviceToHost) );

        total_size += size;
    }

    total_size = 0;
    for(int i=0; i<16; i++){
        int size = filters[i];

        gpuErrchk( cudaMemcpy(&bias[total_size], d_bias[i], size * sizeof(float), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&bias_velocity[total_size], d_bias_velocity[i], size * sizeof(float), cudaMemcpyDeviceToHost) );
        
        total_size += size;
    }
}

void copy_weights_to_cpu_normal(float* weights, float* bias,char** d_weights, char** d_bias){
    const int channels[16] = {3,64,64,128,128,256,256,256,512,512,512,512,512,25088,4096,4096};
    const int filters[16] = {64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096,1000};
    

    int total_size = 0;
    for(int i=0; i<16; i++){
        int size = ( (i<13) ? channels[i]*filters[i]*9 : channels[i]*filters[i] );

        gpuErrchk( cudaMemcpy(&weights[total_size], d_weights[i], size * sizeof(float), cudaMemcpyDeviceToHost) );
        
        total_size += size;
    }

    total_size = 0;
    for(int i=0; i<16; i++){
        int size = filters[i];

        gpuErrchk( cudaMemcpy(&bias[total_size], d_bias[i], size * sizeof(float), cudaMemcpyDeviceToHost) );
        
        total_size += size;
    }
}





//6.41GB required
void vgg16_train_normal(   
                    float* input_buffer, //host
                    char* truth_table_buffer, //host
                    double* loss,
                    char* d_input_tensor,
                    char* d_weights[16],
                    char* d_V[16],
                    char* d_U[13],
                    char* d_velocity[16],
                    char* d_dweights[16],
                    char* d_dbias[16],
                    char* d_bias[16],
                    char* d_bias_velocity[16],
                    char* d_pool_relu_idx[16],
                    char* d_truth_table,
                    char* d_loss,
                    char* d_learning_rate,
                    char* d_regularization,
                    char* d_bias_reg,
                    char* d_momentum,
                    char* d_scratch1,
                    char* d_scratch2,
                    char* d_dldu_scratch,
                    char* d_max,
                    char* d_accuracy,
                    int times_to_run
                )
{
    cudaStream_t mem_transfer_stream;
    cudaStreamCreate(&mem_transfer_stream);
    cudaStream_t seq_stream;
    cudaStreamCreate(&seq_stream);
    cudaStream_t sec_stream;
    cudaStreamCreate(&sec_stream);
    cudaStream_t bias_up_stream;
    cudaStreamCreate(&bias_up_stream);
    

    cudaEvent_t event1;
    cudaEventCreateWithFlags ( &event1, cudaEventDisableTiming );

    cudaEvent_t event_conv;
    cudaEventCreateWithFlags ( &event_conv, cudaEventDisableTiming );

    cudaEvent_t event_batch_end;
    cudaEventCreateWithFlags ( &event_batch_end, cudaEventDisableTiming );

    gpuErrchk( cudaMemsetAsync( d_bias_reg, 0, 1*sizeof(float), seq_stream ) );

    for(int times=0; times<times_to_run; times++){

        if(times>0) {
            cudaStreamWaitEvent ( seq_stream, event_batch_end);
            gpuErrchk( cudaMemsetAsync( d_loss, 0, 1*sizeof(double), seq_stream ) );
        }

        //INFERENCE


        convolution3ch_forward_normal((float*)d_input_tensor, (float*)d_U[0], (float*)d_weights[0], (float*)d_bias[0], d_pool_relu_idx[0], (float*)d_V[0], (float*)d_scratch2, (float*)d_scratch1 ,seq_stream);
        
        cudaEventRecord ( event1, seq_stream );
        
        cudaStreamWaitEvent ( mem_transfer_stream, event1);
        if(times<times_to_run-1) gpuErrchk( cudaMemcpyAsync(d_input_tensor, &input_buffer[(times+1)*INPUT_SIZE], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, mem_transfer_stream) );

        convolution_forward_normal< 1>( (float*)d_scratch1, (float*)d_U[ 1], (float*)d_weights[ 1], (float*)d_bias[ 1], d_pool_relu_idx[ 1], (float*)d_V[ 1], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 2>( (float*)d_scratch1, (float*)d_U[ 2], (float*)d_weights[ 2], (float*)d_bias[ 2], d_pool_relu_idx[ 2], (float*)d_V[ 2], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 3>( (float*)d_scratch1, (float*)d_U[ 3], (float*)d_weights[ 3], (float*)d_bias[ 3], d_pool_relu_idx[ 3], (float*)d_V[ 3], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 4>( (float*)d_scratch1, (float*)d_U[ 4], (float*)d_weights[ 4], (float*)d_bias[ 4], d_pool_relu_idx[ 4], (float*)d_V[ 4], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 5>( (float*)d_scratch1, (float*)d_U[ 5], (float*)d_weights[ 5], (float*)d_bias[ 5], d_pool_relu_idx[ 5], (float*)d_V[ 5], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 6>( (float*)d_scratch1, (float*)d_U[ 6], (float*)d_weights[ 6], (float*)d_bias[ 6], d_pool_relu_idx[ 6], (float*)d_V[ 6], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 7>( (float*)d_scratch1, (float*)d_U[ 7], (float*)d_weights[ 7], (float*)d_bias[ 7], d_pool_relu_idx[ 7], (float*)d_V[ 7], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 8>( (float*)d_scratch1, (float*)d_U[ 8], (float*)d_weights[ 8], (float*)d_bias[ 8], d_pool_relu_idx[ 8], (float*)d_V[ 8], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal< 9>( (float*)d_scratch1, (float*)d_U[ 9], (float*)d_weights[ 9], (float*)d_bias[ 9], d_pool_relu_idx[ 9], (float*)d_V[ 9], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal<10>( (float*)d_scratch1, (float*)d_U[10], (float*)d_weights[10], (float*)d_bias[10], d_pool_relu_idx[10], (float*)d_V[10], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal<11>( (float*)d_scratch1, (float*)d_U[11], (float*)d_weights[11], (float*)d_bias[11], d_pool_relu_idx[11], (float*)d_V[11], (float*)d_scratch2, (float*)d_scratch1, seq_stream );
        convolution_forward_normal<12>( (float*)d_scratch1, (float*)d_U[12], (float*)d_weights[12], (float*)d_bias[12], d_pool_relu_idx[12], (float*)d_V[12], (float*)d_scratch2, (float*)d_V[13]   , seq_stream );


        fully_connected_forward<0>( (float*)d_V[13], (float*)d_weights[13], (float*)d_bias[13], (float*)d_V[14],    d_pool_relu_idx[13], seq_stream );
        fully_connected_forward<1>( (float*)d_V[14], (float*)d_weights[14], (float*)d_bias[14], (float*)d_V[15],    d_pool_relu_idx[14], seq_stream );
        fully_connected_forward<2>( (float*)d_V[15], (float*)d_weights[15], (float*)d_bias[15], (float*)d_scratch1, d_pool_relu_idx[15], seq_stream );
    
        softmax_loss((float*)d_scratch1, (double*) d_loss, d_truth_table, (float*)d_scratch2 ,seq_stream);
        find_accuracy_call<find_max.REGISTERS,find_max.THREADS_X,NUM_CLASSES,fa.MAX_ARRAY_SIZE,fa.THREADS_X,fa.REGISTERS>((float*)d_scratch1,(int*)d_max,BATCH_SIZE,d_truth_table,times_to_run*BATCH_SIZE,(float*)d_accuracy,seq_stream);
        cudaEventRecord ( event1, seq_stream );

        
        
        cudaStreamWaitEvent ( mem_transfer_stream, event1);
        if(times<times_to_run-1) gpuErrchk( cudaMemcpyAsync(d_truth_table, &truth_table_buffer[(times+1)*NUM_CLASSES*BATCH_SIZE], NUM_CLASSES*BATCH_SIZE * sizeof(char), cudaMemcpyHostToDevice, mem_transfer_stream) );


        //BACKPROPAGATION

        
        fully_connected_act_back<2,false>( (float*)d_V[15], (float*)d_scratch2, (float*)d_dweights[15], d_pool_relu_idx[15], (float*)d_weights[15], (float*)d_scratch1, (float*)d_dbias[15] ,seq_stream );
        fully_connected_act_back<1, true>( (float*)d_V[14], (float*)d_scratch1, (float*)d_dweights[14], d_pool_relu_idx[14], (float*)d_weights[14], (float*)d_scratch2, (float*)d_dbias[14] ,seq_stream );
        fully_connected_act_back<0, true>( (float*)d_V[13], (float*)d_scratch2, (float*)d_dweights[13], d_pool_relu_idx[13], (float*)d_weights[13], (float*)d_scratch1, (float*)d_dbias[13] ,seq_stream );

        conv_bias_act_relu_pool_back_normal<12>( (float*)d_V[12], (float*)d_U[12], (float*)d_dweights[12], (float*)d_dldu_scratch, d_pool_relu_idx[12], (float*)d_dbias[12], (float*)d_scratch1, (float*)d_scratch2, event_conv, seq_stream, sec_stream );
        conv_bias_act_relu_pool_back_normal<11>( (float*)d_V[11], (float*)d_U[11], (float*)d_dweights[11], (float*)d_dldu_scratch, d_pool_relu_idx[11], (float*)d_dbias[11], (float*)d_scratch2, (float*)d_scratch1, event_conv, seq_stream, sec_stream );
        conv_bias_act_relu_pool_back_normal<10>( (float*)d_V[10], (float*)d_U[10], (float*)d_dweights[10], (float*)d_dldu_scratch, d_pool_relu_idx[10], (float*)d_dbias[10], (float*)d_scratch1, (float*)d_scratch2, event_conv, seq_stream, sec_stream );
        conv_bias_act_relu_pool_back_normal< 9>( (float*)d_V[ 9], (float*)d_U[ 9], (float*)d_dweights[ 9], (float*)d_dldu_scratch, d_pool_relu_idx[ 9], (float*)d_dbias[ 9], (float*)d_scratch2, (float*)d_scratch1, event_conv, seq_stream, sec_stream );
        conv_bias_act_relu_pool_back_normal< 8>( (float*)d_V[ 8], (float*)d_U[ 8], (float*)d_dweights[ 8], (float*)d_dldu_scratch, d_pool_relu_idx[ 8], (float*)d_dbias[ 8], (float*)d_scratch1, (float*)d_scratch2, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 7>( (float*)d_V[ 7], (float*)d_U[ 7], (float*)d_dweights[ 7], (float*)d_dldu_scratch, d_pool_relu_idx[ 7], (float*)d_dbias[ 7], (float*)d_scratch2, (float*)d_scratch1, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 6>( (float*)d_V[ 6], (float*)d_U[ 6], (float*)d_dweights[ 6], (float*)d_dldu_scratch, d_pool_relu_idx[ 6], (float*)d_dbias[ 6], (float*)d_scratch1, (float*)d_scratch2, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 5>( (float*)d_V[ 5], (float*)d_U[ 5], (float*)d_dweights[ 5], (float*)d_dldu_scratch, d_pool_relu_idx[ 5], (float*)d_dbias[ 5], (float*)d_scratch2, (float*)d_scratch1, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 4>( (float*)d_V[ 4], (float*)d_U[ 4], (float*)d_dweights[ 4], (float*)d_dldu_scratch, d_pool_relu_idx[ 4], (float*)d_dbias[ 4], (float*)d_scratch1, (float*)d_scratch2, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 3>( (float*)d_V[ 3], (float*)d_U[ 3], (float*)d_dweights[ 3], (float*)d_dldu_scratch, d_pool_relu_idx[ 3], (float*)d_dbias[ 3], (float*)d_scratch2, (float*)d_scratch1, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 2>( (float*)d_V[ 2], (float*)d_U[ 2], (float*)d_dweights[ 2], (float*)d_dldu_scratch, d_pool_relu_idx[ 2], (float*)d_dbias[ 2], (float*)d_scratch1, (float*)d_scratch2, event_conv, seq_stream, sec_stream );        
        conv_bias_act_relu_pool_back_normal< 1>( (float*)d_V[ 1], (float*)d_U[ 1], (float*)d_dweights[ 1], (float*)d_dldu_scratch, d_pool_relu_idx[ 1], (float*)d_dbias[ 1], (float*)d_scratch2, (float*)d_scratch1, event_conv, seq_stream, sec_stream );

        conv3ch_bias_act_relu_pool_back((float*)d_V[0], (float*)d_dweights[0], d_pool_relu_idx[0], (float*)d_dbias[0], (float*)d_scratch1, (float*)d_scratch2 ,seq_stream);



        cudaEventRecord ( event1, seq_stream );

        cudaStreamWaitEvent ( bias_up_stream, event1);

        update_weights_normal((float**)d_weights, (float**)d_dweights, (float**)d_velocity, (double*)d_loss, (float*)d_regularization, (float*)d_learning_rate, (float*)d_momentum, seq_stream);
        update_bias((float**)d_bias, (float**)d_dbias, (float**)d_bias_velocity, (double*)d_loss, (float*)d_bias_reg, (float*)d_learning_rate, (float*)d_momentum, bias_up_stream);

        cudaEventRecord ( event_batch_end, bias_up_stream );
    }

    cudaStreamDestroy (mem_transfer_stream);
    cudaStreamDestroy (seq_stream);
    cudaStreamDestroy (sec_stream);
    cudaStreamDestroy (bias_up_stream);

    cudaEventDestroy (event1);
    cudaEventDestroy (event_conv);
    cudaEventDestroy (event_batch_end);
}




void vgg16_main_normal(    float* input_buffer,
                    char* truth_table_buffer,
                    float* weights,
                    float* weight_velocity,
                    float* bias,
                    float* bias_velocity,
                    double* loss,
                    float* momentum,
                    float* regularization,
                    float* learning_rate,
                    float* accuracy,
                    int times_to_run,
                    int output_mode=2
                )
{

    char* memory_pool;
    char* d_input_tensor;
    char* d_V[16];
    char* d_U[13];
    char* d_velocity[16];
    char* d_dweights[16];
    char* d_weights[16];
    char* d_dbias[16];
    char* d_bias[16];
    char* d_bias_velocity[16];
    char* d_pool_relu_idx[16];
    char* d_truth_table;
    char* d_loss;
    char* d_learning_rate;
    char* d_regularization;
    char* d_bias_reg;
    char* d_momentum;
    char* d_scratch1;
    char* d_scratch2;
    char* d_dldu_scratch;
    char* d_max;
    char* d_accuracy;

    static_mem_allocs_normal  
                        (   &memory_pool,
                            &d_input_tensor,
                            &d_truth_table,
                            d_V,
                            d_U,
                            d_weights,
                            d_dbias,
                            d_pool_relu_idx,
                            d_dweights,
                            d_velocity,
                            d_bias,
                            d_bias_velocity,
                            &d_loss,
                            &d_learning_rate,
                            &d_regularization,
                            &d_bias_reg,
                            &d_momentum,
                            &d_scratch1,
                            &d_scratch2,
                            &d_dldu_scratch,
                            &d_max,
                            &d_accuracy,
                            times_to_run
                        );

    copy_inp_to_gpu_normal(weights, weight_velocity, bias, bias_velocity, input_buffer, truth_table_buffer, regularization, momentum, learning_rate,
                    d_weights, d_velocity, d_bias, d_bias_velocity, d_input_tensor, d_truth_table, d_regularization, d_momentum, d_learning_rate);



    vgg16_train_normal(
                    input_buffer,
                    truth_table_buffer,
                    loss,
                    d_input_tensor,
                    d_weights,
                    d_V,
                    d_U,
                    d_velocity,
                    d_dweights,
                    d_dbias,
                    d_bias,
                    d_bias_velocity,
                    d_pool_relu_idx,
                    d_truth_table,
                    d_loss,
                    d_learning_rate,
                    d_regularization,
                    d_bias_reg,
                    d_momentum,
                    d_scratch1,
                    d_scratch2,
                    d_dldu_scratch,
                    d_max,
                    d_accuracy,
                    times_to_run
                );

    gpuErrchk( cudaDeviceSynchronize() );

    if(output_mode==1){
        copy_weights_to_cpu_normal(weights, bias, d_weights, d_bias);
    }
    else if(output_mode==2){
        copy_everything_to_cpu_normal(weights, weight_velocity, bias, bias_velocity, d_weights, d_velocity, d_bias, d_bias_velocity);
    }

    gpuErrchk( cudaMemcpy(loss, d_loss, 1 * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(accuracy, d_accuracy, 1 * sizeof(float), cudaMemcpyDeviceToHost) );

    //printf("%lf ",loss[0]);

    cudaFree(memory_pool);
}














//EXPECTS WEIGHTS IN HWCN AND OUTPUTS SAME FORMAT
void transform_weights(float* d_weights, float* d_U){
    const int channels[16] = { 3, 64,  64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 25088, 4096, 4096};
    const int filters [16] = {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512,  4096, 4096, 1000};

    int trans_weight_index[16] = {0};
    int weight_index      [16] = {0};

    for(int i=1; i<16; i++){
        trans_weight_index[i] = trans_weight_index[i-1] + channels[i-1]*filters[i-1]*( (i-1 < 13) ? 36 : 1 );
        weight_index[i]       = weight_index[i-1]       + channels[i-1]*filters[i-1]*( (i-1 < 13) ?  9 : 1 );
    }


    weight_trans<  3,  64>( &d_weights[weight_index[ 0]], (float*)&d_U[trans_weight_index[ 0]] );
    weight_trans< 64,  64>( &d_weights[weight_index[ 1]], (float*)&d_U[trans_weight_index[ 1]] );
    weight_trans< 64, 128>( &d_weights[weight_index[ 2]], (float*)&d_U[trans_weight_index[ 2]] );
    weight_trans<128, 128>( &d_weights[weight_index[ 3]], (float*)&d_U[trans_weight_index[ 3]] );
    weight_trans<128, 256>( &d_weights[weight_index[ 4]], (float*)&d_U[trans_weight_index[ 4]] );
    weight_trans<256, 256>( &d_weights[weight_index[ 5]], (float*)&d_U[trans_weight_index[ 5]] );
    weight_trans<256, 256>( &d_weights[weight_index[ 6]], (float*)&d_U[trans_weight_index[ 6]] );
    weight_trans<256, 512>( &d_weights[weight_index[ 7]], (float*)&d_U[trans_weight_index[ 7]] );
    weight_trans<512, 512>( &d_weights[weight_index[ 8]], (float*)&d_U[trans_weight_index[ 8]] );
    weight_trans<512, 512>( &d_weights[weight_index[ 9]], (float*)&d_U[trans_weight_index[ 9]] );
    weight_trans<512, 512>( &d_weights[weight_index[10]], (float*)&d_U[trans_weight_index[10]] );
    weight_trans<512, 512>( &d_weights[weight_index[11]], (float*)&d_U[trans_weight_index[11]] );
    weight_trans<512, 512>( &d_weights[weight_index[12]], (float*)&d_U[trans_weight_index[12]] );

    gpuErrchk( cudaMemcpy( (float*)&d_U[trans_weight_index[13]], &d_weights[weight_index[13]],       25088 * 4096 * sizeof(float), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemcpy( (float*)&d_U[trans_weight_index[14]], &d_weights[weight_index[14]],        4096 * 4096 * sizeof(float), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemcpy( (float*)&d_U[trans_weight_index[15]], &d_weights[weight_index[15]], NUM_CLASSES * 4096 * sizeof(float), cudaMemcpyDeviceToDevice) );
}


//EXPECTS WEIGHTS IN HWCN AND OUTPUTS SAME FORMAT
void transform_weights_call(float* weights, float* trans_weights){
    const int weight_size       = 138344128;
    const int trans_weight_size = 182475520;

    float* d_weights;
    float* d_trans_weights;

    gpuErrchk( cudaMalloc((void **) &d_weights, weight_size*sizeof(float)) );
    gpuErrchk( cudaMalloc((void **) &d_trans_weights, trans_weight_size*sizeof(float)) );

    gpuErrchk( cudaMemcpy((float*)d_weights, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice) );

    transform_weights(d_weights, d_trans_weights);

    gpuErrchk( cudaMemcpy((float*)trans_weights, d_trans_weights, trans_weight_size * sizeof(float), cudaMemcpyDeviceToHost) );

    cudaFree(d_weights);
    cudaFree(d_trans_weights);
}
