#pragma once

#include <cmath>

struct weight_update_data{ //ARRAY_SIZE
    const int ARRAY_SIZE;

    const int DATA_PER_THREAD = 64;
    const int WARPS_PER_BLOCK = 1;
    const int THREADS_X = WARPS_PER_BLOCK*32;
    const int DATA_PER_BLOCK = WARPS_PER_BLOCK*32*DATA_PER_THREAD;
    const int BLOCKS = (ARRAY_SIZE + DATA_PER_BLOCK - 1) / DATA_PER_BLOCK;
};

struct softmax_loss_data{ //BATCH SIZE, NUM CLASSES
    const int BATCH_SIZE; const int NUM_CLASSES;
    const int THREADS_X = 32;
    const int REG_NO = (NUM_CLASSES + THREADS_X - 1) / THREADS_X; //data per thread
    const int BLOCKS = BATCH_SIZE;
};

struct find_max_2d_data{ //N,H(find max for H)
    const int H;

    const int REGISTERS = 128;
    const int THREADS_X = REGISTERS/4;

};

struct find_accuracy_data{
    const int MAX_ARRAY_SIZE = 32;
    const int CLASSES = 1000;

    
    const int REGISTERS = 32;
    const int THREADS_X = 32;
    const int BLOCKS_X = 1;
};


__inline__ __device__ float warpReduceXor(float value){
    for(int i=32/2; i>0; i/=2){
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    }
    return value;
}

__inline__ __device__ float warpReduceDown(float value){
    for(int i=32/2; i>0; i/=2){
        value += __shfl_down_sync(0xffffffff, value, i, 32);
    }
    return value;
}

__inline__ __device__ float warpCompare(float value){
    for(int i=32/2; i>0; i/=2){
        float temp = __shfl_xor_sync(0xffffffff, value, i, 32);
        if ( temp > value ) value = temp;
    }
    return value;
}


template<const int ARRAY_SIZE, const int DATA_PER_THREAD, const int DATA_PER_BLOCK, const int THREADS_X, const int BLOCKS>
__global__ __launch_bounds__(THREADS_X,1) void weight_update(float* __restrict__ weights, const float* __restrict__ dweights, float* __restrict__ velocity, double* loss, float* reg, float* learning_rate, float* momentum){
    float weights_reg[DATA_PER_THREAD]  = {0};
    float velocity_reg[DATA_PER_THREAD] = {0};
    float dweights_reg[DATA_PER_THREAD] = {0};
    float lr_reg;
    float momentum_reg;
    float reg_reg;

    float temp_sum[4] = {0};
    
    #pragma unroll
    for(int i=0; i<DATA_PER_THREAD/4; i++){
        bool bound_check = (blockIdx.x!=BLOCKS-1) || (threadIdx.x + i*THREADS_X < ( 1 + (ARRAY_SIZE-1)%DATA_PER_BLOCK )/4);
        if(bound_check) reinterpret_cast<float4*>(weights_reg)[i]  = reinterpret_cast<const float4*>(weights)[blockIdx.x*DATA_PER_BLOCK/4 + threadIdx.x + i*THREADS_X];
        if(bound_check) reinterpret_cast<float4*>(dweights_reg)[i] = reinterpret_cast<const float4*>(dweights)[blockIdx.x*DATA_PER_BLOCK/4 + threadIdx.x + i*THREADS_X];
        if(bound_check) reinterpret_cast<float4*>(velocity_reg)[i] = reinterpret_cast<const float4*>(velocity)[blockIdx.x*DATA_PER_BLOCK/4 + threadIdx.x + i*THREADS_X];
    }
    lr_reg       = learning_rate[0];
    momentum_reg = momentum[0];
    reg_reg      = reg[0];


    #pragma unroll
    for(int i=0; i<DATA_PER_THREAD/4; i++){
        temp_sum[0]      += weights_reg[i*4] * weights_reg[i*4];
        velocity_reg[i*4] = velocity_reg[i*4]*momentum_reg - lr_reg*(dweights_reg[i*4] + reg_reg*weights_reg[i*4]);
        weights_reg[i*4] += velocity_reg[i*4];

        temp_sum[1]          += weights_reg[i*4 + 1] * weights_reg[i*4 + 1];
        velocity_reg[i*4 + 1] = velocity_reg[i*4 + 1]*momentum_reg - lr_reg*(dweights_reg[i*4 + 1] + reg_reg*weights_reg[i*4 + 1]);
        weights_reg[i*4 + 1] += velocity_reg[i*4 + 1];

        temp_sum[2]          += weights_reg[i*4 + 2] * weights_reg[i*4 + 2];
        velocity_reg[i*4 + 2] = velocity_reg[i*4 + 2]*momentum_reg - lr_reg*(dweights_reg[i*4 + 2] + reg_reg*weights_reg[i*4 + 2]);
        weights_reg[i*4 + 2] += velocity_reg[i*4 + 2];

        temp_sum[3]          += weights_reg[i*4 + 3] * weights_reg[i*4 + 3];
        velocity_reg[i*4 + 3] = velocity_reg[i*4 + 3]*momentum_reg - lr_reg*(dweights_reg[i*4 + 3] + reg_reg*weights_reg[i*4 + 3]);
        weights_reg[i*4 + 3] += velocity_reg[i*4 + 3];
    }

    float thread_sum = temp_sum[0] + temp_sum[1] + temp_sum[2] + temp_sum[3];

    float warp_sum   = warpReduceDown(thread_sum);
    double warp_loss = reg_reg * 0.5 * warp_sum;

    if(threadIdx.x%32==0) atomicAdd(loss,warp_loss);

    #pragma unroll
    for(int i=0; i<DATA_PER_THREAD/4; i++){
        bool bound_check = (blockIdx.x!=BLOCKS-1) || (threadIdx.x + i*THREADS_X < ( 1 + (ARRAY_SIZE-1)%DATA_PER_BLOCK )/4);
        if(bound_check) reinterpret_cast<float4*>(weights)[blockIdx.x*DATA_PER_BLOCK/4 + threadIdx.x + i*THREADS_X]  = reinterpret_cast<const float4*>(weights_reg)[i];
        if(bound_check) reinterpret_cast<float4*>(velocity)[blockIdx.x*DATA_PER_BLOCK/4 + threadIdx.x + i*THREADS_X] = reinterpret_cast<const float4*>(velocity_reg)[i];
    }
}


template<const int THREADS_X, const int NUM_CLASSES, const int BATCH_SIZE, const int REG_NO>
__global__ void softmax_loss_gpu(const float* __restrict__ scores, double* __restrict__ loss, const char* __restrict__ truth_table, float* __restrict__ grads){

    float scores_reg[REG_NO] = {-INFINITY};
    char truth_reg[REG_NO]   = {0};
    

    for(int i=0; i<REG_NO/4; i++){ 
        bool bound_check = threadIdx.x + i*REG_NO < NUM_CLASSES/4;

        if(bound_check) reinterpret_cast<float4*>(scores_reg)[i] = reinterpret_cast<const float4*>(scores)[blockIdx.x*NUM_CLASSES/4 + threadIdx.x + i*REG_NO];

        if(bound_check) reinterpret_cast<char4*>(truth_reg)[i]   = reinterpret_cast<const char4*>(truth_table)[blockIdx.x*NUM_CLASSES/4 + threadIdx.x + i*REG_NO];
    }

    float max_thread_score = scores_reg[0];
    for(int i=1; i<REG_NO; i++){
        if (scores_reg[i] > max_thread_score) max_thread_score = scores_reg[i];
    }


    float max_warp_score = warpCompare(max_thread_score);


    float thread_exp_sum = 0;
    for(int i=0; i<REG_NO; i++){
        float temp = scores_reg[i] - max_warp_score;
        
        thread_exp_sum += expf(temp);
    }

    
    float warp_exp_sum = warpReduceXor(thread_exp_sum);
    

    float grads_temp[4];
    float thread_loss = 0;
    for(int i=0; i<REG_NO/4; i++){
        for(int j=0; j<4; j++){
            float softmax = expf(scores_reg[i*4 + j] - max_warp_score) / warp_exp_sum;
            grads_temp[j] = (softmax - truth_reg[i*4 + j]) / BATCH_SIZE;

            thread_loss += truth_reg[i*4 + j] * ( -scores_reg[i*4 + j] + max_warp_score + logf(warp_exp_sum) );
        }
        bool bound_check = threadIdx.x + i*REG_NO < NUM_CLASSES/4;
        if(bound_check) reinterpret_cast<float4*>(grads)[blockIdx.x*NUM_CLASSES/4 + threadIdx.x + i*REG_NO] = reinterpret_cast<float4*>(grads_temp)[0];
    }

    float warp_loss = warpReduceDown(thread_loss);

    if(threadIdx.x%32 == 0) atomicAdd(loss, (double)warp_loss/(double)BATCH_SIZE); //Cast to double in order to avoid possible underflow

}


template<const int ARRAY_SIZE, const int DATA_PER_THREAD, const int DATA_PER_BLOCK, const int THREADS_X, const int BLOCKS>
void weight_update_call(float* __restrict__ d_weights, const float* __restrict__ d_dweights, float* __restrict__ d_velocity, double* d_loss, float* d_reg, float* d_learning_rate, float* d_momentum, cudaStream_t stream = 0){

    weight_update<ARRAY_SIZE, DATA_PER_THREAD, DATA_PER_BLOCK, THREADS_X, BLOCKS><<<BLOCKS , THREADS_X, 0, stream>>>
    (d_weights,d_dweights,d_velocity,d_loss,d_reg,d_learning_rate,d_momentum);
}


template<const int THREADS_X, const int NUM_CLASSES, const int BATCH_SIZE, const int REG_NO, const int BLOCKS>
void softmax_loss_call(const float* __restrict__ d_scores, double* __restrict__ d_loss, const char* __restrict__ d_truth_table, float* __restrict__ d_grads, cudaStream_t stream = 0){

    softmax_loss_gpu<THREADS_X,NUM_CLASSES,BATCH_SIZE,REG_NO><<<BLOCKS,THREADS_X,0,stream>>>(d_scores,d_loss,d_truth_table,d_grads);
}


//input NxH and we want the max in the H dimension
template<const int REGISTERS, const int THREADS_X, const int H>
__global__ __launch_bounds__(32,4) void find_max_2d_vec4(const float* __restrict__ array, int* max, const int N){
    int idx = blockIdx.x*THREADS_X*H/4 + threadIdx.x;

    float regs[REGISTERS];
    __shared__ float smem[THREADS_X*REGISTERS + THREADS_X*4];
    int max_idx = -1;
    float max_value = -INFINITY;

    for(int j=0; j<(H+REGISTERS-1)/REGISTERS; j++){
        #pragma unroll
        for(int i=0; i<THREADS_X; i++){ 
            if(threadIdx.x + j*THREADS_X<H/4) reinterpret_cast<float4*>(regs)[i] = reinterpret_cast<const float4*>(array)[idx + i*H/4 + j*THREADS_X];
        }

        #pragma unroll
        for(int i=0; i<THREADS_X; i++){
            reinterpret_cast<float4*>(smem)[threadIdx.x + i*(THREADS_X+1)] = reinterpret_cast<float4*>(regs)[i];
        }

        #pragma unroll
        for(int i=0; i<THREADS_X; i++){
            reinterpret_cast<float4*>(regs)[i] = reinterpret_cast<float4*>(smem)[threadIdx.x*(THREADS_X+1) + i];
        }

        #pragma unroll
        for(int i=0; i<REGISTERS; i++){
            bool cond = (regs[i]>max_value) && j*REGISTERS+i<H;
            max_idx = cond ? j*REGISTERS + i : max_idx;
            max_value = cond ? regs[i] : max_value;
        }
    }

    if(blockIdx.x*THREADS_X + threadIdx.x<N) max[blockIdx.x*THREADS_X + threadIdx.x] = max_idx;
}


template<const int REGISTERS, const int THREADS_X, const int H>
void find_max_call(float* __restrict__ scores, int* max, const int N, cudaStream_t stream = 0){
    const int BLOCKS_X = (N + THREADS_X - 1) / THREADS_X;

    find_max_2d_vec4<REGISTERS,THREADS_X,H><<<BLOCKS_X,THREADS_X,0,stream>>>(scores,max,N);
}


template<const int THREADS_X, const int CHANNELS, const int FILTERS>
__global__ __launch_bounds__(THREADS_X,1) void weight_trans_gpu(const float* __restrict__ input, float* __restrict__ output){ // in 3x3xCHANNELSxFILTERS  out 36xCHANNELSxFILTERS
    constexpr float SQRT2 = M_SQRT2;
    
    int in_idx = blockIdx.x*THREADS_X + threadIdx.x;

    float in_reg[9*4];
    float tile_reg[6*3*4];
    float out_reg[6*6*4];

    #pragma unroll
    for(int i=0; i<9; i++){
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(in_reg)[i] = reinterpret_cast<const float4*>(input)[in_idx + (i/3)*3*CHANNELS*FILTERS/4 + (i%3)*CHANNELS*FILTERS/4];
    }

    #pragma unroll
    for(int j=0; j<4; j++){
        for(int i=0; i<3; i++){
            tile_reg[i*6*4 + j]       = in_reg[i*4 + j];
            tile_reg[i*6*4 + 1*4 + j] = -(2.f/3)*in_reg[i*4 + j]  - (SQRT2/3)*in_reg[i*4 + j + 3*1*4] - (1.f/3.f)*in_reg[i*4 + j + 3*2*4];
            tile_reg[i*6*4 + 2*4 + j] = -(2.f/3)*in_reg[i*4 + j]  + (SQRT2/3)*in_reg[i*4 + j + 3*1*4] - (1.f/3.f)*in_reg[i*4 + j + 3*2*4];
            tile_reg[i*6*4 + 3*4 + j] = (1.f/6.f)*in_reg[i*4 + j] + (SQRT2/6)*in_reg[i*4 + j + 3*1*4] + (1.f/3.f)*in_reg[i*4 + j + 3*2*4];
            tile_reg[i*6*4 + 4*4 + j] = (1.f/6.f)*in_reg[i*4 + j] - (SQRT2/6)*in_reg[i*4 + j + 3*1*4] + (1.f/3.f)*in_reg[i*4 + j + 3*2*4];
            tile_reg[i*6*4 + 5*4 + j] = in_reg[i*4 + j + 3*2*4];
        }
    }

    #pragma unroll
    for(int j=0; j<4; j++){
        for(int i=0; i<6; i++){
            out_reg[i*6*4 + j]       = tile_reg[i*4 + j];
            out_reg[i*6*4 + 1*4 + j] = -(2.f/3)*tile_reg[i*4 + j]  - (SQRT2/3)*tile_reg[i*4 + j+6*4] - (1.f/3.f)*tile_reg[i*4 + j+12*4];
            out_reg[i*6*4 + 2*4 + j] = -(2.f/3)*tile_reg[i*4 + j]  + (SQRT2/3)*tile_reg[i*4 + j+6*4] - (1.f/3.f)*tile_reg[i*4 + j+12*4];
            out_reg[i*6*4 + 3*4 + j] = (1.f/6.f)*tile_reg[i*4 + j] + (SQRT2/6)*tile_reg[i*4 + j+6*4] + (1.f/3.f)*tile_reg[i*4 + j+12*4];
            out_reg[i*6*4 + 4*4 + j] = (1.f/6.f)*tile_reg[i*4 + j] - (SQRT2/6)*tile_reg[i*4 + j+6*4] + (1.f/3.f)*tile_reg[i*4 + j+12*4];
            out_reg[i*6*4 + 5*4 + j] = tile_reg[i*4 + j+12*4];
        }
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + i*CHANNELS*FILTERS/4*6]                        = reinterpret_cast<float4*>(out_reg)[i*6];
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + i*CHANNELS*FILTERS/4*6 + CHANNELS*FILTERS/4]   = reinterpret_cast<float4*>(out_reg)[i*6 + 1];
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + i*CHANNELS*FILTERS/4*6 + 2*CHANNELS*FILTERS/4] = reinterpret_cast<float4*>(out_reg)[i*6 + 2];
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + i*CHANNELS*FILTERS/4*6 + 3*CHANNELS*FILTERS/4] = reinterpret_cast<float4*>(out_reg)[i*6 + 3];
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + i*CHANNELS*FILTERS/4*6 + 4*CHANNELS*FILTERS/4] = reinterpret_cast<float4*>(out_reg)[i*6 + 4];
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + i*CHANNELS*FILTERS/4*6 + 5*CHANNELS*FILTERS/4] = reinterpret_cast<float4*>(out_reg)[i*6 + 5];
    }
}

template<const int CHANNELS, const int FILTERS>
void weight_trans(float* d_input,float* d_output, cudaStream_t stream = 0){
    const int THREADS_X = ( (CHANNELS*FILTERS<256*128) ? 64 : 256);
    const int BLOCKS_X  = (((FILTERS*CHANNELS)/4) + THREADS_X - 1) / THREADS_X;;

    dim3 threads(THREADS_X);
    dim3 blocks(BLOCKS_X);

    
    weight_trans_gpu<THREADS_X,CHANNELS,FILTERS><<<blocks, threads, 0, stream>>>(d_input,d_output);
    
}



template<const int THREADS_X, const int CHANNELS, const int FILTERS>
__global__ __launch_bounds__(THREADS_X,1) void weight_trans_back(const float* __restrict__ input, float* __restrict__ output){ // in 36xCHANNELSxFILTERS  out 3x3xCHANNELSxFILTERS
    constexpr float SQRT2 = M_SQRT2;

    int in_idx = blockIdx.x*THREADS_X + threadIdx.x;

    float in_reg[6*6*4];
    float temp_reg[6*3*4];
    float out_reg[3*3*4];

    #pragma unroll
    for(int i=0; i<36; i++){
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(in_reg)[i] = reinterpret_cast<const float4*>(input)[in_idx + (i/6)*6*CHANNELS*FILTERS/4 + (i%6)*CHANNELS*FILTERS/4];
    }

    #pragma unroll
    for(int j=0; j<4; j++){
        for(int i=0; i<6; i++){
            temp_reg[i*3*4 + 0*4 + j] = in_reg[i*6*4 + 0*4 + j] - (2.f/3.f)*in_reg[i*6*4 + 1*4 + j]   - (2.f/3.f)*in_reg[i*6*4 + 2*4 + j]    + (1.f/6.f)*in_reg[i*6*4 + 3*4 + j]   + (1.f/6.f)*in_reg[i*6*4 + 4*4 + j];
            temp_reg[i*3*4 + 1*4 + j] =                         - (SQRT2/3.f)*in_reg[i*6*4 + 1*4 + j] + (SQRT2/3.f)*in_reg[i*6*4 + 2*4 + j]  + (SQRT2/6.f)*in_reg[i*6*4 + 3*4 + j] - (SQRT2/6.f)*in_reg[i*6*4 + 4*4 + j];
            temp_reg[i*3*4 + 2*4 + j] =                         - (1.f/3.f)*in_reg[i*6*4 + 1*4 + j]   - (1.f/3.f)*in_reg[i*6*4 + 2*4 + j]    + (1.f/3.f)*in_reg[i*6*4 + 3*4 + j]   + (1.f/3.f)*in_reg[i*6*4 + 4*4 + j]   + in_reg[i*6*4 + 5*4 + j];
        }
    }

    #pragma unroll
    for(int j=0; j<4; j++){
        for(int i=0; i<3; i++){
            out_reg[i*3*4 + 0*4 + j] = temp_reg[i*4 + 0*3*4 + j] - (2.f/3.f)*temp_reg[i*4 + 1*3*4 + j]   - (2.f/3.f)*temp_reg[i*4 + 2*3*4 + j]   + (1.f/6.f)*temp_reg[i*4 + 3*3*4 + j]   + (1.f/6.f)*temp_reg[i*4 + 4*3*4 + j];
            out_reg[i*3*4 + 1*4 + j] =                           - (SQRT2/3.f)*temp_reg[i*4 + 1*3*4 + j] + (SQRT2/3.f)*temp_reg[i*4 + 2*3*4 + j] + (SQRT2/6.f)*temp_reg[i*4 + 3*3*4 + j] - (SQRT2/6.f)*temp_reg[i*4 + 4*3*4 + j];
            out_reg[i*3*4 + 2*4 + j] =                           - (1.f/3.f)*temp_reg[i*4 + 1*3*4 + j]   - (1.f/3.f)*temp_reg[i*4 + 2*3*4 + j]   + (1.f/3.f)*temp_reg[i*4 + 3*3*4 + j]   + (1.f/3.f)*temp_reg[i*4 + 4*3*4 + j]   + 1*temp_reg[i*4 + 5*3*4 + j];
        }
    }

    #pragma unroll
    for(int i=0; i<9; i++){
        if(in_idx<(CHANNELS*FILTERS)/4)reinterpret_cast<float4*>(output)[in_idx + (i%3)*CHANNELS*FILTERS/4*3 + (i/3)*CHANNELS*FILTERS/4] = reinterpret_cast<float4*>(out_reg)[(i/3)*3 + i%3];
    }
}


template<const int CHANNELS, const int FILTERS>
void weight_trans_back_call(float* d_input,float* d_output, cudaStream_t stream = 0){
    const int THREADS_X = ( (CHANNELS*FILTERS<256*128) ? 32 : 128);
    const int BLOCKS_X = (((FILTERS*CHANNELS)/4) + THREADS_X - 1) / THREADS_X;

    dim3 threads(THREADS_X);
    dim3 blocks(BLOCKS_X);

    
    weight_trans_back<THREADS_X,CHANNELS,FILTERS><<<blocks, threads, 0, stream>>>(d_input,d_output);
    
}


template<const int MAX_ARRAY_SIZE,const int CLASSES,const int THREADS_X,const int REGISTERS>
__global__ __launch_bounds__(THREADS_X,1) void find_accuracy_gpu(const int* __restrict__ max_array, const char* __restrict__ truth_table, const int DIVIDE_BY, float* accuracy){
    int max_array_reg[REGISTERS] = {0};

    float thread_acc = 0;
    for(int j=0; j<(MAX_ARRAY_SIZE + REGISTERS*THREADS_X - 1) / (REGISTERS*THREADS_X); j++){
        for(int i=0; i<REGISTERS/4; i++){
            int max_array_idx = threadIdx.x + i*THREADS_X + j*REGISTERS/4*THREADS_X;
            if(max_array_idx<MAX_ARRAY_SIZE/4) reinterpret_cast<int4*>(max_array_reg)[i] = reinterpret_cast<const int4*>(max_array)[max_array_idx];
        }

        for(int i=0; i<REGISTERS; i++){
            int truth_table_idx = (j*THREADS_X*REGISTERS + (i%4) + (i/4)*THREADS_X*4 + threadIdx.x*4)*CLASSES + max_array_reg[i];
            if(truth_table_idx<MAX_ARRAY_SIZE*CLASSES){
                if (truth_table[truth_table_idx] == 1) thread_acc++;
            }
        }
    }


    float warp_acc = warpReduceDown(thread_acc);
    
    
    if(threadIdx.x==0){
        accuracy[0] += warp_acc/DIVIDE_BY;
    }
}


template<const int FM_REGISTERS, const int FM_THREADS_X, const int CLASSES, const int FA_MAX_ARRAY_SIZE, const int FA_THREADS_X, const int FA_REGISTERS>
void find_accuracy_call(float* FM_scores, int* FM_max, const int FM_N, const char* truth_table, const int DIVIDE_BY, float* accuracy, cudaStream_t stream = 0){
    find_max_call<FM_REGISTERS,FM_THREADS_X,CLASSES>(FM_scores,FM_max,FM_N,stream);

    find_accuracy_gpu<FA_MAX_ARRAY_SIZE,CLASSES,FA_THREADS_X,FA_REGISTERS><<<1,FA_THREADS_X,0,stream>>>(FM_max,truth_table,DIVIDE_BY,accuracy);
}
