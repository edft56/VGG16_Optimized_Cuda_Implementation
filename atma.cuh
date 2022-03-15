#pragma once

#include <cmath>

//OUTPUT TRANSFORM FUNCTIONS

struct atma_forward_data{ //FILTERS,HW,BATCH_SIZE,POOL
    const int FILTERS,HW,BATCH_SIZE,POOL;
    const int TILES_X       = (HW + 3) / 4;  
    const int TILES_Y       = TILES_X;
    const int TILES         = TILES_X*TILES_X; 

    const int OUT_CUT       = (HW%4)/2;

    const int IN_SIZE       = 36*FILTERS*TILES*BATCH_SIZE;   
    const int PR_IDX_SIZE   = HW*HW*FILTERS*BATCH_SIZE;
    const int OUT_SIZE      = ( (POOL==1) ? (HW/2)*(HW/2) : HW*HW ) * FILTERS*BATCH_SIZE;

    const int FILTERS_THREADBLOCK = FILTERS; //threads

    const int BLOCKS_X      = TILES_X;
    const int BLOCKS_Y      = TILES_Y;
    const int BLOCKS_Z      = BATCH_SIZE;

    const int IN_STRIDE     = FILTERS*TILES*BATCH_SIZE;
};


struct atma_back_data{ //FILTERS,IN_DIM,BATCH_SIZE,POOL
    const int FILTERS,IN_DIM,BATCH_SIZE,POOL;
    const int TILES_1D      = (IN_DIM + 3) / 4;
    const int TILES         = TILES_1D*TILES_1D;

    const int OUT_CUT       = (IN_DIM%4)/2;

    const int IN_SIZE       = BATCH_SIZE * ( (POOL==1) ? (IN_DIM/2)*(IN_DIM/2) : IN_DIM*IN_DIM ) * FILTERS;
    const int PR_IDX_SIZE   = BATCH_SIZE*IN_DIM*IN_DIM*FILTERS;
    const int OUT_SIZE      = BATCH_SIZE*36*TILES*FILTERS;
    const int DBIAS_SIZE    = FILTERS;

    const int THREADS_X     = FILTERS;

    const int BLOCKS_X      = TILES_1D;
    const int BLOCKS_Y      = TILES_1D;
    const int BLOCKS_Z      = BATCH_SIZE;
};



__global__ __launch_bounds__(1024,1) void zero_out_dbias(float* __restrict__ dbias){ //FILTERS/4 threads
    float temp[4] = {0};
    reinterpret_cast<float4*>(dbias)[threadIdx.x] = reinterpret_cast<float4*>(temp)[0];
}




template<const int FILTERS, const int HW, const int IN_STRIDE, const int TILES_X, const int TILES, const int OUT_CUT>
__global__ void AtMA_relu(const float* __restrict__ input, float* __restrict__ output, float* bias, char* relu_idx){ //input 36x(BATCH_SIZE*TILES)xFILTERS  output NxHWxHWxFILTERS
    constexpr float SQRT2 = M_SQRT2;

    int in_read_idx = blockIdx.z*FILTERS*TILES + blockIdx.y*FILTERS*TILES_X + blockIdx.x*FILTERS + threadIdx.y*FILTERS + threadIdx.x;
    //int out_write_idx = blockIdx.z*FILTERS*HW*HW + blockIdx.y*FILTERS*HW*4 + blockIdx.x*FILTERS*4 + threadIdx.y*FILTERS*4 + threadIdx.x; //224x224x64

    float buf[36];
    float tile_reg[4*6];
    float out[4];

    int out_x_idx   = threadIdx.x;
    int out_y_tile  = threadIdx.y + blockIdx.x;
    int out_z_tile  = blockIdx.y;
    int out_w_idx   = blockIdx.z*FILTERS*HW*HW;

    #pragma unroll
    for(int i=0; i<6; i++){
        buf[i*6]        = input[in_read_idx + i*IN_STRIDE];
        buf[i*6 + 1]    = input[in_read_idx + i*IN_STRIDE + 1*IN_STRIDE*6];
        buf[i*6 + 2]    = input[in_read_idx + i*IN_STRIDE + 2*IN_STRIDE*6];
        buf[i*6 + 3]    = input[in_read_idx + i*IN_STRIDE + 3*IN_STRIDE*6];
        buf[i*6 + 4]    = input[in_read_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        buf[i*6 + 5]    = input[in_read_idx + i*IN_STRIDE + 5*IN_STRIDE*6];
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        tile_reg[0 + i]     = buf[i*6]                  + buf[i*6 + 1]              + buf[i*6 + 2]          + buf[i*6 + 3]          + buf[i*6 + 4];
        tile_reg[6 + i]     = (SQRT2/2.f)*buf[i*6 + 1]  - (SQRT2/2.f)*buf[i*6 + 2]  + SQRT2*buf[i*6 + 3]    - SQRT2*buf[i*6 + 4];
        tile_reg[12 + i]    = 0.5f*buf[i*6 + 1]         + 0.5f*buf[i*6 + 2]         + 2*buf[i*6 + 3]        + 2*buf[i*6 + 4];
        tile_reg[18 + i]    = (SQRT2/4.f)*buf[i*6 + 1]  - (SQRT2/4.f)*buf[i*6 + 2]  + 2*SQRT2*buf[i*6 + 3]  - 2*SQRT2*buf[i*6 + 4]  + buf[i*6 + 5];
    }


    int out_y_idx0 = out_y_tile*4 - OUT_CUT; 
    int out_y_idx1 = out_y_tile*4 + 1 - OUT_CUT;
    int out_y_idx2 = out_y_tile*4 + 2 - OUT_CUT;
    int out_y_idx3 = out_y_tile*4 + 3 - OUT_CUT;

    #pragma unroll
    for(int i=0; i<4; i++){
        int out_z_idx = out_z_tile*4 + i - OUT_CUT;

        out[0] = tile_reg[i*6 + 0]              + tile_reg[i*6 + 1]             + tile_reg[i*6 + 2]         + tile_reg[i*6 + 3]         + tile_reg[i*6 + 4];
        out[1] = (SQRT2/2.f)*tile_reg[i*6 + 1]  - (SQRT2/2.f)*tile_reg[i*6 + 2] + SQRT2*tile_reg[i*6 + 3]   - SQRT2*tile_reg[i*6 + 4];
        out[2] = 0.5f*tile_reg[i*6 + 1]         + 0.5f*tile_reg[i*6 + 2]        + 2*tile_reg[i*6 + 3]       + 2*tile_reg[i*6 + 4];
        out[3] = (SQRT2/4.f)*tile_reg[i*6 + 1]  - (SQRT2/4.f)*tile_reg[i*6 + 2] + 2*SQRT2*tile_reg[i*6 + 3] - 2*SQRT2*tile_reg[i*6 + 4] + tile_reg[i*6 + 5];

        
        out[0] = (out[0] + bias[threadIdx.x] >= 0) ? out[0] + bias[threadIdx.x] : 0;
        out[1] = (out[1] + bias[threadIdx.x] >= 0) ? out[1] + bias[threadIdx.x] : 0;
        out[2] = (out[2] + bias[threadIdx.x] >= 0) ? out[2] + bias[threadIdx.x] : 0;
        out[3] = (out[3] + bias[threadIdx.x] >= 0) ? out[3] + bias[threadIdx.x] : 0;

        if( !(out_z_idx<0 || out_z_idx>HW-1) ){
            if( !(out_y_idx0<0 || out_y_idx0>HW-1) ) output     [out_w_idx + out_z_idx*FILTERS*HW + out_y_idx0*FILTERS + out_x_idx]    = out[0];
            if( !(out_y_idx0<0 || out_y_idx0>HW-1) ) relu_idx   [out_w_idx + out_z_idx*FILTERS*HW + out_y_idx0*FILTERS + out_x_idx]    = (out[0]>0) ? 1 : 0;

            output  [out_w_idx + out_z_idx*FILTERS*HW + out_y_idx1*FILTERS + out_x_idx] = out[1];
            relu_idx[out_w_idx + out_z_idx*FILTERS*HW + out_y_idx1*FILTERS + out_x_idx] = (out[1]>0) ? 1 : 0;

            output  [out_w_idx + out_z_idx*FILTERS*HW + out_y_idx2*FILTERS + out_x_idx] = out[2];
            relu_idx[out_w_idx + out_z_idx*FILTERS*HW + out_y_idx2*FILTERS + out_x_idx] = (out[2]>0) ? 1 : 0;

            if( !(out_y_idx3<0 || out_y_idx3>HW-1) ) output  [out_w_idx + out_z_idx*FILTERS*HW + out_y_idx3*FILTERS + out_x_idx] = out[3];
            if( !(out_y_idx3<0 || out_y_idx3>HW-1) ) relu_idx[out_w_idx + out_z_idx*FILTERS*HW + out_y_idx3*FILTERS + out_x_idx] = (out[3]>0) ? 1 : 0;
        }
    }
}


template<const int FILTERS, const int HW, const int IN_STRIDE, const int TILES_X, const int TILES>
__global__ void AtMA_pool_relu_no_pad(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ bias, char* __restrict__ pool_relu_idx){ //input 36x(BATCH_SIZE*TILES)xFILTERS  output Nx(HW/2)x(HW/2)xFILTERS
    constexpr float SQRT2 = M_SQRT2;

    int in_read_idx     = blockIdx.z*FILTERS*TILES          + blockIdx.y*FILTERS*TILES_X    + blockIdx.x*FILTERS    + threadIdx.x;
    int out_write_idx   = blockIdx.z*FILTERS*(HW/2)*(HW/2)  + blockIdx.y*FILTERS*(HW/2)*2   + blockIdx.x*FILTERS*2  + threadIdx.x;
    int bias_idx        = threadIdx.x;
    int pr_idx          = blockIdx.z*FILTERS*(HW)*(HW)      + blockIdx.y*FILTERS*(HW)*4     + blockIdx.x*FILTERS*4  + threadIdx.x;

    float tile_reg[4*6];

    #pragma unroll
    for(int i=0; i<6; i++){
        tile_reg[ 0 + i]    = input[in_read_idx + i*IN_STRIDE]                              + input[in_read_idx + i*IN_STRIDE + 1*IN_STRIDE*6]              + input[in_read_idx + i*IN_STRIDE + 2*IN_STRIDE*6]          + input[in_read_idx + i*IN_STRIDE + 3*IN_STRIDE*6]          + input[in_read_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        tile_reg[ 6 + i]    = (SQRT2/2.f)*input[in_read_idx + i*IN_STRIDE + 1*IN_STRIDE*6]  - (SQRT2/2.f)*input[in_read_idx + i*IN_STRIDE + 2*IN_STRIDE*6]  + SQRT2*input[in_read_idx + i*IN_STRIDE + 3*IN_STRIDE*6]    - SQRT2*input[in_read_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        tile_reg[12 + i]    = 0.5f*input[in_read_idx + i*IN_STRIDE + 1*IN_STRIDE*6]         + 0.5f*input[in_read_idx + i*IN_STRIDE + 2*IN_STRIDE*6]         + 2*input[in_read_idx + i*IN_STRIDE + 3*IN_STRIDE*6]        + 2*input[in_read_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        tile_reg[18 + i]    = (SQRT2/4.f)*input[in_read_idx + i*IN_STRIDE + 1*IN_STRIDE*6]  - (SQRT2/4.f)*input[in_read_idx + i*IN_STRIDE + 2*IN_STRIDE*6]  + 2*SQRT2*input[in_read_idx + i*IN_STRIDE + 3*IN_STRIDE*6]  - 2*SQRT2*input[in_read_idx + i*IN_STRIDE + 4*IN_STRIDE*6]  + input[in_read_idx + i*IN_STRIDE + 5*IN_STRIDE*6];
    }


    // Produce 2x4 tile and do max pooling and relu on 2x2 tiles.
    #pragma unroll
    for(int i=0; i<4; i+=2){
        
        float x0[4];

        x0[0] = tile_reg[i*6 + 0]               + tile_reg[i*6 + 1]             + tile_reg[i*6 + 2]         + tile_reg[i*6 + 3]         + tile_reg[i*6 + 4];
        x0[1] = (SQRT2/2.f)*tile_reg[i*6 + 1]   - (SQRT2/2.f)*tile_reg[i*6 + 2] + SQRT2*tile_reg[i*6 + 3]   - SQRT2*tile_reg[i*6 + 4];
        x0[2] = 0.5f*tile_reg[i*6 + 1]          + 0.5f*tile_reg[i*6 + 2]        + 2*tile_reg[i*6 + 3]       + 2*tile_reg[i*6 + 4];
        x0[3] = (SQRT2/4.f)*tile_reg[i*6 + 1]   - (SQRT2/4.f)*tile_reg[i*6 + 2] + 2*SQRT2*tile_reg[i*6 + 3] - 2*SQRT2*tile_reg[i*6 + 4] + tile_reg[i*6 + 5];


        float x1[4];

        x1[0] = tile_reg[(i+1)*6 + 0]               + tile_reg[(i+1)*6 + 1]             + tile_reg[(i+1)*6 + 2]         + tile_reg[(i+1)*6 + 3]         + tile_reg[(i+1)*6 + 4];
        x1[1] = (SQRT2/2.f)*tile_reg[(i+1)*6 + 1]   - (SQRT2/2.f)*tile_reg[(i+1)*6 + 2] + SQRT2*tile_reg[(i+1)*6 + 3]   - SQRT2*tile_reg[(i+1)*6 + 4];
        x1[2] = 0.5f*tile_reg[(i+1)*6 + 1]          + 0.5f*tile_reg[(i+1)*6 + 2]        + 2*tile_reg[(i+1)*6 + 3]       + 2*tile_reg[(i+1)*6 + 4];
        x1[3] = (SQRT2/4.f)*tile_reg[(i+1)*6 + 1]   - (SQRT2/4.f)*tile_reg[(i+1)*6 + 2] + 2*SQRT2*tile_reg[(i+1)*6 + 3] - 2*SQRT2*tile_reg[(i+1)*6 + 4] + tile_reg[(i+1)*6 + 5];

        float bias_reg = bias[bias_idx];
        

        float max   = (x0[0]>x0[1]) ? x0[0] : x0[1];
        max         = (x1[0]>max)   ? x1[0] : max;
        max         = (x1[1]>max)   ? x1[1] : max;

        bool pred   = max + bias_reg >= 0;
        max         = (pred) ? max + bias_reg : 0;

        output[out_write_idx + (i/2)*FILTERS*(HW/2)] = max;

        bool done;

        pool_relu_idx[pr_idx + 0 + i*HW*FILTERS]                       = ((pred) && max == x0[0] + bias_reg) ? 1 : 0;
        done = ( (pred) && max == x0[0] + bias_reg ) ? true : false;

        pool_relu_idx[pr_idx + FILTERS + i*HW*FILTERS]                 = ((pred) && max == x0[1] + bias_reg && !done) ? 1 : 0; 
        done = ( (pred) && max == x0[1] + bias_reg && !done ) ? !done : done;

        pool_relu_idx[pr_idx + FILTERS*HW + i*HW*FILTERS]              = ((pred) && max == x1[0] + bias_reg && !done) ? 1 : 0;
        done = ( (pred) && max == x1[0] + bias_reg && !done ) ? !done : done;

        pool_relu_idx[pr_idx + FILTERS*HW + FILTERS + i*HW*FILTERS]    = ((pred) && max == x1[1] + bias_reg && !done) ? 1 : 0;


        max = (x0[2]>x0[3]) ? x0[2] : x0[3];
        max = (x1[2]>max)   ? x1[2] : max;
        max = (x1[3]>max)   ? x1[3] : max;

        pred    = max + bias_reg > 0;
        max     = (pred) ? max + bias_reg : 0;

        
        output[out_write_idx + (i/2)*FILTERS*(HW/2) + FILTERS] = max;

        pool_relu_idx[pr_idx + 2*FILTERS + i*HW*FILTERS]                 = ((pred) && max == x0[2] + bias_reg) ? 1 : 0;
        done = ( (pred) && max == x0[2] + bias_reg ) ? true : false;

        pool_relu_idx[pr_idx + 3*FILTERS + i*HW*FILTERS]                 = ((pred) && max == x0[3] + bias_reg && !done) ? 1 : 0;
        done = ( (pred) && max == x0[3] + bias_reg && !done ) ? !done : done;

        pool_relu_idx[pr_idx + FILTERS*HW + 2*FILTERS + i*HW*FILTERS]    = ((pred) && max == x1[2] + bias_reg && !done) ? 1 : 0;
        done = ( (pred) && max == x1[2] + bias_reg && !done ) ? !done : done;

        pool_relu_idx[pr_idx + FILTERS*HW + 3*FILTERS + i*HW*FILTERS]    = ((pred) && max == x1[3] + bias_reg && !done) ? 1 : 0;

    }
}


template<const int FILTERS, const int HW, const int IN_STRIDE, const int TILES_X, const int TILES, const int BLOCKS_X, const int BLOCKS_Y>
__global__ void AtMA_pool_relu_pad(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ bias, char* __restrict__ pool_relu_idx){ //input 36x(BATCH_SIZE*TILES)xFILTERS  output Nx(HW/2)x(HW/2)xFILTERS
    constexpr float SQRT2 = M_SQRT2;
    
    int base_in_idx = blockIdx.z*FILTERS*TILES          + blockIdx.y*FILTERS*TILES_X    + blockIdx.x*FILTERS    + threadIdx.x;
    int out_idx     = blockIdx.z*FILTERS*(HW/2)*(HW/2)  + blockIdx.y*FILTERS*(HW/2)*2   + blockIdx.x*FILTERS*2  + threadIdx.x;
    int bias_idx    = threadIdx.x;
    int pr_idx      = blockIdx.z*FILTERS*(HW)*(HW)      + blockIdx.y*FILTERS*(HW)*4     + blockIdx.x*FILTERS*4  + threadIdx.x;

    float tile_reg[3*5];
    float tile_reg_right[3*5];
    float tile_reg_down[5];
    float tile_reg_diag[5];


    #pragma unroll
    for(int i=1; i<6; i++){
        tile_reg[ 0 + (i-1)] = (SQRT2/2.0f)*input[base_in_idx + i*IN_STRIDE + 1*IN_STRIDE*6] - (SQRT2/2.0f)*input[base_in_idx + i*IN_STRIDE + 2*IN_STRIDE*6] + SQRT2*input[base_in_idx + i*IN_STRIDE + 3*IN_STRIDE*6]   - SQRT2*input[base_in_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        tile_reg[ 5 + (i-1)] = 0.5f*input[base_in_idx + i*IN_STRIDE + 1*IN_STRIDE*6]         + 0.5f*input[base_in_idx + i*IN_STRIDE + 2*IN_STRIDE*6]         + 2*input[base_in_idx + i*IN_STRIDE + 3*IN_STRIDE*6]       + 2*input[base_in_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        tile_reg[10 + (i-1)] = (SQRT2/4.0f)*input[base_in_idx + i*IN_STRIDE + 1*IN_STRIDE*6] - (SQRT2/4.0f)*input[base_in_idx + i*IN_STRIDE + 2*IN_STRIDE*6] + 2*SQRT2*input[base_in_idx + i*IN_STRIDE + 3*IN_STRIDE*6] - 2*SQRT2*input[base_in_idx + i*IN_STRIDE + 4*IN_STRIDE*6] + input[base_in_idx + i*IN_STRIDE + 5*IN_STRIDE*6];
    }

    if(blockIdx.x!=BLOCKS_X-1){
        int in_idx = base_in_idx + FILTERS;
        #pragma unroll
        for(int i=0; i<5; i++){
            tile_reg_right[ 0 + (i)] = (SQRT2/2.0f)*input[in_idx + i*IN_STRIDE + 1*IN_STRIDE*6] - (SQRT2/2.0f)*input[in_idx + i*IN_STRIDE + 2*IN_STRIDE*6] + SQRT2*input[in_idx + i*IN_STRIDE + 3*IN_STRIDE*6]   - SQRT2*input[in_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
            tile_reg_right[ 5 + (i)] = 0.5f*input[in_idx + i*IN_STRIDE + 1*IN_STRIDE*6]         + 0.5f*input[in_idx + i*IN_STRIDE + 2*IN_STRIDE*6]         + 2*input[in_idx + i*IN_STRIDE + 3*IN_STRIDE*6]       + 2*input[in_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
            tile_reg_right[10 + (i)] = (SQRT2/4.0f)*input[in_idx + i*IN_STRIDE + 1*IN_STRIDE*6] - (SQRT2/4.0f)*input[in_idx + i*IN_STRIDE + 2*IN_STRIDE*6] + 2*SQRT2*input[in_idx + i*IN_STRIDE + 3*IN_STRIDE*6] - 2*SQRT2*input[in_idx + i*IN_STRIDE + 4*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 5*IN_STRIDE*6];
        }
    }

    if(blockIdx.y!=BLOCKS_Y-1){
        int in_idx = base_in_idx + TILES_X*FILTERS;
        #pragma unroll
        for(int i=1; i<6; i++){
            tile_reg_down[0 + (i-1)] = input[in_idx + i*IN_STRIDE] + input[in_idx + i*IN_STRIDE + 1*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 2*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 3*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        }
    }

    if(blockIdx.y!=BLOCKS_Y-1 && blockIdx.x!=BLOCKS_X-1){
        int in_idx = base_in_idx + FILTERS + TILES_X*FILTERS;
        #pragma unroll
        for(int i=0; i<5; i++){
            tile_reg_diag[0 + (i)] = input[in_idx + i*IN_STRIDE] + input[in_idx + i*IN_STRIDE + 1*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 2*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 3*IN_STRIDE*6] + input[in_idx + i*IN_STRIDE + 4*IN_STRIDE*6];
        }
    }

    float bias_reg = bias[bias_idx];


    float out11 = (SQRT2/2.0f)*tile_reg[0*5 + 0] - (SQRT2/2.0f)*tile_reg[0*5 + 1] + SQRT2*tile_reg[0*5 + 2] - SQRT2*tile_reg[0*5 + 3]; 
    float out12 = 0.5f*tile_reg[0*5 + 0]         + 0.5f*tile_reg[0*5 + 1]         + 2*tile_reg[0*5 + 2]     + 2*tile_reg[0*5 + 3]; 
    float out21 = (SQRT2/2.0f)*tile_reg[1*5 + 0] - (SQRT2/2.0f)*tile_reg[1*5 + 1] + SQRT2*tile_reg[1*5 + 2] - SQRT2*tile_reg[1*5 + 3]; 
    float out22 = 0.5f*tile_reg[1*5 + 0]         + 0.5f*tile_reg[1*5 + 1]         + 2*tile_reg[1*5 + 2]     + 2*tile_reg[1*5 + 3]; 

    
    float max = (out11>out12) ? out11 : out12;
    max       = (out21>max) ? out21 : max;
    max       = (out22>max) ? out22 : max;

    max             = max + bias_reg;
    bool pred       = max > 0;
    output[out_idx] = (pred) ? max : 0;  

    bool done;
    pool_relu_idx[pr_idx + 0]                       = (pred && max==out11 + bias_reg) ? 1 : 0;
    done                                            = pred && max==out11 + bias_reg ? true : false;

    pool_relu_idx[pr_idx + FILTERS]                 = (pred && max==out12 + bias_reg && !done) ? 1 : 0;
    done                                            = pred && max==out12 + bias_reg && !done ? !done : done;

    pool_relu_idx[pr_idx + FILTERS*HW]              = (pred && max==out21 + bias_reg && !done) ? 1 : 0;
    done                                            = pred && max==out21 + bias_reg && !done ? !done : done;

    pool_relu_idx[pr_idx + FILTERS*HW + FILTERS]    = (pred && max==out22 + bias_reg && !done) ? 1 : 0;
    

    if(blockIdx.x!=BLOCKS_X-1){
        float out13  = (SQRT2/4.0f)*tile_reg[0*5 + 0] - (SQRT2/4.0f)*tile_reg[0*5 + 1] + 2*SQRT2*tile_reg[0*5 + 2] - 2*SQRT2*tile_reg[0*5 + 3] + tile_reg[0*5 + 4]; 
        float out23  = (SQRT2/4.0f)*tile_reg[1*5 + 0] - (SQRT2/4.0f)*tile_reg[1*5 + 1] + 2*SQRT2*tile_reg[1*5 + 2] - 2*SQRT2*tile_reg[1*5 + 3] + tile_reg[1*5 + 4]; 
        float right1 = tile_reg_right[0*5 + 0]        + tile_reg_right[0*5 + 1]        + tile_reg_right[0*5 + 2]   + tile_reg_right[0*5 + 3]   + tile_reg_right[0*5 + 4];
        float right2 = tile_reg_right[1*5 + 0]        + tile_reg_right[1*5 + 1]        + tile_reg_right[1*5 + 2]   + tile_reg_right[1*5 + 3]   + tile_reg_right[1*5 + 4];

        float max = (out13>right1) ? out13  : right1;
        max       = (out23>max)    ? out23  : max;
        max       = (right2>max)   ? right2 : max;

        max                       = max + bias_reg;
        bool pred                 = max > 0;
        output[out_idx + FILTERS] = (pred) ? max : 0;  
        
        pool_relu_idx[pr_idx + 2*FILTERS]               = (pred && max==out13 + bias_reg) ? 1 : 0;
        done                                            = pred && max==out13 + bias_reg ? true : false;

        pool_relu_idx[pr_idx + 3*FILTERS]               = (pred && max==right1 + bias_reg && !done) ? 1 : 0;
        done                                            = pred && max==right1 + bias_reg && !done ? !done : done;

        pool_relu_idx[pr_idx + 2*FILTERS + HW*FILTERS]  = (pred && max==out23 + bias_reg && !done) ? 1 : 0;
        done                                            = pred && max==out23 + bias_reg && !done ? !done : done;

        pool_relu_idx[pr_idx + 3*FILTERS + FILTERS*HW]  = (pred && max==right2 + bias_reg && !done) ? 1 : 0;
    }

    if(blockIdx.y!=BLOCKS_Y-1){
        float out31 = (SQRT2/2.0f)*tile_reg[2*5 + 0] - (SQRT2/2.0f)*tile_reg[2*5 + 1] + SQRT2*tile_reg[2*5 + 2] - SQRT2*tile_reg[2*5 + 3]; 
        float out32 = 0.5f*tile_reg[2*5 + 0]         + 0.5f*tile_reg[2*5 + 1]         + 2*tile_reg[2*5 + 2]     + 2*tile_reg[2*5 + 3]; 
        float down1 = (SQRT2/2.0f)*tile_reg_down[0]  - (SQRT2/2.0f)*tile_reg_down[1]  + SQRT2*tile_reg_down[2]  - SQRT2*tile_reg_down[3];
        float down2 = 0.5f*tile_reg_down[0]          + 0.5f*tile_reg_down[1]          + 2*tile_reg_down[2]      + 2*tile_reg_down[3];
        
        float max = (out31>out32) ? out31 : out32;
        max       = (down1>max)   ? down1 : max;
        max       = (down2>max)   ? down2 : max;

        max                              = max + bias_reg;
        bool pred                        = max > 0;
        output[out_idx + (HW/2)*FILTERS] = (pred) ? max : 0;  
        
        pool_relu_idx[pr_idx + 2*FILTERS*HW]            = (pred && max==out31 + bias_reg) ?    1 : 0;
        done                                            = pred && max==out31 + bias_reg   ? true : false;

        pool_relu_idx[pr_idx + FILTERS + 2*HW*FILTERS]  = (pred && max==out32 + bias_reg && !done) ?     1 : 0;
        done                                            = pred && max==out32 + bias_reg && !done   ? !done : done;

        pool_relu_idx[pr_idx + 3*FILTERS*HW]            = (pred && max==down1 + bias_reg && !done) ?     1 : 0;
        done                                            = pred && max==down1 + bias_reg && !done   ? !done : done;

        pool_relu_idx[pr_idx + 3*FILTERS*HW + FILTERS]  = (pred && max==down2 + bias_reg && !done) ?     1 : 0;
    }

    if(blockIdx.y!=BLOCKS_Y-1 && blockIdx.x!=BLOCKS_X-1){
        float out33  = (SQRT2/4.0f)*tile_reg[2*5 + 0] - (SQRT2/4.0f)*tile_reg[2*5 + 1] + 2*SQRT2*tile_reg[2*5 + 2] - 2*SQRT2*tile_reg[2*5 + 3] + tile_reg[2*5 + 4];
        float right3 = tile_reg_right[2*5 + 0]        + tile_reg_right[2*5 + 1]        + tile_reg_right[2*5 + 2]   + tile_reg_right[2*5 + 3]   + tile_reg_right[2*5 + 4];
        float down3  = (SQRT2/4.0f)*tile_reg_down[0]  - (SQRT2/4.0f)*tile_reg_down[1]  + 2*SQRT2*tile_reg_down[2]  - 2*SQRT2*tile_reg_down[3]  + tile_reg_down[4];
        float diag0  = tile_reg_diag[0]               + tile_reg_diag[1]               + tile_reg_diag[2]          + tile_reg_diag[3]          + tile_reg_diag[4];

        float max = (out33>right3) ? out33 : right3;
        max       = (down3>max)    ? down3 : max;
        max       = (diag0>max)    ? diag0 : max;

        max                                         = max + bias_reg;
        bool pred                                   = max > 0;
        output[out_idx + (HW/2)*FILTERS + FILTERS]  = (pred) ? max : 0; 
        
        pool_relu_idx[pr_idx + 2*FILTERS*HW + 2*FILTERS]  = (pred && max==out33 + bias_reg) ?    1 : 0;
        done                                              = pred && max==out33 + bias_reg   ? true : false;

        pool_relu_idx[pr_idx + 3*FILTERS + 2*HW*FILTERS]  = (pred && max==right3 + bias_reg && !done) ?     1 : 0;
        done                                              = pred && max==right3 + bias_reg && !done   ? !done : done;

        pool_relu_idx[pr_idx + 3*FILTERS*HW + 2*FILTERS]  = (pred && max==down3 + bias_reg && !done) ?     1 : 0;
        done                                              = pred && max==down3 + bias_reg && !done   ? !done : done;

        pool_relu_idx[pr_idx + 3*FILTERS*HW + 3*FILTERS]  = (pred && max==diag0 + bias_reg && !done) ? 1 : 0;
    }
}


template<const int FILTERS, const int HW, const int IN_STRIDE, const int TILES_X, const int TILES, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int OUT_CUT, const int FILTERS_THREADBLOCK, const int POOL>
void atma_forward(float* input, float* output, float* bias, char* pool_relu_idx, cudaStream_t stream = 0){
    dim3 threads(FILTERS_THREADBLOCK);
    dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

    if(POOL==0){
        AtMA_relu<FILTERS, HW, IN_STRIDE, TILES_X, TILES, OUT_CUT><<<blocks, threads, 0, stream>>>(input,output,bias,pool_relu_idx); // actually just relu_idx
    }
    else{
        if(OUT_CUT==0){
            AtMA_pool_relu_no_pad<FILTERS, HW, IN_STRIDE, TILES_X, TILES><<<blocks, threads, 0, stream>>>(input,output,bias,pool_relu_idx);
        }
        else{
            AtMA_pool_relu_pad<FILTERS, HW, IN_STRIDE, TILES_X, TILES, BLOCKS_X, BLOCKS_Y><<<blocks, threads, 0, stream>>>(input,output,bias,pool_relu_idx);
        }
    }
}





// does both
template<const int FILTERS, const int IN_DIM, const int BATCH_SIZE, const int TILES_1D, const int TILES, const int OUT_CUT>
__global__ __launch_bounds__(FILTERS,1) void atma_back_pad_batch_relu_gpu(const float* __restrict__ data, float* output, const char* __restrict__ pool_relu_idx, float* __restrict__ dbias){ //input NxIN_DIMxIN_DIMxFILTERS  output 36x(BATCH_SIZE*TILES)xFILTERS
    constexpr float SQRT2 = M_SQRT2;
    
    int in_idx = blockIdx.z*IN_DIM*IN_DIM*FILTERS + blockIdx.y*4*FILTERS*IN_DIM - OUT_CUT*FILTERS*IN_DIM + blockIdx.x*FILTERS*4 - OUT_CUT*FILTERS + threadIdx.x;
    int out_idx = blockIdx.z*TILES*FILTERS + blockIdx.y*FILTERS*TILES_1D + blockIdx.x*FILTERS + threadIdx.x;

    float buf[16]={0};

    int in_x_idx = blockIdx.x*4 - OUT_CUT;
    int in_y_idx = blockIdx.y*4 - OUT_CUT;

    #pragma unroll
    for(int x=0; x<4; x++){
        for(int y=0; y<4; y++){
           if( !(in_x_idx+x<0 || in_x_idx+x>IN_DIM-1 || in_y_idx+y<0 || in_y_idx+y>IN_DIM-1) ) buf[x*4+y] = data[in_idx + y*FILTERS*IN_DIM + x*FILTERS];
        }
    }

    
    float bias_add=0;

    float map[4*4] = {0};


    #pragma unroll
    for(int x=0; x<4; x++){
        for(int y=0; y<4; y++){
            if( !(in_x_idx+x<0 || in_x_idx+x>IN_DIM-1 || in_y_idx+y<0 || in_y_idx+y>IN_DIM-1) ) map[x*4 + y] = pool_relu_idx[in_idx + x*FILTERS + y*FILTERS*IN_DIM];
        }
    }

    
    float tile_reg[4*6];
    #pragma unroll
    for(int i=0; i<4; i++){
        tile_reg[i*6 + 0] = map[i*4 + 0]*buf[i*4 + 0];
        tile_reg[i*6 + 1] = map[i*4 + 0]*buf[i*4 + 0] + (SQRT2/2.f)*map[i*4 + 1]*buf[i*4 + 1] + 0.5f*map[i*4 + 2]*buf[i*4 + 2]  + (SQRT2/4.f)*map[i*4 + 3]*buf[i*4 + 3];
        tile_reg[i*6 + 2] = map[i*4 + 0]*buf[i*4 + 0] - (SQRT2/2.f)*map[i*4 + 1]*buf[i*4 + 1] + 0.5f*map[i*4 + 2]*buf[i*4 + 2]  - (SQRT2/4.f)*map[i*4 + 3]*buf[i*4 + 3];
        tile_reg[i*6 + 3] = map[i*4 + 0]*buf[i*4 + 0] + SQRT2*map[i*4 + 1]*buf[i*4 + 1]       + 2*map[i*4 + 2]*buf[i*4 + 2]     + 2*SQRT2*map[i*4 + 3]*buf[i*4 + 3];
        tile_reg[i*6 + 4] = map[i*4 + 0]*buf[i*4 + 0] - SQRT2*map[i*4 + 1]*buf[i*4 + 1]       + 2*map[i*4 + 2]*buf[i*4 + 2]     - 2*SQRT2*map[i*4 + 3]*buf[i*4 + 3];
        tile_reg[i*6 + 5] = map[i*4 + 3]*buf[i*4 + 3];

        bias_add += map[i*4 + 0]*buf[i*4 + 0] + map[i*4 + 1]*buf[i*4 + 1] +  map[i*4 + 2]*buf[i*4 + 2] + map[i*4 + 3]*buf[i*4 + 3];
    }

    atomicAdd(&dbias[threadIdx.x],bias_add);

    float buf_out[6*6];

    #pragma unroll
    for(int i=0; i<6; i++){
        buf_out[i*6]     = tile_reg[i];
        buf_out[i*6 + 1] = tile_reg[i] + (SQRT2/2.f)*tile_reg[i + 6] + 0.5f*tile_reg[i + 12] + (SQRT2/4.f)*tile_reg[i + 18];
        buf_out[i*6 + 2] = tile_reg[i] - (SQRT2/2.f)*tile_reg[i + 6] + 0.5f*tile_reg[i + 12] - (SQRT2/4.f)*tile_reg[i + 18];
        buf_out[i*6 + 3] = tile_reg[i] + SQRT2*tile_reg[i + 6]       + 2*tile_reg[i + 12]    + 2*SQRT2*tile_reg[i + 18];
        buf_out[i*6 + 4] = tile_reg[i] - SQRT2*tile_reg[i + 6]       + 2*tile_reg[i + 12]    - 2*SQRT2*tile_reg[i + 18];
        buf_out[i*6 + 5] = tile_reg[i + 18];
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE]                              = buf_out[i*6];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 1*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 1];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 2*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 2];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 3*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 3];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 4*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 4];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 5*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 5];
    }
}


//ONLY NON-PADDED
template<const int FILTERS, const int IN_DIM, const int BATCH_SIZE, const int TILES_1D, const int TILES>
__global__ __launch_bounds__(FILTERS,1) void atma_back_batch_pool_relu_gpu(const float* __restrict__ data, float* output,const char* __restrict__ pool_relu_idx, float* __restrict__ dbias){ //input NxIN_DIMxIN_DIMxFILTERS  output 36x(BATCH_SIZE*TILES)xFILTERS
    constexpr float SQRT2 = M_SQRT2;
    
    int in_idx = blockIdx.z*IN_DIM*IN_DIM*FILTERS + blockIdx.y*4*FILTERS*IN_DIM + blockIdx.x*FILTERS*4 + threadIdx.x;
    int out_idx = blockIdx.z*TILES*FILTERS + blockIdx.y*FILTERS*TILES_1D + blockIdx.x*FILTERS + threadIdx.x;
    int data_idx = blockIdx.z*(IN_DIM/2)*(IN_DIM/2)*FILTERS + blockIdx.y*FILTERS*(IN_DIM/2)*2 + blockIdx.x*FILTERS*2 + threadIdx.x;

    float tile_reg[4*6];
    

    float buf[4];
    buf[0] = data[data_idx];
    buf[1] = data[data_idx + FILTERS];
    buf[2] = data[data_idx + FILTERS*(IN_DIM/2)];
    buf[3] = data[data_idx + FILTERS*(IN_DIM/2) + FILTERS];

    
    float bias_add=0;

    float map[4*4];


    #pragma unroll
    for(int i=0; i<4; i++){
        map[i*4]     = pool_relu_idx[in_idx + i*FILTERS];
        map[i*4 + 1] = pool_relu_idx[in_idx + i*FILTERS + FILTERS*IN_DIM];
        map[i*4 + 2] = pool_relu_idx[in_idx + i*FILTERS + 2*FILTERS*IN_DIM];
        map[i*4 + 3] = pool_relu_idx[in_idx + i*FILTERS + 3*FILTERS*IN_DIM];
    }

    

    #pragma unroll
    for(int i=0; i<2; i++){
        tile_reg[i*6 + 0] = map[i*4 + 0]*buf[0];
        tile_reg[i*6 + 1] = map[i*4 + 0]*buf[0] + (SQRT2/2.f)*map[i*4 + 1]*buf[0] + 0.5f*map[i*4 + 2]*buf[2] + (SQRT2/4.f)*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 2] = map[i*4 + 0]*buf[0] - (SQRT2/2.f)*map[i*4 + 1]*buf[0] + 0.5f*map[i*4 + 2]*buf[2] - (SQRT2/4.f)*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 3] = map[i*4 + 0]*buf[0] + SQRT2*map[i*4 + 1]*buf[0]       + 2*map[i*4 + 2]*buf[2]    + 2*SQRT2*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 4] = map[i*4 + 0]*buf[0] - SQRT2*map[i*4 + 1]*buf[0]       + 2*map[i*4 + 2]*buf[2]    - 2*SQRT2*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 5] = map[i*4 + 3]*buf[2];

        bias_add += map[i*4 + 0]*buf[0] + map[i*4 + 1]*buf[0] +  map[i*4 + 2]*buf[2] + map[i*4 + 3]*buf[2];
    }

    #pragma unroll
    for(int i=2; i<4; i++){
        tile_reg[i*6 + 0] = map[i*4 + 0]*buf[1];
        tile_reg[i*6 + 1] = map[i*4 + 0]*buf[1] + (SQRT2/2.f)*map[i*4 + 1]*buf[1] + 0.5f*map[i*4 + 2]*buf[3]    + (SQRT2/4.f)*map[i*4 + 3]*buf[3];
        tile_reg[i*6 + 2] = map[i*4 + 0]*buf[1] - (SQRT2/2.f)*map[i*4 + 1]*buf[1] + 0.5f*map[i*4 + 2]*buf[3]    - (SQRT2/4.f)*map[i*4 + 3]*buf[3];
        tile_reg[i*6 + 3] = map[i*4 + 0]*buf[1] + SQRT2*map[i*4 + 1]*buf[1]       + 2*map[i*4 + 2]*buf[3]       + 2*SQRT2*map[i*4 + 3]*buf[3];
        tile_reg[i*6 + 4] = map[i*4 + 0]*buf[1] - SQRT2*map[i*4 + 1]*buf[1]       + 2*map[i*4 + 2]*buf[3]       - 2*SQRT2*map[i*4 + 3]*buf[3];
        tile_reg[i*6 + 5] = map[i*4 + 3]*buf[3];

        bias_add += map[i*4 + 0]*buf[1] + map[i*4 + 1]*buf[1] +  map[i*4 + 2]*buf[3] + map[i*4 + 3]*buf[3];
    }

    atomicAdd(&dbias[threadIdx.x],bias_add);

    float buf_out[6*6];

    #pragma unroll
    for(int i=0; i<6; i++){
        buf_out[i*6]     = tile_reg[i];
        buf_out[i*6 + 1] = tile_reg[i] + (SQRT2/2.f)*tile_reg[i + 6] + 0.5f*tile_reg[i + 12]    + (SQRT2/4.f)*tile_reg[i + 18];
        buf_out[i*6 + 2] = tile_reg[i] - (SQRT2/2.f)*tile_reg[i + 6] + 0.5f*tile_reg[i + 12]    - (SQRT2/4.f)*tile_reg[i + 18];
        buf_out[i*6 + 3] = tile_reg[i] + SQRT2*tile_reg[i + 6]       + 2*tile_reg[i + 12]       + 2*SQRT2*tile_reg[i + 18];
        buf_out[i*6 + 4] = tile_reg[i] - SQRT2*tile_reg[i + 6]       + 2*tile_reg[i + 12]       - 2*SQRT2*tile_reg[i + 18];
        buf_out[i*6 + 5] = tile_reg[i + 18];
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE]                              = buf_out[i*6];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 1*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 1];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 2*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 2];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 3*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 3];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 4*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 4];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 5*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 5];
    }
}

//ONLY PADDED
template<const int FILTERS, const int IN_DIM, const int BATCH_SIZE, const int TILES_1D, const int TILES, const int OUT_CUT>
__global__ __launch_bounds__(512,1) void atma_back_pad_batch_pool_relu_gpu(const float* __restrict__ data, float* output, const char* __restrict__ pool_relu_idx, float* __restrict__ dbias){ //input NxIN_DIMxIN_DIMxFILTERS  output 36x(BATCH_SIZE*TILES)xFILTERS
    constexpr float SQRT2 = M_SQRT2;
    
    int out_idx = blockIdx.z*TILES*FILTERS + blockIdx.y*FILTERS*TILES_1D + blockIdx.x*FILTERS + threadIdx.x;
    int data_idx = blockIdx.z*(IN_DIM/2)*(IN_DIM/2)*FILTERS + blockIdx.y*FILTERS*(IN_DIM/2)*2 - OUT_CUT*FILTERS*(IN_DIM/2) + blockIdx.x*FILTERS*2 - OUT_CUT*FILTERS + threadIdx.x;

    float tile_reg[4*6];

    float bias_add=0;

    int in_x_idx = threadIdx.x + blockIdx.z*IN_DIM*IN_DIM*FILTERS;
    int in_y_idx = (blockIdx.x*4 - OUT_CUT)*FILTERS;
    int in_z_idx = (blockIdx.y*4 - OUT_CUT)*FILTERS*IN_DIM;

    int data_x_idx = blockIdx.x*2 - OUT_CUT;
    int data_y_idx = blockIdx.y*2 - OUT_CUT;
    

    float buf[9]={0};
    if(data_x_idx >= 0      && data_x_idx < IN_DIM/2     && data_y_idx >=0      && data_y_idx < IN_DIM/2)     buf[0] = data[data_idx]; 
    if(data_x_idx >= 0      && data_x_idx < IN_DIM/2     && data_y_idx + 1 >=0  && data_y_idx + 1 < IN_DIM/2) buf[1] = data[data_idx + FILTERS*(IN_DIM/2)];
    if(data_x_idx >= 0      && data_x_idx < IN_DIM/2     && data_y_idx + 2 >=0  && data_y_idx + 2 < IN_DIM/2) buf[2] = data[data_idx + 2*FILTERS*(IN_DIM/2)];

    if(data_x_idx + 1 >= 0  && data_x_idx + 1 < IN_DIM/2 && data_y_idx >=0      && data_y_idx < IN_DIM/2)     buf[3] = data[data_idx + FILTERS];
    if(data_x_idx + 1 >= 0  && data_x_idx + 1 < IN_DIM/2 && data_y_idx + 1 >=0  && data_y_idx + 1 < IN_DIM/2) buf[4] = data[data_idx + FILTERS*(IN_DIM/2) + FILTERS];
    if(data_x_idx + 1 >= 0  && data_x_idx + 1 < IN_DIM/2 && data_y_idx + 2 >=0  && data_y_idx + 2 < IN_DIM/2) buf[5] = data[data_idx + 2*FILTERS*(IN_DIM/2) + FILTERS];

    if(data_x_idx + 2 >= 0  && data_x_idx + 2 < IN_DIM/2 && data_y_idx >=0      && data_y_idx < IN_DIM/2)     buf[6] = data[data_idx + 2*FILTERS];
    if(data_x_idx + 2 >= 0  && data_x_idx + 2 < IN_DIM/2 && data_y_idx + 1 >=0  && data_y_idx + 1 < IN_DIM/2) buf[7] = data[data_idx + FILTERS*(IN_DIM/2) + 2*FILTERS];
    if(data_x_idx + 2 >= 0  && data_x_idx + 2 < IN_DIM/2 && data_y_idx + 2 >=0  && data_y_idx + 2 < IN_DIM/2) buf[8] = data[data_idx + 2*FILTERS*(IN_DIM/2) + 2*FILTERS];

    float map[4*4]={0};

    bool pred_top_y = in_z_idx >= 0;
    bool pred_bot_y = in_z_idx + 3*FILTERS*IN_DIM <= (IN_DIM - 1)*FILTERS*IN_DIM;

    #pragma unroll
    for(int i=0; i<4; i++){

        bool pred_left_x = in_y_idx >= 0;
        bool pred_right_x = in_y_idx <= (IN_DIM-1)*FILTERS;
        
        if (pred_left_x && pred_right_x && pred_top_y)  map[i*4] = pool_relu_idx[in_z_idx + in_y_idx + in_x_idx];
        if (pred_left_x && pred_right_x)                map[i*4 + 1] = pool_relu_idx[in_z_idx + FILTERS*IN_DIM + in_y_idx + in_x_idx];
        if (pred_left_x && pred_right_x)                map[i*4 + 2] = pool_relu_idx[in_z_idx + 2*FILTERS*IN_DIM + in_y_idx + in_x_idx];
        if (pred_left_x && pred_right_x && pred_bot_y)  map[i*4 + 3] = pool_relu_idx[in_z_idx + 3*FILTERS*IN_DIM + in_y_idx + in_x_idx];

        in_y_idx += FILTERS;
    }

    #pragma unroll
    for(int i=0; i<1; i++){
        tile_reg[i*6 + 0] = map[i*4 + 0]*buf[0];
        tile_reg[i*6 + 1] = map[i*4 + 0]*buf[0] + (SQRT2/2.f)*map[i*4 + 1]*buf[1] + 0.5f*map[i*4 + 2]*buf[1] + (SQRT2/4.f)*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 2] = map[i*4 + 0]*buf[0] - (SQRT2/2.f)*map[i*4 + 1]*buf[1] + 0.5f*map[i*4 + 2]*buf[1] - (SQRT2/4.f)*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 3] = map[i*4 + 0]*buf[0] + SQRT2*map[i*4 + 1]*buf[1]       + 2*map[i*4 + 2]*buf[1]    + 2*SQRT2*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 4] = map[i*4 + 0]*buf[0] - SQRT2*map[i*4 + 1]*buf[1]       + 2*map[i*4 + 2]*buf[1]    - 2*SQRT2*map[i*4 + 3]*buf[2];
        tile_reg[i*6 + 5] = map[i*4 + 3]*buf[2];

        bias_add += map[i*4 + 0]*buf[0] + map[i*4 + 1]*buf[1] +  map[i*4 + 2]*buf[1] + map[i*4 + 3]*buf[2];
    }

    #pragma unroll
    for(int i=1; i<3; i++){
        tile_reg[i*6 + 0] = map[i*4 + 0]*buf[3];
        tile_reg[i*6 + 1] = map[i*4 + 0]*buf[3] + (SQRT2/2.f)*map[i*4 + 1]*buf[4] + 0.5f*map[i*4 + 2]*buf[4] + (SQRT2/4.f)*map[i*4 + 3]*buf[5];
        tile_reg[i*6 + 2] = map[i*4 + 0]*buf[3] - (SQRT2/2.f)*map[i*4 + 1]*buf[4] + 0.5f*map[i*4 + 2]*buf[4] - (SQRT2/4.f)*map[i*4 + 3]*buf[5];
        tile_reg[i*6 + 3] = map[i*4 + 0]*buf[3] + SQRT2*map[i*4 + 1]*buf[4]       + 2*map[i*4 + 2]*buf[4]    + 2*SQRT2*map[i*4 + 3]*buf[5];
        tile_reg[i*6 + 4] = map[i*4 + 0]*buf[3] - SQRT2*map[i*4 + 1]*buf[4]       + 2*map[i*4 + 2]*buf[4]    - 2*SQRT2*map[i*4 + 3]*buf[5];
        tile_reg[i*6 + 5] = map[i*4 + 3]*buf[5];

        bias_add += map[i*4 + 0]*buf[3] + map[i*4 + 1]*buf[4] +  map[i*4 + 2]*buf[4] + map[i*4 + 3]*buf[5];
    }  

    #pragma unroll
    for(int i=3; i<4; i++){
        tile_reg[i*6 + 0] = map[i*4 + 0]*buf[6];
        tile_reg[i*6 + 1] = map[i*4 + 0]*buf[6] + (SQRT2/2.f)*map[i*4 + 1]*buf[7] + 0.5f*map[i*4 + 2]*buf[7] + (SQRT2/4.f)*map[i*4 + 3]*buf[8];
        tile_reg[i*6 + 2] = map[i*4 + 0]*buf[6] - (SQRT2/2.f)*map[i*4 + 1]*buf[7] + 0.5f*map[i*4 + 2]*buf[7] - (SQRT2/4.f)*map[i*4 + 3]*buf[8];
        tile_reg[i*6 + 3] = map[i*4 + 0]*buf[6] + SQRT2*map[i*4 + 1]*buf[7]       + 2*map[i*4 + 2]*buf[7]    + 2*SQRT2*map[i*4 + 3]*buf[8];
        tile_reg[i*6 + 4] = map[i*4 + 0]*buf[6] - SQRT2*map[i*4 + 1]*buf[7]       + 2*map[i*4 + 2]*buf[7]    - 2*SQRT2*map[i*4 + 3]*buf[8];
        tile_reg[i*6 + 5] = map[i*4 + 3]*buf[8];

        bias_add += map[i*4 + 0]*buf[6] + map[i*4 + 1]*buf[7] +  map[i*4 + 2]*buf[7] + map[i*4 + 3]*buf[8];
    }

    atomicAdd(&dbias[threadIdx.x],bias_add);

    float buf_out[6*6];

    #pragma unroll
    for(int i=0; i<6; i++){
        buf_out[i*6] = tile_reg[i];
        buf_out[i*6 + 1] = tile_reg[i] + (SQRT2/2.f)*tile_reg[i + 6] + 0.5f*tile_reg[i + 12] + (SQRT2/4.f)*tile_reg[i + 18];
        buf_out[i*6 + 2] = tile_reg[i] - (SQRT2/2.f)*tile_reg[i + 6] + 0.5f*tile_reg[i + 12] - (SQRT2/4.f)*tile_reg[i + 18];
        buf_out[i*6 + 3] = tile_reg[i] + SQRT2*tile_reg[i + 6]       + 2*tile_reg[i + 12]    + 2*SQRT2*tile_reg[i + 18];
        buf_out[i*6 + 4] = tile_reg[i] - SQRT2*tile_reg[i + 6]       + 2*tile_reg[i + 12]    - 2*SQRT2*tile_reg[i + 18];
        buf_out[i*6 + 5] = tile_reg[i + 18];
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 1*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 1];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 2*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 2];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 3*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 3];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 4*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 4];
        output[out_idx + i*6*TILES*FILTERS*BATCH_SIZE + 5*TILES*FILTERS*BATCH_SIZE] = buf_out[i*6 + 5];
    }
}

template<const int FILTERS, const int IN_DIM, const int BATCH_SIZE, const int TILES_1D, const int TILES, const int OUT_CUT, const int THREADS_X, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int POOL>
void atma_back(float* input, float* output, char* pool_relu_idx, float* dbias, cudaStream_t stream = 0){
    dim3 threads(THREADS_X);
    dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

    zero_out_dbias<<<dim3(1,1,1),dim3(FILTERS/4),0,stream>>>(dbias);

    if(POOL==1){
        if(OUT_CUT==0){
            atma_back_batch_pool_relu_gpu<FILTERS, IN_DIM, BATCH_SIZE, TILES_1D, TILES><<<blocks, threads, 0, stream>>>(input,output,pool_relu_idx,dbias);
        }
        else{
            atma_back_pad_batch_pool_relu_gpu<FILTERS, IN_DIM, BATCH_SIZE, TILES_1D, TILES, OUT_CUT><<<blocks, threads, 0, stream>>>(input,output,pool_relu_idx,dbias);
        }
    }
    else{
        atma_back_pad_batch_relu_gpu<FILTERS, IN_DIM, BATCH_SIZE, TILES_1D, TILES, OUT_CUT><<<blocks, threads, 0, stream>>>(input,output,pool_relu_idx,dbias);
    }
}



