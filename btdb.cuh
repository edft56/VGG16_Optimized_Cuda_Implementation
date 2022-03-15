#pragma once

#include <cmath>

//INPUT TRANSFORM FUNCTIONS

struct btdb_forward_data{ //CHANNELS,HW,BATCH_SIZE
    const int CHANNELS,HW,BATCH_SIZE;
    const int TILES_1D  = (HW + 3) / 4;
    const int IN_PAD    = 1 + (HW%4)/2;
    const int TILES     = TILES_1D*TILES_1D;
    
    const int CHANNELS_THREADBLOCK = CHANNELS;

    const int IN_SIZE   = HW*HW*CHANNELS*BATCH_SIZE;
    const int OUT_SIZE  = 36*(TILES)*CHANNELS*BATCH_SIZE;
    
    const int BLOCKS_X  = TILES;
    const int BLOCKS_Y  = CHANNELS/CHANNELS_THREADBLOCK;
    const int BLOCKS_Z  = BATCH_SIZE;
};

struct btdb_back_data{ //CHANNELS,OUT_DIM,BATCH_SIZE
    const int CHANNELS, OUT_DIM, BATCH_SIZE;

    const int IN_PAD    = 1 + (OUT_DIM%4)/2;
    const int TILES_1D  = (OUT_DIM + 3) / 4;
    const int TILES     = TILES_1D*TILES_1D;  

    const int THREADS_X = 32;
    
    const int BLOCKS_X  = TILES;
    const int BLOCKS_Y  = BATCH_SIZE;
    const int BLOCKS_Z  = CHANNELS/THREADS_X;
   
    const int stride    = TILES*CHANNELS*BATCH_SIZE;

    const int IN_SIZE   = BATCH_SIZE*36*TILES*CHANNELS;
    const int OUT_SIZE  = BATCH_SIZE*OUT_DIM*OUT_DIM*CHANNELS;
};



template<const int CHANNELS, const int HW, const int BATCH_SIZE, const int IN_PAD, const int TILES, const int TILES_1D, const int CHANNELS_THREADBLOCK>
__global__ __launch_bounds__(CHANNELS,1) void BtdB_batched(const float* __restrict__ input, float* __restrict__ output){ // in BATCH_SIZExHWxHWxCHANNELS  out 36xBATCH_SIZExTILESxCHANNELS
    constexpr float SQRT2 = M_SQRT2;

    int out_idx = blockIdx.z*CHANNELS*TILES + blockIdx.y*CHANNELS_THREADBLOCK + blockIdx.x*CHANNELS + threadIdx.y*CHANNELS + threadIdx.x;

    float tile_reg[36];
    float v[36]={0};

    int in_x_idx    = blockIdx.y*CHANNELS_THREADBLOCK + threadIdx.x;
    int in_tile_y   = threadIdx.y                     + (blockIdx.x%TILES_1D);
    int in_tile_z   = blockIdx.x/TILES_1D;
    int in_w_idx    = blockIdx.z*CHANNELS*HW*HW;

    int in_z_idx0 = in_tile_z*4 - IN_PAD; 
    int in_z_idx1 = in_tile_z*4 +       1 - IN_PAD;
    int in_z_idx2 = in_tile_z*4 +       2 - IN_PAD;
    int in_z_idx3 = in_tile_z*4 +       3 - IN_PAD;
    int in_z_idx4 = in_tile_z*4 +       4 - IN_PAD;
    int in_z_idx5 = in_tile_z*4 +       5 - IN_PAD;


    #pragma unroll
    for(int i=0; i<6; i++){ //top bottom pad
        int in_y_idx = in_tile_y*4 + i - IN_PAD;
        
        if(!(in_y_idx<0 || in_z_idx0<0 || in_y_idx>HW-1 || in_z_idx0>HW-1)) v[i*6 + 0] = input[in_w_idx + in_z_idx0*CHANNELS*HW + in_y_idx*CHANNELS + in_x_idx];
        if(!(in_y_idx<0 || in_z_idx1<0 || in_y_idx>HW-1 || in_z_idx1>HW-1)) v[i*6 + 1] = input[in_w_idx + in_z_idx1*CHANNELS*HW + in_y_idx*CHANNELS + in_x_idx];
        if(!(in_y_idx<0 || in_z_idx2<0 || in_y_idx>HW-1 || in_z_idx2>HW-1)) v[i*6 + 2] = input[in_w_idx + in_z_idx2*CHANNELS*HW + in_y_idx*CHANNELS + in_x_idx];
        if(!(in_y_idx<0 || in_z_idx3<0 || in_y_idx>HW-1 || in_z_idx3>HW-1)) v[i*6 + 3] = input[in_w_idx + in_z_idx3*CHANNELS*HW + in_y_idx*CHANNELS + in_x_idx];
        if(!(in_y_idx<0 || in_z_idx4<0 || in_y_idx>HW-1 || in_z_idx4>HW-1)) v[i*6 + 4] = input[in_w_idx + in_z_idx4*CHANNELS*HW + in_y_idx*CHANNELS + in_x_idx];
        if(!(in_y_idx<0 || in_z_idx5<0 || in_y_idx>HW-1 || in_z_idx5>HW-1)) v[i*6 + 5] = input[in_w_idx + in_z_idx5*CHANNELS*HW + in_y_idx*CHANNELS + in_x_idx];
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        tile_reg[i*6]     = v[i*6 + 0]              - (5.f/2.f)*v[i*6 + 2]  + v[i*6 + 4];
        tile_reg[i*6 + 1] = -SQRT2*v[i*6 + 1]       - 2*v[i*6 + 2]          + (SQRT2/2.f)*v[i*6 + 3] + v[i*6 + 4];
        tile_reg[i*6 + 2] = SQRT2*v[i*6 + 1]        - 2*v[i*6 + 2]          - (SQRT2/2.f)*v[i*6 + 3] + v[i*6 + 4];
        tile_reg[i*6 + 3] = (-SQRT2/2.f)*v[i*6 + 1] - (1.f/2.f)*v[i*6 + 2]  + SQRT2*v[i*6 + 3]       + v[i*6 + 4];
        tile_reg[i*6 + 4] = (SQRT2/2.f)*v[i*6 + 1]  - (1.f/2.f)*v[i*6 + 2]  - SQRT2*v[i*6 + 3]       + v[i*6 + 4];
        tile_reg[i*6 + 5] = v[i*6 + 1]              - (5.f/2.f)*v[i*6 + 3]  + v[i*6 + 5];
    }

    #pragma unroll
    for(int i=0; i<6; i++){
        output[out_idx + i*TILES*CHANNELS*BATCH_SIZE*6]                               = tile_reg[i]                  - (5.f/2.f)*tile_reg[i + 12] + tile_reg[i + 24];
        output[out_idx + i*TILES*CHANNELS*BATCH_SIZE*6 + TILES*CHANNELS*BATCH_SIZE]   = -SQRT2*tile_reg[i + 6]       - 2*tile_reg[i + 12]         + (SQRT2/2.f)*tile_reg[i + 18] + tile_reg[i + 24];
        output[out_idx + i*TILES*CHANNELS*BATCH_SIZE*6 + 2*TILES*CHANNELS*BATCH_SIZE] = SQRT2*tile_reg[i + 6]        - 2*tile_reg[i + 12]         - (SQRT2/2.f)*tile_reg[i + 18] + tile_reg[i + 24];
        output[out_idx + i*TILES*CHANNELS*BATCH_SIZE*6 + 3*TILES*CHANNELS*BATCH_SIZE] = (-SQRT2/2.f)*tile_reg[i + 6] - (1.f/2.f)*tile_reg[i + 12] + SQRT2*tile_reg[i + 18]       + tile_reg[i + 24];
        output[out_idx + i*TILES*CHANNELS*BATCH_SIZE*6 + 4*TILES*CHANNELS*BATCH_SIZE] = (SQRT2/2.f)*tile_reg[i + 6]  - (1.f/2.f)*tile_reg[i + 12] - SQRT2*tile_reg[i + 18]       + tile_reg[i + 24];
        output[out_idx + i*TILES*CHANNELS*BATCH_SIZE*6 + 5*TILES*CHANNELS*BATCH_SIZE] = tile_reg[i + 6]              - (5.f/2.f)*tile_reg[i + 18] + tile_reg[i + 30];
    }
}

template<const int CHANNELS,const int HW,const int BATCH_SIZE,const int IN_PAD,const int TILES,const int TILES_1D,const int CHANNELS_THREADBLOCK,const int BLOCKS_X,const int BLOCKS_Y, const int BLOCKS_Z>
void btdb_forward(float* d_input, float* d_output, cudaStream_t stream = 0){
    
    dim3 threads(CHANNELS_THREADBLOCK);
    dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

    
    BtdB_batched<CHANNELS, HW, BATCH_SIZE, IN_PAD, TILES, TILES_1D, CHANNELS_THREADBLOCK>
    <<<blocks, threads, 0 , stream>>>(d_input,d_output);

}

template<const int CHANNELS,const int OUT_DIM,const int IN_PAD,const int TILES_1D,const int TILES,const int THREADS_X,const int stride>
__global__ __launch_bounds__(32,16) void Btdb_back_batch_gpu_new(const float* __restrict__ input, float* __restrict__ output){ //in 36x(BATCH_SIZE*TILES)xCHANNELS  out BATCH_SIZE*HW*HW*CHANNELS
    constexpr float SQRT2 = M_SQRT2;
    
    int in_idx  = blockIdx.z*THREADS_X + blockIdx.y*TILES*CHANNELS           + blockIdx.x*CHANNELS                      + threadIdx.x;
    int out_idx = blockIdx.z*THREADS_X + blockIdx.y*OUT_DIM*OUT_DIM*CHANNELS + (blockIdx.x/TILES_1D)*CHANNELS*OUT_DIM*4 + (blockIdx.x%TILES_1D)*CHANNELS*4 - CHANNELS*OUT_DIM*IN_PAD - CHANNELS*IN_PAD +  threadIdx.x;


    bool x_left_pred        = (blockIdx.x%TILES_1D)*4 > IN_PAD-1;      //X LEFT EDGE PREDICATES
    bool x_left_pred2       = (blockIdx.x%TILES_1D)*4 + 1 > IN_PAD-1;
    bool y_bottom_edge_pred = blockIdx.x/TILES_1D == TILES_1D - 1;
    bool x_right_edge_pred  = blockIdx.x%TILES_1D == TILES_1D - 1;
    bool x_left_edge_pred   = blockIdx.x%TILES_1D > 0;
    bool y_upper_edge_pred  = blockIdx.x/(TILES_1D) > 0;



    // READ DIAG EDGE OF PREVIOUS TILES (2X4), IF NOT UPPER EDGE TILE AND NOT LEFT EDGE TILE
    int diag_tile_idx = in_idx - CHANNELS*TILES_1D - CHANNELS;
    float buf_diag[36];

    if( x_left_edge_pred && y_upper_edge_pred ){
        #pragma unroll
        for(int i=0; i<6; i++){
            #pragma unroll
            for(int j=0; j<6; j++){
                buf_diag[i*6+j] = input[diag_tile_idx + i*stride + j*stride*6];
            }
        }
    }



    // READ BOTTOM EDGE OF PREVIOUS Y TILE (2X4), IF NOT UPPER EDGE TILE
    int up_idx = in_idx - CHANNELS*TILES_1D;
    float buf_up[6*6];
    if( y_upper_edge_pred ){
        #pragma unroll
        for(int i=0; i<6; i++){
            #pragma unroll
            for(int j=0; j<6; j++){
                buf_up[i*6+j] = input[up_idx + i*stride + j*stride*6];
            }
        }
    }




    float diag_tile_reg[2*6];
    // COMPUTE INTERMEDIATE DIAG EDGE OF PREVIOUS TILES (2X4), IF NOT UPPER EDGE TILE AND NOT LEFT EDGE TILE
    #pragma unroll
    for(int i=0; i<6; i++){
        if( x_left_edge_pred && y_upper_edge_pred )diag_tile_reg[i*2]     = buf_diag[i*6] + buf_diag[i*6 + 1] + buf_diag[i*6 + 2] + buf_diag[i*6 + 3] + buf_diag[i*6 + 4];
        if( x_left_edge_pred && y_upper_edge_pred )diag_tile_reg[i*2 + 1] = buf_diag[i*6 + 5];
    }
              
    


    

    float up_tile_reg[2*6];
    // COMPUTE INTERMEDIATE BOTTOM EDGE OF PREVIOUS Y TILE (2X4), IF NOT UPPER EDGE TILE
    #pragma unroll
    for(int i=0; i<6; i++){ 
        if( y_upper_edge_pred )up_tile_reg[i*2]     = buf_up[i*6] + buf_up[i*6 + 1] + buf_up[i*6 + 2] + buf_up[i*6 + 3] + buf_up[i*6 + 4];
        if( y_upper_edge_pred )up_tile_reg[i*2 + 1] = buf_up[i*6 + 5];
    }
    
    


    

    float buf_main[6*6];
    // READ MAIN TILE INPUT
    #pragma unroll
    for(int i=0; i<6; i++){
        #pragma unroll
        for(int j=0; j<6; j++){
            buf_main[i*6+j] = input[in_idx + i*stride + j*stride*6];
        }
    }

    float tile_reg[6*6];
    // COMPUTE INTERMEDIATE MAIN 4x4 TILE
    #pragma unroll
    for(int i=0; i<6; i++){
        tile_reg[i*6]     = buf_main[i*6 + 0];
        tile_reg[i*6 + 1] =                             - SQRT2*buf_main[i*6 + 1]       + SQRT2*buf_main[i*6 + 2]       - (SQRT2/2.f)*buf_main[i*6 + 3] + (SQRT2/2.f)*buf_main[i*6 + 4] + 1*buf_main[i*6 + 5];
        tile_reg[i*6 + 2] = (-5.f/2.f)*buf_main[i*6]    - 2*buf_main[i*6 + 1]           - 2*buf_main[i*6 + 2]           - (1.f/2.f)*buf_main[i*6 + 3]   - (1.f/2.f)*buf_main[i*6 + 4];
        tile_reg[i*6 + 3] =                             + (SQRT2/2.f)*buf_main[i*6 + 1] - (SQRT2/2.f)*buf_main[i*6 + 2] + SQRT2*buf_main[i*6 + 3]       - SQRT2*buf_main[i*6 + 4]       - (5.f/2.f)*buf_main[i*6 + 5];
    }

    // COMPUTE INTERMEDIATE BOTTOM 2X4, IF Y EDGE TILE
    #pragma unroll
    for(int i=0; i<6; i++){
        if( y_bottom_edge_pred )tile_reg[i*6 + 4] = buf_main[i*6] + buf_main[i*6 + 1] + buf_main[i*6 + 2] + buf_main[i*6 + 3] + buf_main[i*6 + 4];
        if( y_bottom_edge_pred )tile_reg[i*6 + 5] = buf_main[i*6 + 5];
    }








    float buf_left[6*6];
    // READ RIGHT EDGE OF PREVIOUS X TILE (4X2), IF NOT LEFT EDGE TILE
    int left_idx = in_idx - CHANNELS;
    if(x_left_edge_pred){
        #pragma unroll
        for(int i=0; i<6; i++){
            #pragma unroll
            for(int j=0; j<6; j++){
                buf_left[i*6+j] = input[left_idx + i*stride + j*stride*6];
            }
        }
    }


    float left_tile_reg[6*6];
    // COMPUTE INTERMEDIATE RIGHT EDGE OF PREVIOUS X TILE (4X2), IF NOT LEFT EDGE TILE
    #pragma unroll
    for(int i=0; i<6; i++){
        if(x_left_edge_pred) left_tile_reg[i*6]     =     buf_left[i*6 + 0];
        if(x_left_edge_pred) left_tile_reg[i*6 + 1] =                           -SQRT2*buf_left[i*6 + 1]      + SQRT2*buf_left[i*6 + 2]       - (SQRT2/2.f)*buf_left[i*6 + 3] + (SQRT2/2.f)*buf_left[i*6 + 4] + 1*buf_left[i*6 + 5];
        if(x_left_edge_pred) left_tile_reg[i*6 + 2] = (-5.f/2.f)*buf_left[i*6]  -2*buf_left[i*6 + 1]          - 2*buf_left[i*6 + 2]           - (1.f/2.f)*buf_left[i*6 + 3]   - (1.f/2.f)*buf_left[i*6 + 4];
        if(x_left_edge_pred) left_tile_reg[i*6 + 3] =                           (SQRT2/2.f)*buf_left[i*6 + 1] - (SQRT2/2.f)*buf_left[i*6 + 2] + SQRT2*buf_left[i*6 + 3]       - SQRT2*buf_left[i*6 + 4]       - (5.f/2.f)*buf_left[i*6 + 5];
    }

    //COMPUTE THE REMAINING INTERMEDIATE OVERLAPPING LEFT TILE IF ITS A Y EDGE TILE 
    #pragma unroll
    for(int i=0; i<6; i++){
        if(x_left_edge_pred && y_bottom_edge_pred) left_tile_reg[i*6 + 4] = buf_left[i*6]       + buf_left[i*6 + 1] + buf_left[i*6 + 2] + buf_left[i*6 + 3] + buf_left[i*6 + 4];
        if(x_left_edge_pred && y_bottom_edge_pred) left_tile_reg[i*6 + 5] = buf_left[i*6 + 5];
    }











    float out[6*6];

    // COMPUTE RIGHT 4X2, IF X EDGE TILE
    #pragma unroll
    for(int i=0; i<4; i++){
        if( x_right_edge_pred ) out[i*6 + 4] = tile_reg[i]          + tile_reg[i + 1*6] + tile_reg[i + 2*6] + tile_reg[i + 3*6] + tile_reg[i + 4*6];
        if( x_right_edge_pred ) out[i*6 + 5] = tile_reg[i + 5*6];
    }
    

    // WRITE HALF OF RIGHT EDGE(4x2), IF X EDGE TILE
    #pragma unroll
    for(int i=2; i<4; i++){ 
        if(IN_PAD<2 && x_right_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 4*CHANNELS] = out[i*6 + 4];
        if(IN_PAD<1 && x_right_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 5*CHANNELS] = out[i*6 + 5];
    }

    //IF ITS A X EDGE TILE ADD THE OVERLAPPING UPPER TILE
    #pragma unroll
    for(int i=0; i<2; i++){
        if(x_right_edge_pred && y_upper_edge_pred) out[i*6 + 4] += up_tile_reg[i]       + up_tile_reg[i + 1*2] + up_tile_reg[i + 2*2] + up_tile_reg[i + 3*2] + up_tile_reg[i + 4*2];
        if(x_right_edge_pred && y_upper_edge_pred) out[i*6 + 5] += up_tile_reg[i + 5*6];
    }
    

    #pragma unroll
    for(int i=0; i<2; i++){ 
        bool y_up_pad_pred = (blockIdx.x/TILES_1D)*4 + i > IN_PAD-1;

        if(IN_PAD<2 && y_up_pad_pred && x_right_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 4*CHANNELS] = out[i*6 + 4];
        if(IN_PAD<1 && y_up_pad_pred && x_right_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 5*CHANNELS] = out[i*6 + 5];
    }



    // COMPUTE AND WRITE DIAGONAL 2X2, IF LAST TILE ON DIAGONAL
    #pragma unroll
    for(int i=4; i<6; i++){
        if(y_bottom_edge_pred && x_right_edge_pred)out[i*6 + 4] = tile_reg[i]       + tile_reg[i + 1*6] + tile_reg[i + 2*6] + tile_reg[i + 3*6] + tile_reg[i + 4*6];
        if(y_bottom_edge_pred && x_right_edge_pred)out[i*6 + 5] = tile_reg[i + 5*6];
    }
    #pragma unroll
    for(int i=4; i<6; i++){ 
        bool y_bottom_pad_pred = i < 6-IN_PAD;

        if(IN_PAD<2 && y_bottom_pad_pred && y_bottom_edge_pred && x_right_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 4*CHANNELS] = out[i*6 + 4];
        if(IN_PAD<1 && y_bottom_pad_pred && y_bottom_edge_pred && x_right_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 5*CHANNELS] = out[i*6 + 5];
    }

    // COMPUTE MAIN 4X4 TILE
    #pragma unroll
    for(int i=0; i<4; i++){
        out[i*6]     =      1*tile_reg[i];
        out[i*6 + 1] =                         - SQRT2*tile_reg[i + 1*6]        + SQRT2*tile_reg[i + 2*6]       - (SQRT2/2.f)*tile_reg[i + 3*6] + (SQRT2/2.f)*tile_reg[i + 4*6] + 1*tile_reg[i + 5*6];
        out[i*6 + 2] =  -(5.f/2.f)*tile_reg[i] - 2*tile_reg[i + 1*6]            - 2*tile_reg[i + 2*6]           - (1.f/2.f)*tile_reg[i + 3*6]   - (1.f/2.f)*tile_reg[i + 4*6];
        out[i*6 + 3] =                         + (SQRT2/2.f)*tile_reg[i + 1*6]  - (SQRT2/2.f)*tile_reg[i + 2*6] + SQRT2*tile_reg[i + 3*6]       - SQRT2*tile_reg[i + 4*6]       - (5.f/2.f)*tile_reg[i + 5*6];
    }

    


    
    
    // COMPUTE BOTTOM 2X4, IF Y EDGE TILE
    #pragma unroll
    for(int i=4; i<6; i++){
        if( y_bottom_edge_pred ) out[i*6]     =     1*tile_reg[i];
        if( y_bottom_edge_pred ) out[i*6 + 1] =                        - SQRT2*tile_reg[i + 1*6]       + SQRT2*tile_reg[i + 2*6]       - (SQRT2/2.f)*tile_reg[i + 3*6] + (SQRT2/2.f)*tile_reg[i + 4*6] + 1*tile_reg[i + 5*6];
        if( y_bottom_edge_pred ) out[i*6 + 2] = -(5.f/2.f)*tile_reg[i] - 2*tile_reg[i + 1*6]           - 2*tile_reg[i + 2*6]           - (1.f/2.f)*tile_reg[i + 3*6]   - (1.f/2.f)*tile_reg[i + 4*6];
        if( y_bottom_edge_pred ) out[i*6 + 3] =                        + (SQRT2/2.f)*tile_reg[i + 1*6] - (SQRT2/2.f)*tile_reg[i + 2*6] + SQRT2*tile_reg[i + 3*6]       - SQRT2*tile_reg[i + 4*6]       - (5.f/2.f)*tile_reg[i + 5*6];
    }         
    







    //COMPUTE AND ADD RIGHT EDGE OF PREVIOUS X TILE (4X2), IF NOT LEFT EDGE TILE
    #pragma unroll
    for(int i=0; i<4; i++){
        if(x_left_edge_pred)out[i*6]     += left_tile_reg[i]       + left_tile_reg[i + 1*6] + left_tile_reg[i + 2*6] + left_tile_reg[i + 3*6] + left_tile_reg[i + 4*6];
        if(x_left_edge_pred)out[i*6 + 1] += left_tile_reg[i + 5*6];
    }

    
    //COMPUTE AND ADD THE OVERLAPPING LEFT TILE, IF ITS A Y EDGE TILE
    
    #pragma unroll
    for(int i=4; i<6; i++){
        if(x_left_edge_pred && y_bottom_edge_pred)out[i*6]     += left_tile_reg[i] + left_tile_reg[i + 1*6] + left_tile_reg[i + 2*6] + left_tile_reg[i + 3*6] + left_tile_reg[i + 4*6];
        if(x_left_edge_pred && y_bottom_edge_pred)out[i*6 + 1] += left_tile_reg[i + 5*6];
    }
    

    // WRITE BOTTOM EDGE(2x4), IF Y EDGE TILE
    #pragma unroll
    for(int i=4; i<6; i++){ 
        bool y_bottom_pad_pred = i < 6-IN_PAD;
        
        if(x_left_pred && y_bottom_pad_pred && y_bottom_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM]               = out[i*6];
        if(x_left_pred2 && y_bottom_pad_pred && y_bottom_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 1*CHANNELS] = out[i*6 + 1];
        if(y_bottom_pad_pred && y_bottom_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 2*CHANNELS]                 = out[i*6 + 2];
        if(y_bottom_pad_pred && y_bottom_edge_pred) output[out_idx + i*CHANNELS*OUT_DIM + 3*CHANNELS]                 = out[i*6 + 3];  
    }  






    // COMPUTE DIAG EDGE OF PREVIOUS TILES (2X4) AND ADD IT, IF NOT UPPER EDGE TILE AND NOT LEFT EDGE TILE
    #pragma unroll
    for(int i=0; i<2; i++){
        if( x_left_edge_pred && y_upper_edge_pred )out[i*6]     += diag_tile_reg[i]        + diag_tile_reg[i + 1*2] + diag_tile_reg[i + 2*2] + diag_tile_reg[i + 3*2] + diag_tile_reg[i + 4*2];
        if( x_left_edge_pred && y_upper_edge_pred )out[i*6 + 1] += diag_tile_reg[i + 5*2];
    }




    // COMPUTE AND ADD BOTTOM EDGE OF PREVIOUS Y TILE (2X4), IF NOT UPPER EDGE TILE
    #pragma unroll
    for(int i=0; i<2; i++){
        if(y_upper_edge_pred)out[i*6]     +=     1*up_tile_reg[i];
        if(y_upper_edge_pred)out[i*6 + 1] +=                           - SQRT2*up_tile_reg[i + 1*2]       + SQRT2*up_tile_reg[i + 2*2]       - (SQRT2/2.f)*up_tile_reg[i + 3*2] + (SQRT2/2.f)*up_tile_reg[i + 4*2] + 1*up_tile_reg[i + 5*2];
        if(y_upper_edge_pred)out[i*6 + 2] += -(5.f/2.f)*up_tile_reg[i] - 2*up_tile_reg[i + 1*2]           - 2*up_tile_reg[i + 2*2]           - (1.f/2.f)*up_tile_reg[i + 3*2]   - (1.f/2.f)*up_tile_reg[i + 4*2];
        if(y_upper_edge_pred)out[i*6 + 3] +=                           + (SQRT2/2.f)*up_tile_reg[i + 1*2] - (SQRT2/2.f)*up_tile_reg[i + 2*2] + SQRT2*up_tile_reg[i + 3*2]       - SQRT2*up_tile_reg[i + 4*2]       - (5.f/2.f)*up_tile_reg[i + 5*2];
    }

    

    




    // WRITE HALF OF MAIN TILE
    #pragma unroll
    for(int i=0; i<2; i++){
        bool y_up_pad_pred = (blockIdx.x/TILES_1D)*4 + i > IN_PAD-1;
        
        if(x_left_pred && y_up_pad_pred) output[out_idx + i*CHANNELS*OUT_DIM]               = out[i*6];
        if(x_left_pred2 && y_up_pad_pred) output[out_idx + i*CHANNELS*OUT_DIM + 1*CHANNELS] = out[i*6 + 1];
        if(y_up_pad_pred) output[out_idx + i*CHANNELS*OUT_DIM + 2*CHANNELS]                 = out[i*6 + 2];
        if(y_up_pad_pred) output[out_idx + i*CHANNELS*OUT_DIM + 3*CHANNELS]                 = out[i*6 + 3];
    }

    // WRITE OTHER HALF OF MAIN TILE
    #pragma unroll
    for(int i=2; i<4; i++){ 
        if(x_left_pred) output[out_idx + i*CHANNELS*OUT_DIM]                = out[i*6];
        if(x_left_pred2) output[out_idx + i*CHANNELS*OUT_DIM + 1*CHANNELS]  = out[i*6 + 1];
        output[out_idx + i*CHANNELS*OUT_DIM + 2*CHANNELS]                   = out[i*6 + 2];
        output[out_idx + i*CHANNELS*OUT_DIM + 3*CHANNELS]                   = out[i*6 + 3];
    }
}

template<const int CHANNELS,const int OUT_DIM,const int IN_PAD,const int TILES_1D,const int TILES,const int THREADS_X,const int stride,const int BLOCKS_X,const int BLOCKS_Y,const int BLOCKS_Z>
void btdb_backward(float* d_input, float* d_output, cudaStream_t stream = 0){
    
    dim3 threads(THREADS_X);
    dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

    
    Btdb_back_batch_gpu_new<CHANNELS, OUT_DIM, IN_PAD, TILES_1D, TILES, THREADS_X, stride>
    <<<blocks, threads, 0, stream>>>(d_input,d_output);

}

