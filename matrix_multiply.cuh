#pragma once


struct mm64x64x8_nn_data{ //V_DIM,ITER_DIM,U_DIM,BATCH_SIZE
    const int V_DIM,ITER_DIM,U_DIM,BATCH_SIZE;

    const int V_DIM_0 = 36*BATCH_SIZE;
    const int U_DIM_0 = 36;
    

    const int REG_V_Y = 8;
    const int REG_U_X = 8;

    const int THREADS_Y = 8;
    const int THREADS_X = 8;

    const int V_SIZE = V_DIM_0*ITER_DIM*V_DIM;
    const int U_SIZE = U_DIM_0*ITER_DIM*U_DIM;
    const int OUT_SIZE = V_DIM_0*U_DIM*V_DIM;

    const int SMEM_V_X = 8;
    const int SMEM_V_Y = 64;
    const int SMEM_U_Y = 8; 
    const int SMEM_U_X = 64; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y + 56;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X;
    const int SMEM_SIZE = SMEM_V_SIZE + SMEM_U_SIZE;

    const int BLOCKS_GMEM = (ITER_DIM + SMEM_V_X - 1) / SMEM_V_X;

    const int BLOCKS_X = (V_DIM + SMEM_V_Y - 1 ) / SMEM_V_Y;
    const int BLOCKS_Y = (U_DIM + SMEM_U_X - 1) / SMEM_U_X;
    const int BLOCKS_Z = V_DIM_0;
};

struct mm32x32x8_nn_data{ //V_DIM,ITER_DIM,U_DIM,RELU
    const int V_DIM,ITER_DIM,U_DIM;
    const bool RELU = false;


    const int REG_V_Y = 8;
    const int REG_U_X = 4;

    const int THREADS_X = 32;

    const int V_SIZE = ITER_DIM*V_DIM;
    const int U_SIZE = ITER_DIM*U_DIM;
    const int OUT_SIZE = U_DIM*V_DIM;


    const int SMEM_V_X = 8;
    const int SMEM_V_Y = 32;
    const int SMEM_U_Y = 8; 
    const int SMEM_U_X = 32; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y + 56;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X;

    const int BLOCKS_GMEM = ITER_DIM / SMEM_V_X; //overal number of shared memory tiles 

    const int BLOCKS_X = (V_DIM + SMEM_V_Y - 1) / SMEM_V_Y;
    const int BLOCKS_Y = (U_DIM + SMEM_U_X - 1) / SMEM_U_X;
    const int BLOCKS_Z = 1;
};

struct mm64x64x3_nn_data{ //V_DIM,ITER_DIM,U_DIM,BATCH_SIZE
    const int V_DIM,ITER_DIM,U_DIM,BATCH_SIZE;

    const int V_DIM_0 = 36*BATCH_SIZE;
    const int U_DIM_0 = 36;
    
    const int REG_V_Y = 4;
    const int REG_U_X = 4;

    const int THREADS_X = 256;

    const int V_SIZE = V_DIM_0*ITER_DIM*V_DIM;
    const int U_SIZE = U_DIM_0*ITER_DIM*U_DIM;
    const int OUT_SIZE = V_DIM_0*U_DIM*V_DIM;

    const int SMEM_V_X = 3;
    const int SMEM_V_Y = 64;
    const int SMEM_U_Y = 3; 
    const int SMEM_U_X = 64; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X;

    const int BLOCKS_GMEM = (ITER_DIM + SMEM_V_X - 1) / SMEM_V_X;

    const int BLOCKS_X = (V_DIM + SMEM_V_Y - 1 ) / SMEM_V_Y;
    const int BLOCKS_Y = (U_DIM + SMEM_U_X - 1) / SMEM_U_X;
    const int BLOCKS_Z = V_DIM_0;
};

struct mm32x32x32_nt_data{ //V_DIM,ITER_DIM,U_DIM,BATCH_SIZE
    const int V_DIM, ITER_DIM, U_DIM; // batch_size,filters,iter (32,4096,25088)
    
    const int REG_V_Y = 4;
    const int REG_U_X = 4;

    const int THREADS_X = 64;

    const int V_SIZE = ITER_DIM*V_DIM;
    const int U_SIZE = ITER_DIM*U_DIM;
    const int OUT_SIZE = U_DIM*V_DIM;
    const int DBIAS_SIZE = ITER_DIM;
    const int PR_IDX_SIZE = V_SIZE;

    const int SMEM_V_X = 32;
    const int SMEM_V_Y = 32;
    const int SMEM_U_Y = 32; 
    const int SMEM_U_X = 32; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y + 28; 
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X + 28;

    const int BLOCKS_GMEM = (ITER_DIM+SMEM_V_X-1) / SMEM_V_X; //overal number of shared memory tiles 

    const int BLOCKS_X = (V_DIM+SMEM_V_Y-1)/SMEM_V_Y;
    const int BLOCKS_Y = (U_DIM+SMEM_U_X-1)/SMEM_U_X;
    const int BLOCKS_Z = 1;
};

struct mm128x128x8_tn_data{ //V_DIM,ITER_DIM,U_DIM
    const int V_DIM,ITER_DIM,U_DIM; // iter,batch size,filters (25088,32,4096)
    
    const int NUM = 1;     
    const int DEN = 1;

    const int REG_V_Y = 8;
    const int REG_U_X = 8;

    const int THREADS_X = 256;

    const int V_SIZE = ITER_DIM*V_DIM;
    const int U_SIZE = ITER_DIM*U_DIM;
    const int OUT_SIZE = U_DIM*V_DIM;
    const int PR_IDX_SIZE = U_SIZE;

    const int SMEM_V_X = 8;
    const int SMEM_V_Y = 128;
    const int SMEM_U_Y = 8; 
    const int SMEM_U_X = 128; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X;
    const int SMEM_SIZE = SMEM_U_SIZE + SMEM_V_SIZE;

    const int BLOCKS_GMEM = (ITER_DIM+SMEM_V_X-1) / SMEM_V_X; //overal number of shared memory tiles 

    const int BLOCKS_X = (U_DIM+SMEM_V_Y-1) / SMEM_V_Y;
    const int BLOCKS_Y = (V_DIM+SMEM_U_X-1) / SMEM_U_X;
    const int BLOCKS_Z = 1;
};

struct mm3ch_back_data{ //V_DIM,ITER_DIM,U_DIM,BATCH_SIZE
    const int V_DIM,ITER_DIM,U_DIM,BATCH_SIZE; // channels,tiles,filters

    const int NUM = 1;  
    const int DEN = 1;

    const int REG_V_Y = 3;
    const int REG_U_X = 1;

    const int THREADS_X = 32;

    const int V_SIZE = BATCH_SIZE*36*ITER_DIM*V_DIM;
    const int U_SIZE = BATCH_SIZE*36*ITER_DIM*U_DIM;
    const int OUT_SIZE = 36*U_DIM*V_DIM;

    const int SMEM_V_X = 32;
    const int SMEM_V_Y = 3;
    const int SMEM_U_Y = 32;
    const int SMEM_U_X = 32;

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X;
    const int SMEM_SIZE = SMEM_U_SIZE + SMEM_V_SIZE;

    const int BLOCKS_GMEM = (ITER_DIM*BATCH_SIZE + SMEM_V_X - 1) / SMEM_V_X; //IN THIS KERNEL IT'S LIKE BATCH_SIZE=1 AND ITER_DIM = ITER_DIM*BATCH_SIZE

    const int BLOCKS_X = 1;
    const int BLOCKS_Y = (U_DIM + SMEM_U_X - 1)/SMEM_U_X;
    const int BLOCKS_Z = 36;
};

struct mm_64x64x8_nt_data{ //V_DIM,ITER_DIM,U_DIM,BATCH_SIZE
    const int V_DIM,ITER_DIM,U_DIM,BATCH_SIZE; // tiles,filters,channels

    const int REG_V_Y = 8;
    const int REG_U_X = 8;

    const int THREADS_Y = 8;
    const int THREADS_X = 8;

    const int V_SIZE = 36*ITER_DIM*V_DIM*BATCH_SIZE;
    const int U_SIZE = 36*ITER_DIM*U_DIM;
    const int OUT_SIZE = 36*U_DIM*V_DIM*BATCH_SIZE;


    const int SMEM_V_X = 8;
    const int SMEM_V_Y = 64;
    const int SMEM_U_Y = 8; 
    const int SMEM_U_X = 64; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y + 56;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X + 28;
    const int SMEM_SIZE = SMEM_V_SIZE + SMEM_U_SIZE;

    const int BLOCKS_GMEM = (ITER_DIM + SMEM_V_X -1) / SMEM_V_X; 


    const int BLOCKS_X = (V_DIM + SMEM_V_Y - 1) /64;
    const int BLOCKS_Y = (U_DIM + SMEM_U_X - 1) /64;
    const int BLOCKS_Z = 36*BATCH_SIZE;
};

struct mm_64x64x8_tn_data{ //V_DIM,ITER_DIM,U_DIM,BATCH_SIZE
    const int V_DIM,ITER_DIM,U_DIM,BATCH_SIZE; //channels,tiles,filters

    const int NUM = 1;
    const int DEN = 1;

    const int REG_V_Y = 8;
    const int REG_U_X = 8;

    const int THREADS_Y = 8;
    const int THREADS_X = 8;

    const int V_SIZE = 36*ITER_DIM*V_DIM*BATCH_SIZE;
    const int U_SIZE = 36*ITER_DIM*U_DIM*BATCH_SIZE;
    const int OUT_SIZE = 36*U_DIM*V_DIM;

    const int SMEM_V_X = 8;
    const int SMEM_V_Y = 64;
    const int SMEM_U_Y = 8; 
    const int SMEM_U_X = 64; 

    const int SMEM_V_SIZE = SMEM_V_X * SMEM_V_Y;
    const int SMEM_U_SIZE = SMEM_U_Y * SMEM_U_X;
    const int SMEM_SIZE = SMEM_U_SIZE + SMEM_V_SIZE;

    const int BLOCKS_GMEM = (ITER_DIM*BATCH_SIZE + SMEM_V_X - 1) / SMEM_V_X; //overal number of shared memory tiles 

    const int BLOCKS_X = (V_DIM + SMEM_V_Y - 1) / SMEM_V_Y;
    const int BLOCKS_Y = (U_DIM + SMEM_U_X - 1) / SMEM_U_X;
    const int BLOCKS_Z = 36;
};


namespace mm64x64x8_nn_ns{

    template<const int SMEM_SIZE>
    __inline__ __device__ void swap_pointers_add(float*& p1, float*& p2, int idx){ 
        p1 = p1 + (idx&1)*SMEM_SIZE - (!(idx&1))*SMEM_SIZE;
        p2 = p2 - (idx&1)*SMEM_SIZE + (!(idx&1))*SMEM_SIZE; 
    }

    __inline__ __device__ void swap_pointers(float*& p1, float*& p2){
        float* temp = p1;
        p1 = p2;
        p2 = temp; 
    }

    __inline__ __device__ void mv(int idx, float result_thread[64], float V_reg[8], float U_reg[8]){

        result_thread[idx] += V_reg[idx/8] * U_reg[idx%8];
    }

    template<const int THREADS_X, const int U_DIM, const int ITER_DIM, const int SMEM_V_X, const int SMEM_U_Y>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[8], float U_smem[8],int n_gmem,int U_read_idx,int V_read_idx,int V_in_bound){
        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(U_smem)[i] = reinterpret_cast<const float2*>(U)[U_read_idx + i*THREADS_X + n_gmem*SMEM_U_Y*U_DIM/2];
        }  

        #pragma unroll
        for(int i=0; i<8; i++){
            int V_idx = V_read_idx + i*(ITER_DIM) + n_gmem*SMEM_V_X;
            if( V_idx < V_in_bound )V_smem[i] = V[V_read_idx + i*(ITER_DIM) + n_gmem*SMEM_V_X];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float U_smem[8],float V_smem[8],float* smem){
        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(smem)[SMEM_V_SIZE/2 + threadIdx.y*8 + threadIdx.x + i*64] = reinterpret_cast<float2*>(U_smem)[i];
        }

        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(smem)[threadIdx.x*9 + threadIdx.y + i*71] = reinterpret_cast<float4*>(V_smem)[i];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int local_tid,int n_smem){
        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(V_reg)[i] = reinterpret_cast<float4*>(smem)[n_smem*9 + threadIdx.y + i*71];
        }

        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(U_reg)[i] = reinterpret_cast<float2*>(smem)[SMEM_V_SIZE/2 + (threadIdx.x) + n_smem*8 + i*64];
        }

    }

    //writing the multiplications like this sometimes results in slightly better performance
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        mv(1,result_thread,V_reg,U_reg); 
        mv(0,result_thread,V_reg,U_reg);
        mv(2,result_thread,V_reg,U_reg);
        mv(3,result_thread,V_reg,U_reg);
        mv(5,result_thread,V_reg,U_reg);
        mv(4,result_thread,V_reg,U_reg);
        mv(6,result_thread,V_reg,U_reg);
        mv(7,result_thread,V_reg,U_reg);
        mv(33,result_thread,V_reg,U_reg);
        mv(32,result_thread,V_reg,U_reg);
        mv(34,result_thread,V_reg,U_reg);
        mv(35,result_thread,V_reg,U_reg);
        mv(37,result_thread,V_reg,U_reg);
        mv(36,result_thread,V_reg,U_reg);
        mv(38,result_thread,V_reg,U_reg);
        mv(39,result_thread,V_reg,U_reg);
        mv(45,result_thread,V_reg,U_reg);
        mv(44,result_thread,V_reg,U_reg);
        mv(46,result_thread,V_reg,U_reg);
        mv(47,result_thread,V_reg,U_reg);
        mv(41,result_thread,V_reg,U_reg);
        mv(40,result_thread,V_reg,U_reg);
        mv(42,result_thread,V_reg,U_reg);
        mv(43,result_thread,V_reg,U_reg);
        mv(13,result_thread,V_reg,U_reg);
        mv(12,result_thread,V_reg,U_reg);
        mv(14,result_thread,V_reg,U_reg);
        mv(15,result_thread,V_reg,U_reg);
        mv(9,result_thread,V_reg,U_reg);
        mv(8,result_thread,V_reg,U_reg);
        mv(10,result_thread,V_reg,U_reg);
        mv(11,result_thread,V_reg,U_reg);
        mv(17,result_thread,V_reg,U_reg);
        mv(16,result_thread,V_reg,U_reg);
        mv(18,result_thread,V_reg,U_reg);
        mv(19,result_thread,V_reg,U_reg);
        mv(21,result_thread,V_reg,U_reg);
        mv(20,result_thread,V_reg,U_reg);
        mv(22,result_thread,V_reg,U_reg);
        mv(23,result_thread,V_reg,U_reg);
        mv(49,result_thread,V_reg,U_reg);
        mv(48,result_thread,V_reg,U_reg);
        mv(50,result_thread,V_reg,U_reg);
        mv(51,result_thread,V_reg,U_reg);
        mv(53,result_thread,V_reg,U_reg);
        mv(52,result_thread,V_reg,U_reg);
        mv(54,result_thread,V_reg,U_reg);
        mv(55,result_thread,V_reg,U_reg);
        mv(61,result_thread,V_reg,U_reg);
        mv(60,result_thread,V_reg,U_reg);
        mv(62,result_thread,V_reg,U_reg);
        mv(63,result_thread,V_reg,U_reg);
        mv(57,result_thread,V_reg,U_reg);
        mv(56,result_thread,V_reg,U_reg);
        mv(58,result_thread,V_reg,U_reg);
        mv(59,result_thread,V_reg,U_reg);
        mv(29,result_thread,V_reg,U_reg);
        mv(28,result_thread,V_reg,U_reg);
        mv(30,result_thread,V_reg,U_reg);
        mv(31,result_thread,V_reg,U_reg);
        mv(25,result_thread,V_reg,U_reg);
        mv(24,result_thread,V_reg,U_reg);
        mv(26,result_thread,V_reg,U_reg);
        mv(27,result_thread,V_reg,U_reg);
    }

    template<const int U_DIM, const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void write_output(float* output,float* smem,float* result_thread,int out_idx,int V_out_bound) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/2; j++){
                
                if(out_idx + i*U_DIM/2 + j*8 < V_out_bound) reinterpret_cast<float2*>(output)[out_idx + i*U_DIM/2 + j*8] = reinterpret_cast<float2*>(result_thread)[i*REG_U_X/2 + j]; 
            }
        }

    }


    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_X, const int SMEM_V_Y, const int SMEM_SIZE,
            const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int THREADS_X, const int BLOCKS_GMEM>
    __global__ __launch_bounds__(64,((U_DIM==64)?1:8)) void mm64x64x8_nn(const float* __restrict__ V,const float* __restrict__ U, float* output){ //V 36xBATCH_SIZExTILESxCHANNELS  U 36xCHANNELSxFILTERS  out 36xTILESxFILTERS
        int local_tid = threadIdx.y*blockDim.x + threadIdx.x;
        int U_read_idx = (blockIdx.z/BATCH_SIZE)*ITER_DIM*U_DIM/2 + blockIdx.y*SMEM_U_X/2 + threadIdx.y*U_DIM/2 + threadIdx.x;
        int V_read_idx = blockIdx.z * ITER_DIM*V_DIM + blockIdx.x * SMEM_V_Y*ITER_DIM + threadIdx.y*ITER_DIM*8 + threadIdx.x;
        
        int V_in_bound = (blockIdx.z + 1) * ITER_DIM*V_DIM;
        
        
        __shared__ float smem[SMEM_SIZE];
        __shared__ float smem2[SMEM_SIZE];

        float* smem_comp = smem2;
        float* smem_buf = smem;
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[8];
        float U_smem[8];
        
        load_glob_to_reg<THREADS_X, U_DIM, ITER_DIM, SMEM_V_X, SMEM_U_Y>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx,V_in_bound);
        store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
        __syncthreads();
        smem_buf = &smem2[0];
        smem_comp = &smem[0];


        #pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){
            load_glob_to_reg<THREADS_X, U_DIM, ITER_DIM, SMEM_V_X, SMEM_U_Y>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx,V_in_bound);
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,local_tid,n_smem);
                matmul(result_thread,V_reg,U_reg);
            }
            
            store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
            __syncthreads();
            swap_pointers_add<SMEM_SIZE>(smem_comp,smem_buf,n_gmem);
        }

        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,local_tid,n_smem);

            matmul(result_thread,V_reg,U_reg);
        }
        
        int out_idx = blockIdx.z*V_DIM*U_DIM/2 + blockIdx.y*SMEM_U_X/2 + blockIdx.x*SMEM_V_Y*U_DIM/2 + threadIdx.y*8*U_DIM/2 + threadIdx.x;
        int V_out_bound = (blockIdx.z+1) * V_DIM*U_DIM/2;
        
        write_output<U_DIM, REG_V_Y, REG_U_X>(output,smem,result_thread,out_idx,V_out_bound);
    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_X, 
            const int SMEM_V_Y, const int SMEM_SIZE,const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int THREADS_X, const int BLOCKS_GMEM,
            const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z>
    void mm64x64x8_nn_wrapper(float* d_V, float* d_U, float* d_output, cudaStream_t stream = 0){
        
        dim3 threads(THREADS_X,THREADS_X);
        dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

        
        mm64x64x8_nn<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, SMEM_U_X, SMEM_U_Y, SMEM_V_X, SMEM_V_Y, SMEM_SIZE,SMEM_V_SIZE, REG_V_Y, REG_U_X, THREADS_X, BLOCKS_GMEM>
        <<<blocks, threads, 0, stream>>>(d_V,d_U,d_output);

    }

}

namespace mm32x32x8_nn_ns{

    __inline__ __device__ void swap_pointers(float*& p1, float*& p2){ //same instructions as xor swap
        float* temp = p1;
        p1 = p2;
        p2 = temp; 
    }

    template<const int V_DIM, const int U_DIM, const int ITER_DIM, const int SMEM_U_Y, const int SMEM_V_X, const int SMEM_U_X, const int BLOCKS_Y>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[8], float U_smem[8],int n_gmem,int U_read_idx,int V_read_idx){
        #pragma unroll
        for(int i=0; i<2; i++){
            if(blockIdx.y!=BLOCKS_Y-1 || threadIdx.x%8 < (1 + (U_DIM-1)%SMEM_U_X)/4) reinterpret_cast<float4*>(U_smem)[i] = reinterpret_cast<const float4*>(U)[U_read_idx + i*4*U_DIM/4 + n_gmem*SMEM_U_Y*U_DIM/4];
        }  

        #pragma unroll
        for(int i=0; i<8; i++){
            if((threadIdx.x/8)*8 + i < V_DIM) V_smem[i] = V[V_read_idx + i*(ITER_DIM) + n_gmem*SMEM_V_X];
            
        }
    }

    template<const int V_DIM, const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float U_smem[8],float V_smem[8],float* smem){
        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x/8)*8 + (threadIdx.x%8) + i*8*4] = reinterpret_cast<float4*>(U_smem)[i];
        }

        #pragma unroll
        for(int i=0; i<2; i++){
            if((threadIdx.x/8)*8 + i < V_DIM) reinterpret_cast<float4*>(smem)[(threadIdx.x%8)*5 + (threadIdx.x/8) + i*39] = reinterpret_cast<float4*>(V_smem)[i];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int n_smem){
        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(V_reg)[i] = reinterpret_cast<float4*>(smem)[n_smem*5 + (threadIdx.x/8) + i*39];
        }

        #pragma unroll
        for(int i=0; i<1; i++){
            reinterpret_cast<float4*>(U_reg)[i] = reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x%8) + n_smem*8];
        }

    }

    template<const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        for(int i=0; i<REG_V_Y; i++){
            for(int j=0; j<REG_U_X; j++){
                result_thread[i*REG_U_X + j] += V_reg[i] * U_reg[j];
            }
        }

    }

    template<const int REG_V_Y, const int REG_U_X, const int V_DIM, const int U_DIM, const int SMEM_U_X, const int BLOCKS_Y>
    __inline__ __device__ void write_output(float* output,float* smem,float* result_thread,int out_idx) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/4; j++){
                bool bound_check = (threadIdx.x/8)*8 + i < V_DIM && (blockIdx.y!=BLOCKS_Y-1 || threadIdx.x%8 < (1 + (U_DIM-1)%SMEM_U_X)/4);
                if(bound_check) reinterpret_cast<float4*>(output)[out_idx + i*U_DIM/4] = reinterpret_cast<float4*>(result_thread)[i]; 
            }
        }

    }

    template<const int V_DIM, const int U_DIM, const int REG_V_Y, const int REG_U_X, const int SMEM_U_X, const int BLOCKS_Y, const bool RELU>
    __inline__ __device__ void bias_add_relu(float* result_thread,const float* bias, char* relu_idx, int out_idx) {
        float bias_buf[REG_V_Y];
        int bias_idx = blockIdx.y*SMEM_U_X/4 + threadIdx.x%8;

        if(bias_idx<U_DIM/4) reinterpret_cast<float4*>(bias_buf)[0] = reinterpret_cast<const float4*>(bias)[bias_idx];

        char relu_idx_reg[4];

        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X; j++){
                if(RELU == true){
                    bool pred = result_thread[i*REG_U_X + j] + bias_buf[j]>0;
                    result_thread[i*REG_U_X + j] = (pred) ? result_thread[i*REG_U_X + j] + bias_buf[j] : 0;
                    relu_idx_reg[j] = (pred) ? 1 : 0;
                }
                else{
                    result_thread[i*REG_U_X + j] = result_thread[i*REG_U_X + j] + bias_buf[j];
                }
            }

            if(RELU == true){
                bool bound_check = (threadIdx.x/8)*8 + i < V_DIM && (blockIdx.y!=BLOCKS_Y-1 || threadIdx.x%8 < (1 + (U_DIM-1)%SMEM_U_X)/4);
                if(bound_check) reinterpret_cast<char4*>(relu_idx)[out_idx + i*U_DIM/4] = reinterpret_cast<char4*>(relu_idx_reg)[0];
            }
        }
    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_X, const int SMEM_V_Y,
    const int SMEM_V_SIZE, const int SMEM_U_SIZE, const int REG_V_Y, const int REG_U_X, const int THREADS_X, const int BLOCKS_GMEM, const int BLOCKS_Y, const bool RELU>
    __global__ __launch_bounds__(32,8) void mm32x32x8_nn(const float* __restrict__ V,const float* __restrict__ U, float* __restrict__ output,const float* __restrict__ bias, char* __restrict__ relu_idx){ //V V_DIMxITER_DIM  U ITER_DIM*U_DIM  out V_DIMxU_DIM
        int U_read_idx = blockIdx.y*SMEM_U_X/4 + (threadIdx.x/8)*U_DIM/4 + threadIdx.x%8;
        int V_read_idx = blockIdx.x*SMEM_V_Y*ITER_DIM + (threadIdx.x/8)*ITER_DIM*8 + threadIdx.x%8;
        int out_idx = blockIdx.y*SMEM_U_X/4 + blockIdx.x*SMEM_V_Y*U_DIM/4 + (threadIdx.x/8)*8*U_DIM/4 + threadIdx.x%8;
        

        __shared__ float smem[SMEM_U_SIZE + SMEM_V_SIZE];
        __shared__ float smem2[SMEM_U_SIZE + SMEM_V_SIZE];

        float* smem_comp = smem2;
        float* smem_buf = smem;
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[8];
        float U_smem[8];
        
        load_glob_to_reg<V_DIM, U_DIM, ITER_DIM, SMEM_U_Y, SMEM_V_X, SMEM_U_X, BLOCKS_Y>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx);
        store_reg_to_smem<V_DIM, SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
        
        smem_buf = &smem2[0];
        smem_comp = &smem[0];


        #pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){
            load_glob_to_reg<V_DIM, U_DIM, ITER_DIM, SMEM_U_Y, SMEM_V_X, SMEM_U_X, BLOCKS_Y>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx);
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
            }
            
            store_reg_to_smem<V_DIM, SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
            swap_pointers(smem_comp,smem_buf);
        }

        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);

            matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
        }
        
        bias_add_relu<V_DIM, U_DIM, REG_V_Y, REG_U_X, SMEM_U_X, BLOCKS_Y, RELU>(result_thread,bias,relu_idx, out_idx);

        write_output<REG_V_Y, REG_U_X, V_DIM, U_DIM, SMEM_U_X, BLOCKS_Y>(output,smem,result_thread,out_idx);
    }


    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_X, const int SMEM_V_Y,const int SMEM_V_SIZE, const int SMEM_U_SIZE,
    const int REG_V_Y, const int REG_U_X, const int THREADS_X, const int BLOCKS_GMEM, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int RELU>
    void mm32x32x8_nn_wrapper(float* d_V, float* d_U, float* d_output, float* d_bias, char* d_relu_idx, cudaStream_t stream = 0){
        
        dim3 threads(THREADS_X);
        dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

        
        mm32x32x8_nn<V_DIM, ITER_DIM, U_DIM, SMEM_U_X, SMEM_U_Y, SMEM_V_X, SMEM_V_Y, SMEM_V_SIZE, SMEM_U_SIZE, REG_V_Y, REG_U_X, THREADS_X, BLOCKS_GMEM, BLOCKS_Y,RELU>
        <<<blocks, threads, 0, stream>>>(d_V,d_U,d_output,d_bias,d_relu_idx);

    }
}

namespace mm64x64x3_nn_ns{

    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float* V_smem, float* U_smem,int U_read_idx,int V_read_idx){

        if(threadIdx.x<3*64) *U_smem = U[U_read_idx];
        
        if(threadIdx.x<3*64) *V_smem = V[V_read_idx];
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float* U_smem,float* V_smem,float* smem){

        if(threadIdx.x<3*64) smem[SMEM_V_SIZE + threadIdx.x] = *U_smem;

        if(threadIdx.x<3*64) smem[(threadIdx.x%3)*64 + (threadIdx.x/3)] = *V_smem;
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int n_smem){

        reinterpret_cast<float4*>(V_reg)[0] = reinterpret_cast<float4*>(smem)[n_smem*16 + threadIdx.x/16];
        
        reinterpret_cast<float4*>(U_reg)[0] = reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x%16) + n_smem*16];
    }

    template<const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        for(int i=0; i<REG_V_Y; i++){
            for(int j=0; j<REG_U_X; j++){
                result_thread[i*REG_U_X + j] += V_reg[i] * U_reg[j];
            }
        }

    }

    template<const int U_DIM, const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void write_output(float* output,float* result_thread,int out_idx) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/4; j++){
                reinterpret_cast<float4*>(output)[out_idx + i*(U_DIM/4)] = reinterpret_cast<float4*>(result_thread)[i*REG_U_X/4 + j]; 
            }
        }

    }


    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_V_Y, const int SMEM_U_Y, const int SMEM_V_SIZE, const int SMEM_U_SIZE, const int REG_V_Y, const int REG_U_X>
    __global__ __launch_bounds__(256,2) void mm64x64x3_nn(const float* __restrict__ V,const float* __restrict__ U, float* output){ //V 36xBATCH_SIZExTILESxCHANNELS  U 36xCHANNELSxFILTERS  out 36xTILESxFILTERS
        int U_read_idx = (blockIdx.z/BATCH_SIZE)*ITER_DIM*U_DIM + (threadIdx.x/64)*U_DIM + threadIdx.x%64;
        int V_read_idx = blockIdx.z*ITER_DIM*V_DIM + blockIdx.x * SMEM_V_Y*ITER_DIM + threadIdx.x;
        int out_idx = blockIdx.z*V_DIM*U_DIM/4 + blockIdx.x*SMEM_V_Y*U_DIM/4 + (threadIdx.x/16)*4*(U_DIM/4) + threadIdx.x%16;
        
        __shared__ float smem[SMEM_U_SIZE + SMEM_V_SIZE];
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem;
        float U_smem;
        
        load_glob_to_reg(U,V,&V_smem,&U_smem,U_read_idx,V_read_idx);
        store_reg_to_smem<SMEM_V_SIZE>(&U_smem,&V_smem,smem);
        __syncthreads();
        
        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem,n_smem);
            matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
        }

        write_output<U_DIM, REG_V_Y, REG_U_X>(output,result_thread,out_idx);
    }



    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_V_Y, const int SMEM_U_Y, const int SMEM_V_SIZE, const int SMEM_U_SIZE, const int REG_V_Y, const int REG_U_X,
             const int THREADS_X, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z>
    void mm64x64x3_nn_wrapper(float* d_V, float* d_U, float* d_output, cudaStream_t stream = 0){
        
        dim3 threads(THREADS_X);
        dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);

        
        mm64x64x3_nn<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, SMEM_V_Y, SMEM_U_Y, SMEM_V_SIZE, SMEM_U_SIZE, REG_V_Y, REG_U_X>
        <<<blocks, threads, 0, stream>>>(d_V,d_U,d_output);

    }
}

namespace mm32x32x32_nt_ns{

    __inline__ __device__ void swap_pointers(float*& p1, float*& p2){ 
        float* temp = p1;
        p1 = p2;
        p2 = temp; 
    }

    template<const int V_DIM, const int U_DIM, const int ITER_DIM, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_Y, const int SMEM_V_X>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[16], float U_smem[16],int n_gmem,int U_read_idx,int V_read_idx){
        #pragma unroll
        for(int i=0; i<4; i++){
            if(blockIdx.y*SMEM_U_X + threadIdx.x/8 + i*8 < U_DIM) reinterpret_cast<float4*>(U_smem)[i] = reinterpret_cast<const float4*>(U)[U_read_idx + i*8*ITER_DIM/4 + n_gmem*SMEM_U_Y/4];
        }  

        #pragma unroll
        for(int i=0; i<4; i++){
            if(blockIdx.x*SMEM_V_Y + threadIdx.x/8 + i*8 <V_DIM) reinterpret_cast<float4*>(V_smem)[i] = reinterpret_cast<const float4*>(V)[V_read_idx + (i*8)*ITER_DIM/4 + n_gmem*SMEM_V_X/4];
        }
    }

    template<const int SMEM_V_SIZE, const int SMEM_V_X>
    __inline__ __device__ void store_reg_to_smem(float U_smem[16],float V_smem[16],float* smem,float* dbias,int n_gmem){
        #pragma unroll
        for(int i=0; i<16; i++){
            smem[SMEM_V_SIZE + (threadIdx.x%8)*132 + (threadIdx.x/8) + (i%4)*32 + (i/4)*8] = U_smem[i];
        }

        float dbias_temp[4]={0};
        #pragma unroll
        for(int i=0; i<16; i++){
            smem[ (threadIdx.x%8)*132 + (threadIdx.x/8) + (i%4)*32 + (i/4)*8] = V_smem[i];
            if(blockIdx.y==0) dbias_temp[i%4] += V_smem[i]; // the if protects against adding the same elements many times
        }

        #pragma unroll
        for(int i=0; i<4; i++){
            if(blockIdx.y==0) atomicAdd(&dbias[(threadIdx.x%8)*4 + n_gmem*SMEM_V_X + i],dbias_temp[i]); //computation of bias gradient happens only here and not at the other mm.
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int n_smem){
        #pragma unroll
        for(int i=0; i<1; i++){
            reinterpret_cast<float4*>(V_reg)[i] = reinterpret_cast<float4*>(smem)[n_smem*8 + (threadIdx.x/8) + n_smem/4];
        }

        #pragma unroll
        for(int i=0; i<1; i++){
            reinterpret_cast<float4*>(U_reg)[i] = reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x%8) + n_smem*8 + n_smem/4];
        }

    }

    template<const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        for(int i=0; i<REG_V_Y; i++){
            for(int j=0; j<REG_U_X; j++){
                result_thread[i*REG_U_X + j] += V_reg[i] * U_reg[j];
            }
        }

    }

    template<const int REG_V_Y, const int REG_U_X, const int SMEM_U_X, const int U_DIM, const int OUT_SIZE>
    __inline__ __device__ void write_output(float* output,float* smem,float* result_thread,int out_idx) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/4; j++){
                if((threadIdx.x%8)*4 + blockIdx.y*SMEM_U_X<U_DIM && out_idx + (i)*U_DIM/4 < OUT_SIZE/4) reinterpret_cast<float4*>(output)[out_idx + (i)*U_DIM/4] = reinterpret_cast<float4*>(result_thread)[i];  
            }
        }
    }


    template<const int V_DIM, const int U_DIM, const int ITER_DIM, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_Y, const int SMEM_V_X, 
    const int SMEM_U_SIZE, const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int OUT_SIZE>
    __global__ __launch_bounds__(64,2) void mm32x32x32_nt(const float* __restrict__ V,const float* __restrict__ U, float* output, float* __restrict__ dbias){ //V V_DIMxITER_DIM  U U_DIM*ITER_DIM  out V_DIMxU_DIM  V=DL/DM  U=Ut
        int U_read_idx = blockIdx.y*SMEM_U_X*ITER_DIM/4 + (threadIdx.x/8)*ITER_DIM/4 + threadIdx.x%8; 
        int V_read_idx = blockIdx.x*SMEM_V_Y*ITER_DIM/4 + (threadIdx.x/8)*ITER_DIM/4 + threadIdx.x%8;
        int out_idx = blockIdx.y*SMEM_U_X/4 + blockIdx.x*SMEM_V_Y*U_DIM/4 + (threadIdx.x/8)*4*U_DIM/4 + threadIdx.x%8;
        
        __shared__ float smem[SMEM_U_SIZE + SMEM_V_SIZE];
        __shared__ float smem2[SMEM_U_SIZE + SMEM_V_SIZE];

        float* smem_comp = smem2;
        float* smem_buf = smem;
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[16];
        float U_smem[16];
        
        load_glob_to_reg<V_DIM, U_DIM, ITER_DIM, SMEM_U_X, SMEM_U_Y, SMEM_V_Y, SMEM_V_X>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx);
        store_reg_to_smem<SMEM_V_SIZE, SMEM_V_X>(U_smem,V_smem,smem_buf,dbias,0);
        __syncthreads();
        smem_buf = &smem2[0];
        smem_comp = &smem[0];


        #pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){
            load_glob_to_reg<V_DIM, U_DIM, ITER_DIM, SMEM_U_X, SMEM_U_Y, SMEM_V_Y, SMEM_V_X>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx);
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
            }
            
            store_reg_to_smem<SMEM_V_SIZE, SMEM_V_X>(U_smem,V_smem,smem_buf,dbias,n_gmem);
            __syncthreads();
            swap_pointers(smem_comp,smem_buf);
        }

        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            if(n_smem<1+(ITER_DIM-1)%32){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
            }
        }
        
        write_output<REG_V_Y, REG_U_X, SMEM_U_X, U_DIM, OUT_SIZE>(output,smem,result_thread,out_idx);
    }


    __global__ __launch_bounds__(1024,1) void zero_out_dbias(float* __restrict__ dbias){ //FILTERS/4 threads
        float temp[4] = {0};
        reinterpret_cast<float4*>(dbias)[threadIdx.x] = reinterpret_cast<float4*>(temp)[0];
    }

    template<const int V_DIM, const int U_DIM, const int ITER_DIM, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_U_SIZE,
    const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int OUT_SIZE, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int THREADS_X>
    void mm32x32x32_nt_wrapper(float* d_dldm,float*  d_U, float* d_dldo, float* d_dbias, cudaStream_t stream = 0){

        dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);
        dim3 threads(THREADS_X);

        zero_out_dbias<<<dim3(1,1,1),dim3(ITER_DIM/4),0,stream>>>(d_dbias);


        mm32x32x32_nt<V_DIM, U_DIM, ITER_DIM, SMEM_U_X, SMEM_U_Y, SMEM_V_Y, SMEM_V_X, SMEM_U_SIZE, SMEM_V_SIZE, REG_V_Y, REG_U_X, BLOCKS_GMEM, OUT_SIZE>
        <<<blocks,threads,0,stream>>>(d_dldm,d_U,d_dldo,d_dbias);
    }
}

namespace mm128x128x8_tn_ns{

    __inline__ __device__ void swap_pointers(float*& p1, float*& p2){
        float* temp = p1;
        p1 = p2;
        p2 = temp; 
    }

    template<const int V_DIM, const int U_DIM, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_U_Y, const int SMEM_U_X>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[4], float U_smem[4],int n_gmem,int U_read_idx,int V_read_idx){
        #pragma unroll
        for(int i=0; i<1; i++){
            if(blockIdx.x*SMEM_V_Y/4 + threadIdx.x%32 < U_DIM/4) reinterpret_cast<float4*>(U_smem)[i] = reinterpret_cast<const float4*>(U)[U_read_idx + n_gmem*SMEM_U_Y*U_DIM/4];
        }  

        #pragma unroll
        for(int i=0; i<1; i++){
            if(blockIdx.y*SMEM_U_X/4 + threadIdx.x%32 < V_DIM/4) reinterpret_cast<float4*>(V_smem)[i] = reinterpret_cast<const float4*>(V)[V_read_idx + n_gmem*SMEM_V_X*V_DIM/4];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float U_smem[4],float V_smem[4],float* smem){
        #pragma unroll
        for(int i=0; i<1; i++){
            reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x/32)*32 + (threadIdx.x%32)] = reinterpret_cast<float4*>(U_smem)[i];
        }

        #pragma unroll
        for(int i=0; i<1; i++){
            reinterpret_cast<float4*>(smem)[(threadIdx.x/32)*32 + (threadIdx.x%32)] = reinterpret_cast<float4*>(V_smem)[i];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int n_smem){
        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(V_reg)[i] = reinterpret_cast<float4*>(smem)[n_smem*32 + (threadIdx.x/16)*2 + i];
        }

        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(U_reg)[i] = reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x%16) + n_smem*32 + i*16];
        }

    }

    template<const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        for(int i=0; i<REG_V_Y; i++){
            for(int j=0; j<REG_U_X; j++){
                result_thread[i*REG_U_X + j] += V_reg[i] * U_reg[j];
            }
        }
    }

    template<const int REG_V_Y, const int REG_U_X, const int NUM, const int DEN>
    __inline__ __device__ void mul_result(float* result_thread){
        for(int i=0; i<REG_V_Y*REG_U_X; i++){
            result_thread[i] /= (float)DEN;
        }
    }

    template<const int U_DIM, const int OUT_SIZE, const int SMEM_U_X, const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void write_output(float* output,float* smem,float* result_thread,int out_idx) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/4; j++){
                int filter_idx = blockIdx.x*SMEM_U_X/4 + threadIdx.x%16 + j*16;

                if(filter_idx < U_DIM/4 && out_idx + i*U_DIM/4 + j*16 < OUT_SIZE/4)reinterpret_cast<float4*>(output)[out_idx + i*U_DIM/4 + j*16] = reinterpret_cast<float4*>(result_thread)[i*REG_U_X/4 + j];  
            }
        }

    }


    template<const int V_DIM, const int U_DIM, const int ITER_DIM, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_U_SIZE, const int SMEM_V_SIZE,
    const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int OUT_SIZE, const int NUM, const int DEN>
    __global__ __launch_bounds__(256,1) void mm128x128x8_tn(const float* __restrict__ V,const float* __restrict__ U, float* output){ //V 36xITER_DIMxV_DIM dldm 36xITER_DIMxU_DIM out 36xV_DIMxU_DIM
        int U_read_idx = blockIdx.x*SMEM_V_Y/4 + (threadIdx.x/32)*U_DIM/4 + threadIdx.x%32;
        int V_read_idx = blockIdx.y*SMEM_U_X/4 + (threadIdx.x/32)*V_DIM/4 + threadIdx.x%32;
        int out_idx = blockIdx.x*SMEM_U_X/4 + blockIdx.y*SMEM_V_Y*U_DIM/4 + (threadIdx.x/16)*8*U_DIM/4 + threadIdx.x%16;
        
        __shared__ float smem[SMEM_U_SIZE + SMEM_V_SIZE];
        __shared__ float smem2[SMEM_U_SIZE + SMEM_V_SIZE];

        float* smem_comp = smem2;
        float* smem_buf = smem;
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[4];
        float U_smem[4];
        
        load_glob_to_reg<V_DIM, U_DIM, SMEM_V_Y, SMEM_V_X, SMEM_U_Y, SMEM_U_X>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx);
        store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
        __syncthreads();
        smem_buf = &smem2[0];
        smem_comp = &smem[0];


        #pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){
            
            load_glob_to_reg<V_DIM, U_DIM, SMEM_V_Y, SMEM_V_X, SMEM_U_Y, SMEM_U_X>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx);
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);  
            }
            
            store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
            __syncthreads();
            swap_pointers(smem_comp,smem_buf);
        }


        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            if(n_smem<1+(ITER_DIM-1)%8){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
            }
        }
        
        mul_result<REG_V_Y, REG_U_X, NUM, DEN>(result_thread);

        write_output<U_DIM, OUT_SIZE, SMEM_U_X, REG_V_Y, REG_U_X>(output,smem,result_thread,out_idx);
    }


    template<const int SIZE>
    __global__ __launch_bounds__(128,1) void relu_back(float* __restrict__ d_dldm,const char* __restrict__ d_relu_idx){
        int idx = blockIdx.x*blockDim.x*64/4 + threadIdx.x;
        float buf[64];
        char buf_rel[64];

        #pragma unroll
        for(int i=0; i<64/4; i++){
            if(idx + i*blockDim.x<SIZE/4) reinterpret_cast<float4*>(buf)[i] = reinterpret_cast<float4*>(d_dldm)[idx + i*blockDim.x];
            if(idx + i*blockDim.x<SIZE/4) reinterpret_cast<char4*>(buf_rel)[i] = reinterpret_cast<const char4*>(d_relu_idx)[idx + i*blockDim.x];
        }

        #pragma unroll
        for(int i=0; i<64; i++){
            buf[i] = buf[i] * buf_rel[i];
        }

        #pragma unroll
        for(int i=0; i<64/4; i++){
            if(idx + i*blockDim.x<SIZE/4) reinterpret_cast<float4*>(d_dldm)[idx + i*blockDim.x] = reinterpret_cast<float4*>(buf)[i];
        }
    }

    template<const int V_DIM, const int U_DIM, const int ITER_DIM, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_U_SIZE, const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X,
    const int BLOCKS_GMEM, const int OUT_SIZE, const int THREADS_X, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int NUM, const int DEN>
    void mm128x128x8_tn_wrapper(float* __restrict__ d_V, float* __restrict__ d_dldm, float* d_dldu, cudaStream_t stream = 0){

        dim3 blocks(BLOCKS_X,BLOCKS_Y,BLOCKS_Z);
        dim3 threads(THREADS_X);

        mm128x128x8_tn<V_DIM, U_DIM, ITER_DIM, SMEM_V_Y, SMEM_V_X, SMEM_U_X, SMEM_U_Y, SMEM_U_SIZE, SMEM_V_SIZE, REG_V_Y, REG_U_X, BLOCKS_GMEM, OUT_SIZE, NUM, DEN>
        <<<blocks,threads,0,stream>>>(d_V,d_dldm,d_dldu);
    }

    template<const int U_DIM, const int ITER_DIM>
    void relu_back_wrap(float* __restrict__ d_dldm, char* __restrict__ d_relu_idx, cudaStream_t stream = 0){
        const int elements_per_thread_relu = 64;
        const int threads_rel = 128;
        const int elements_per_threadblock_relu = threads_rel * elements_per_thread_relu;

        dim3 blocks_relu  { (ITER_DIM*U_DIM + elements_per_threadblock_relu -1 )/elements_per_threadblock_relu };
        dim3 threads_relu  {threads_rel};

        relu_back<ITER_DIM*U_DIM><<<blocks_relu,threads_relu,0,stream>>>(d_dldm,d_relu_idx);
    }
}

namespace mm3ch_back_ns{

    template<const int SMEM_SIZE>
    __inline__ __device__ void swap_pointers_add(float*& p1, float*& p2, int idx){ 
        p1 = p1 + (idx&1)*SMEM_SIZE - (!(idx&1))*SMEM_SIZE;
        p2 = p2 - (idx&1)*SMEM_SIZE + (!(idx&1))*SMEM_SIZE; 
    }

    template<const int V_DIM, const int U_DIM>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[3], float U_smem[32], int n_gmem, int U_read_idx,int V_read_idx){
        #pragma unroll
        for(int i=0; i<8; i++){
            reinterpret_cast<float4*>(U_smem)[i] = reinterpret_cast<const float4*>(U)[U_read_idx + i*4*U_DIM/4 + n_gmem*32*U_DIM/4];
        }  

        reinterpret_cast<float3*>(V_smem)[0] = reinterpret_cast<const float3*>(V)[V_read_idx + n_gmem*32*V_DIM/3];
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float U_smem[32],float V_smem[3],float* smem){
        #pragma unroll
        for(int i=0; i<8; i++){
            reinterpret_cast<float4*>(smem)[SMEM_V_SIZE/4 + (threadIdx.x/8)*8 + threadIdx.x%8 + i*32] = reinterpret_cast<float4*>(U_smem)[i];
        }

        reinterpret_cast<float3*>(smem)[threadIdx.x] = reinterpret_cast<float3*>(V_smem)[0];
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int n_smem){
        reinterpret_cast<float3*>(V_reg)[0] = reinterpret_cast<float3*>(smem)[n_smem];
        
        #pragma unroll
        for(int i=0; i<1; i++){
            U_reg[i] = smem[SMEM_V_SIZE + threadIdx.x + n_smem*32];
        }
    }

    template<const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X; j++){
                result_thread[i*REG_U_X + j] += V_reg[i] * U_reg[j];
                //if(blockIdx.z==0 && blockIdx.x==0 && blockIdx.y==0 && i*REG_U_X + j==0 && threadIdx.y*blockDim.x + threadIdx.x==0){
                //    printf ("%f * %f \n",V_reg[i],U_reg[j]);
                //}
            }
        }

    }

    template<const int REG_V_Y, const int REG_U_X, const int NUM, const int DEN>
    __inline__ __device__ void mul_result(float* result_thread){
        #pragma unroll
        for(int i=0; i<REG_V_Y*REG_U_X; i++){
            result_thread[i] /= (float)DEN; 
        }
    }

    template<const int U_DIM, const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void write_output(float* output,float* result_thread,int out_idx) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X; j++){
                output[out_idx + i*U_DIM] = result_thread[i*REG_U_X + j]; 
            }
        }

    }

    // IN THIS KERNEL IT'S LIKE BATCH_SIZE=1 AND ITER_DIM = ITER_DIM*BATCH_SIZE. otherwise we would get an output size of BATCH_SIZE*U_DIM_0*U_DIM*V_DIM. 
    // What we want though is to add up the BATCH_SIZE dimension and average out its result.
    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_SIZE,
    const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int NUM, const int DEN>
    __global__ __launch_bounds__(32,10) void mm3ch_back(const float* __restrict__ V,const float* __restrict__ U, float* output){ //V 36xBATCHxTILESxCHANNELS  dldm 36xBATCHxTILESxFILTERS  out 36xCHANNELSxFILTERS
        int U_read_idx = (blockIdx.z)*ITER_DIM*U_DIM*BATCH_SIZE/4 + blockIdx.y*SMEM_U_X/4 + (threadIdx.x/8)*U_DIM/4 + threadIdx.x%8; //IN THIS KERNEL IT'S LIKE BATCH_SIZE=1 AND ITER_DIM = ITER_DIM*BATCH_SIZE
        int V_read_idx = blockIdx.z*ITER_DIM*V_DIM*BATCH_SIZE/3 + threadIdx.x*V_DIM/3;
        int out_idx = blockIdx.z*V_DIM*U_DIM + blockIdx.y*SMEM_U_X + threadIdx.x;
        
        __shared__ float smem[SMEM_SIZE];
        __shared__ float smem2[SMEM_SIZE];

        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[3];
        float U_smem[32];

        float* smem_comp = smem2;
        float* smem_buf = smem;
        
        load_glob_to_reg<V_DIM, U_DIM>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx);
        store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
        smem_buf = &smem2[0];
        smem_comp = &smem[0];

        //#pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){

            load_glob_to_reg<V_DIM, U_DIM>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx);
            
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
            }
            store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
            swap_pointers_add<SMEM_SIZE>(smem_comp,smem_buf,n_gmem);
        }

        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
            matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
        }
        
        mul_result<REG_V_Y, REG_U_X, NUM, DEN>(result_thread);

        write_output<U_DIM, REG_V_Y, REG_U_X>(output,result_thread,out_idx);
    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_U_X, const int SMEM_U_Y,
    const int SMEM_SIZE, const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int THREADS_X,
    const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int NUM, const int DEN>
    void mm3ch_back_wrapper(float* __restrict__ d_V,float* __restrict__ d_dldm, float* d_dldu, cudaStream_t stream = 0){
        dim3 blocks (BLOCKS_X,BLOCKS_Y,BLOCKS_Z);
        dim3 threads (THREADS_X);

        mm3ch_back<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, SMEM_U_X, SMEM_U_Y, SMEM_SIZE, SMEM_V_SIZE, REG_V_Y, REG_U_X, BLOCKS_GMEM, NUM, DEN>
        <<<blocks,threads,0,stream>>>(d_V,d_dldm,d_dldu);
    }
}

namespace mm_64x64x8_nt_ns{

    template<const int SMEM_SIZE>
    __inline__ __device__ void swap_pointers_add(float*& p1, float*& p2, int idx){ 
    p1 = p1 + (idx&1)*SMEM_SIZE - (!(idx&1))*SMEM_SIZE;
    p2 = p2 - (idx&1)*SMEM_SIZE + (!(idx&1))*SMEM_SIZE; 
    }

    __inline__ __device__ void mv(int idx, float result_thread[64], float V_reg[8], float U_reg[8]){
        result_thread[idx] += V_reg[idx/8] * U_reg[idx%8];
    }

    template<const int ITER_DIM, const int SMEM_V_X>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[8], float U_smem[8],int n_gmem,int U_read_idx,int V_read_idx,int V_in_bound){
        #pragma unroll
        for(int i=0; i<8; i++){
            U_smem[i] = U[U_read_idx + ITER_DIM*(i%2) + (i/2)*16*ITER_DIM + n_gmem*SMEM_V_X];
        }

        #pragma unroll
        for(int i=0; i<8; i++){
            int V_idx = V_read_idx + i*(ITER_DIM) + n_gmem*SMEM_V_X;
            if( V_idx < V_in_bound )V_smem[i] = V[V_read_idx + i*(ITER_DIM) + n_gmem*SMEM_V_X];
        }  
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float U_smem[8],float V_smem[8],float* smem){
        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(smem)[SMEM_V_SIZE/2 + threadIdx.y + threadIdx.x*34 + i*8] = reinterpret_cast<float2*>(U_smem)[i];
        }

        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(smem)[threadIdx.x*9 + threadIdx.y + i*71] = reinterpret_cast<float4*>(V_smem)[i];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int local_tid,int n_smem){
        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(V_reg)[i] = reinterpret_cast<float4*>(smem)[n_smem*9 + threadIdx.y + i*71];
        }

        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(U_reg)[i] = reinterpret_cast<float2*>(smem)[SMEM_V_SIZE/2 + (threadIdx.x) + n_smem*34 + i*8];
        }

    }

    template<const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        for(int i=0; i<REG_V_Y; i++){
            for(int j=0; j<REG_U_X; j++){
                result_thread[i*REG_U_X + j] += V_reg[i] * U_reg[j];
            }
        }
    }

    template<const int U_DIM, const int REG_V_Y, const int REG_U_X>
    __inline__ __device__ void write_output(float* output,float* smem,float* result_thread,int out_idx,int V_out_bound) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/2; j++){
                if(out_idx + i*U_DIM/2 + j*8 < V_out_bound) reinterpret_cast<float2*>(output)[out_idx + i*U_DIM/2 + j*8] = reinterpret_cast<float2*>(result_thread)[i*4 + j]; //same output layout as matmul64x64 
            }
        }

    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_U_Y, const int SMEM_U_X, const int SMEM_SIZE,
    const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM>
    __global__ __launch_bounds__(64,((U_DIM==64)?6:8)) void mm_64x64x8_nt(const float* __restrict__ V,const float* __restrict__ U, float* output){ // dldm 36xBATCHxTILESxFILTERS  U 36xCHANNELSxFILTERS out 36xBATCHxTILESxCHANNELS
        int local_tid = threadIdx.y*blockDim.x + threadIdx.x;
        int U_read_idx = (blockIdx.z/BATCH_SIZE)*ITER_DIM*U_DIM + blockIdx.y*SMEM_U_X*ITER_DIM + threadIdx.y*ITER_DIM*2 + threadIdx.x;
        int V_read_idx = blockIdx.z*ITER_DIM*V_DIM + blockIdx.x*SMEM_V_Y*ITER_DIM + threadIdx.y*ITER_DIM*8 + threadIdx.x;

        int V_in_bound = (blockIdx.z + 1) * ITER_DIM*V_DIM;
        
        
        __shared__ float smem[SMEM_SIZE];
        __shared__ float smem2[SMEM_SIZE];

        float* smem_comp = smem2;
        float* smem_buf = smem;
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[8];
        float U_smem[8];
        
        load_glob_to_reg<ITER_DIM, SMEM_V_X>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx,V_in_bound);
        store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
        __syncthreads();
        smem_buf = &smem2[0];
        smem_comp = &smem[0];


        #pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){

            load_glob_to_reg<ITER_DIM, SMEM_V_X>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx,V_in_bound);
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,local_tid,n_smem);
                matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
            }
            
            store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
            __syncthreads();
            swap_pointers_add<SMEM_SIZE>(smem_comp,smem_buf,n_gmem);
        }

        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,local_tid,n_smem);

            matmul<REG_V_Y, REG_U_X>(result_thread,V_reg,U_reg);
        }
        

        int out_idx = blockIdx.z*V_DIM*U_DIM/2 + blockIdx.y*SMEM_U_X/2 + blockIdx.x*SMEM_V_Y*U_DIM/2 + threadIdx.y*8*U_DIM/2 + threadIdx.x;
        int V_out_bound = (blockIdx.z+1) * V_DIM*U_DIM/2;

        write_output<U_DIM, REG_V_Y, REG_U_X>(output,smem,result_thread,out_idx,V_out_bound);
    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_U_Y, const int SMEM_U_X, const int SMEM_SIZE,
    const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int THREADS_X, const int THREADS_Y, const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z>
    void mm_64x64x8_nt_wrapper(const float* __restrict__ d_dldm,const float* __restrict__ d_U, float* d_dldv, cudaStream_t stream=0){
        dim3 blocks {BLOCKS_X,BLOCKS_Y,BLOCKS_Z};
        dim3 threads {THREADS_X,THREADS_Y};


        mm_64x64x8_nt<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, SMEM_V_Y, SMEM_V_X, SMEM_U_Y, SMEM_U_X, SMEM_SIZE, SMEM_V_SIZE, REG_V_Y, REG_U_X, BLOCKS_GMEM>
        <<<blocks,threads,0,stream>>>(d_dldm,d_U,d_dldv);
    }
}

namespace mm_64x64x8_tn_ns{

    template<const int SMEM_SIZE>
    __inline__ __device__ void swap_pointers_add(float*& p1, float*& p2, int idx){ 
        p1 = p1 + (idx&1)*SMEM_SIZE - (!(idx&1))*SMEM_SIZE;
        p2 = p2 - (idx&1)*SMEM_SIZE + (!(idx&1))*SMEM_SIZE; 
    }

    __inline__ __device__ void mv(int idx, float result_thread[64], float V_reg[8], float U_reg[8]){
        result_thread[idx] += V_reg[idx/8] * U_reg[idx%8];
    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int THREADS_X, const int BLOCKS_GMEM, const int SMEM_U_Y, const int SMEM_V_X>
    __inline__ __device__ void load_glob_to_reg(const float* U,const float* V,float V_smem[8], float U_smem[8],int n_gmem,int U_read_idx,int V_read_idx){

        bool bounds_check = ( (n_gmem != BLOCKS_GMEM-1) || threadIdx.y < 1 + ((ITER_DIM*BATCH_SIZE-1)%8) );

        #pragma unroll
        for(int i=0; i<4; i++){
            if(bounds_check) reinterpret_cast<float2*>(U_smem)[i] = reinterpret_cast<const float2*>(U)[U_read_idx + i*THREADS_X + n_gmem*SMEM_U_Y*U_DIM/2];
        }  

        #pragma unroll
        for(int i=0; i<2; i++){
            if(bounds_check) reinterpret_cast<float4*>(V_smem)[i] = reinterpret_cast<const float4*>(V)[V_read_idx + i*8 + n_gmem*SMEM_V_X*V_DIM/4];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void store_reg_to_smem(float U_smem[8],float V_smem[8],float* smem){
        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(smem)[SMEM_V_SIZE/2 + threadIdx.y*8 + threadIdx.x + i*64] = reinterpret_cast<float2*>(U_smem)[i];
        }

        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(smem)[threadIdx.x + threadIdx.y*8 + i*64] = reinterpret_cast<float4*>(V_smem)[i];
        }
    }

    template<const int SMEM_V_SIZE>
    __inline__ __device__ void load_registers(float* V_reg,float* U_reg,float* smem,int n_smem){
        #pragma unroll
        for(int i=0; i<2; i++){
            reinterpret_cast<float4*>(V_reg)[i] = reinterpret_cast<float4*>(smem)[n_smem*8 + threadIdx.y + i*64];
        }

        #pragma unroll
        for(int i=0; i<4; i++){
            reinterpret_cast<float2*>(U_reg)[i] = reinterpret_cast<float2*>(smem)[SMEM_V_SIZE/2 + (threadIdx.x) + n_smem*8 + i*64];
        }

    }

    //writing the multiplications like this sometimes results in slightly better performance
    __inline__ __device__ void matmul(float* result_thread, float* V_reg, float* U_reg){
        mv(1,result_thread,V_reg,U_reg); 
        mv(0,result_thread,V_reg,U_reg);
        mv(2,result_thread,V_reg,U_reg);
        mv(3,result_thread,V_reg,U_reg);
        mv(5,result_thread,V_reg,U_reg);
        mv(4,result_thread,V_reg,U_reg);
        mv(6,result_thread,V_reg,U_reg);
        mv(7,result_thread,V_reg,U_reg);
        mv(33,result_thread,V_reg,U_reg);
        mv(32,result_thread,V_reg,U_reg);
        mv(34,result_thread,V_reg,U_reg);
        mv(35,result_thread,V_reg,U_reg);
        mv(37,result_thread,V_reg,U_reg);
        mv(36,result_thread,V_reg,U_reg);
        mv(38,result_thread,V_reg,U_reg);
        mv(39,result_thread,V_reg,U_reg);
        mv(45,result_thread,V_reg,U_reg);
        mv(44,result_thread,V_reg,U_reg);
        mv(46,result_thread,V_reg,U_reg);
        mv(47,result_thread,V_reg,U_reg);
        mv(41,result_thread,V_reg,U_reg);
        mv(40,result_thread,V_reg,U_reg);
        mv(42,result_thread,V_reg,U_reg);
        mv(43,result_thread,V_reg,U_reg);
        mv(13,result_thread,V_reg,U_reg);
        mv(12,result_thread,V_reg,U_reg);
        mv(14,result_thread,V_reg,U_reg);
        mv(15,result_thread,V_reg,U_reg);
        mv(9,result_thread,V_reg,U_reg);
        mv(8,result_thread,V_reg,U_reg);
        mv(10,result_thread,V_reg,U_reg);
        mv(11,result_thread,V_reg,U_reg);
        mv(17,result_thread,V_reg,U_reg);
        mv(16,result_thread,V_reg,U_reg);
        mv(18,result_thread,V_reg,U_reg);
        mv(19,result_thread,V_reg,U_reg);
        mv(21,result_thread,V_reg,U_reg);
        mv(20,result_thread,V_reg,U_reg);
        mv(22,result_thread,V_reg,U_reg);
        mv(23,result_thread,V_reg,U_reg);
        mv(49,result_thread,V_reg,U_reg);
        mv(48,result_thread,V_reg,U_reg);
        mv(50,result_thread,V_reg,U_reg);
        mv(51,result_thread,V_reg,U_reg);
        mv(53,result_thread,V_reg,U_reg);
        mv(52,result_thread,V_reg,U_reg);
        mv(54,result_thread,V_reg,U_reg);
        mv(55,result_thread,V_reg,U_reg);
        mv(61,result_thread,V_reg,U_reg);
        mv(60,result_thread,V_reg,U_reg);
        mv(62,result_thread,V_reg,U_reg);
        mv(63,result_thread,V_reg,U_reg);
        mv(57,result_thread,V_reg,U_reg);
        mv(56,result_thread,V_reg,U_reg);
        mv(58,result_thread,V_reg,U_reg);
        mv(59,result_thread,V_reg,U_reg);
        mv(29,result_thread,V_reg,U_reg);
        mv(28,result_thread,V_reg,U_reg);
        mv(30,result_thread,V_reg,U_reg);
        mv(31,result_thread,V_reg,U_reg);
        mv(25,result_thread,V_reg,U_reg);
        mv(24,result_thread,V_reg,U_reg);
        mv(26,result_thread,V_reg,U_reg);
        mv(27,result_thread,V_reg,U_reg);
    }

    template<const int REG_U_X, const int REG_V_Y, const int NUM, const int DEN>
    __inline__ __device__ void mul_result(float* result_thread){
        for(int i=0; i<REG_V_Y*REG_U_X; i++){
            result_thread[i] /=  (float)DEN;
        }
    }

    template<const int U_DIM, const int REG_U_X, const int REG_V_Y>
    __inline__ __device__ void write_output(float* output,float* result_thread,int out_idx) {
        #pragma unroll
        for(int i=0; i<REG_V_Y; i++){
            #pragma unroll
            for(int j=0; j<REG_U_X/2; j++){
                reinterpret_cast<float2*>(output)[out_idx + (i%4)*U_DIM/2 + (i/4)*U_DIM/2*32 + j*8] = reinterpret_cast<float2*>(result_thread)[i*4 + j];  
            }
        }
    }


    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_Y, const int SMEM_V_X, const int SMEM_SIZE, 
    const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int THREADS_X, const int NUM, const int DEN>
    __global__ __launch_bounds__(64,((V_DIM==64)?1:8)) void mm_64x64x8_tn(const float* __restrict__ V,const float* __restrict__ U, float* output){ //V 36xTILESxCHANNELS  dldm 36xTILESxFILTERS  out 36xCHANNELSxFILTERS
        int U_read_idx = (blockIdx.z)*(ITER_DIM)*U_DIM*BATCH_SIZE/2 + blockIdx.y*SMEM_U_X/2 + threadIdx.y*U_DIM/2 + threadIdx.x;
        int V_read_idx = blockIdx.z*V_DIM*ITER_DIM*BATCH_SIZE/4 + blockIdx.x * SMEM_V_Y/4 + threadIdx.y*V_DIM/4 + threadIdx.x;
        int out_idx = blockIdx.z*V_DIM*U_DIM/2 + blockIdx.y*SMEM_U_X/2 + blockIdx.x*SMEM_V_Y*U_DIM/2 + threadIdx.y*U_DIM/2*4 + threadIdx.x;
        
        __shared__ float smem[SMEM_SIZE];
        __shared__ float smem2[SMEM_SIZE];

        float* smem_comp = smem2;
        float* smem_buf = smem;
    
        float result_thread[REG_V_Y*REG_U_X]={0};
        float V_reg[REG_V_Y];
        float U_reg[REG_U_X];

        float V_smem[8];
        float U_smem[8];
        
        load_glob_to_reg<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, THREADS_X, BLOCKS_GMEM, SMEM_U_Y, SMEM_V_X>(U,V,V_smem,U_smem,0,U_read_idx,V_read_idx);
        store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
        __syncthreads();
        smem_buf = &smem2[0];
        smem_comp = &smem[0];


        #pragma unroll 1
        for(int n_gmem=1; n_gmem<BLOCKS_GMEM; n_gmem++){
            
            load_glob_to_reg<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, THREADS_X, BLOCKS_GMEM, SMEM_U_Y, SMEM_V_X>(U,V,V_smem,U_smem,n_gmem,U_read_idx,V_read_idx);
            #pragma unroll
            for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul(result_thread,V_reg,U_reg);  
            }
            
            store_reg_to_smem<SMEM_V_SIZE>(U_smem,V_smem,smem_buf);
            __syncthreads();
            swap_pointers_add<SMEM_SIZE>(smem_comp,smem_buf,n_gmem);
        }


        #pragma unroll
        for(int n_smem=0; n_smem<SMEM_U_Y; n_smem++){
            if(n_smem<1+(ITER_DIM*BATCH_SIZE-1)%8){
                load_registers<SMEM_V_SIZE>(V_reg,U_reg,smem_comp,n_smem);
                matmul(result_thread,V_reg,U_reg);
            }
        }

        mul_result<REG_U_X, REG_V_Y, NUM, DEN>(result_thread);
        
        write_output<U_DIM, REG_U_X, REG_V_Y>(output,result_thread,out_idx);
    }

    template<const int V_DIM, const int ITER_DIM, const int U_DIM, const int BATCH_SIZE, const int SMEM_U_X, const int SMEM_U_Y, const int SMEM_V_Y, const int SMEM_V_X,
    const int SMEM_SIZE, const int SMEM_V_SIZE, const int REG_V_Y, const int REG_U_X, const int BLOCKS_GMEM, const int THREADS_X, const int THREADS_Y, 
    const int BLOCKS_X, const int BLOCKS_Y, const int BLOCKS_Z, const int NUM, const int DEN>
    void mm_64x64x8_tn_wrapper(float* __restrict__ d_V,float* __restrict__ d_dldm, float* d_dldu, cudaStream_t stream=0){
        dim3 blocks {BLOCKS_X,BLOCKS_Y,BLOCKS_Z};
        dim3 threads {THREADS_X,THREADS_Y};


        mm_64x64x8_tn<V_DIM, ITER_DIM, U_DIM, BATCH_SIZE, SMEM_U_X, SMEM_U_Y, SMEM_V_Y, SMEM_V_X, SMEM_SIZE, SMEM_V_SIZE, REG_V_Y, REG_U_X, BLOCKS_GMEM, THREADS_X, NUM, DEN>
        <<<blocks,threads,0,stream>>>(d_V,d_dldm,d_dldu);
    }
}
