This is an optimized version of vgg16 for Imagenet written in CUDA. Its main purpose is to show how much performance is left on the table when training neural networks.  

btdb file contains the input transforms needed for Winograd Conv and atma contains the output transforms.

The GPU kernels were originally optimized for a gtx 1080. Convolutions are implemented using the Winograd F(4,3) algorithm.  
A detailed explanation is presented in the aptly named pdf which is comprised of the relevant parts of my Master's thesis. There, among other things, I attempt to explain step-by-step how to write optimized matrix-multiply kernels for the GPU. It is quite an interesting and complex process!

A quick summary of performance results for a batch of 32 images is shown below. Some functions like weight update and softmax loss are missing from Nvidia's Cudnn and thus no results for Training are shown.  
However, detailed performance results per layer are shown in the detailed explanation pdf.

| Execution Time (ms)  | Custom | Nvidia Cudnn | Tensorflow |
|----------------------|--------|--------------|------------|
| Inference            |  84ms  |    91.5ms    |    116ms   |
| Training             | 243ms  |      -       |    355ms   |

While cleaning up the code I also tested it on a newer RTX 3080 card. Training times are as fast as training on Tensor Cores using FP16 mixed precision mode on Tensorflow which, to be honest, was quite a surprise. Probably goes to show how latency-bound such workloads are.  

*PS: Pls no judging my messy code. C++ and templates made my life difficult.*