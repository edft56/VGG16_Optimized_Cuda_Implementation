#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/platform/logging.h"
#include "/usr/include/c++/10/bits/unique_ptr.h"



#include "vgg16.h"

using namespace tensorflow;


REGISTER_OP("Vgg16WeightTrans")
    .Input  ("vgg16_weights:            float32")
    
    .Output ("vgg16_trans_weights:      float32");


REGISTER_OP("Vgg16CustomInfer")
    .Input  ("vgg16_image_input:        float32")
    .Input  ("vgg16_weights:            float32")
    .Input  ("vgg16_bias:               float32")
    .Input  ("vgg16_times_to_run:       int32")

    .Output ("vgg16_guesses: int32");


REGISTER_OP("Vgg16CustomTrainNormal")
    .Input  ("vgg16_image_input:        float32")
    .Input  ("vgg16_truth_table:        int8")
    .Input  ("vgg16_weights:            float32")
    .Input  ("vgg16_weight_velocity:    float32")
    .Input  ("vgg16_bias:               float32")
    .Input  ("vgg16_bias_velocity:      float32")
    .Input  ("vgg16_regularization:     float32")
    .Input  ("vgg16_momentum:           float32")
    .Input  ("vgg16_learning_rate:      float32")
    .Input  ("vgg16_times_to_run:       int32")
    .Input  ("vgg16_output_mode:        int32")

    .Output ("vgg16_weights_out:        float32")
    .Output ("vgg16_weight_velocity_out:float32")
    .Output ("vgg16_bias_out:           float32")
    .Output ("vgg16_bias_velocity_out:  float32")
    .Output ("vgg16_loss_out:           float64")
    .Output ("vgg16_accuracy:           float32");




class Vgg16WeightTransOp : public OpKernel {
 public:
  explicit Vgg16WeightTransOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& weights = context->input(0);

    // Create an output tensor
    Tensor* trans_weights = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {WEIGHT_SIZE}, &trans_weights));

    transform_weights_call( (float*)(weights.flat<float>().data()), (float*)(trans_weights->flat<float>().data()) );
  }
};




class Vgg16CustomInferOp : public OpKernel {
 public:
  explicit Vgg16CustomInferOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& image_input = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& bias = context->input(2);
    const Tensor& times_to_run = context->input(3);

    TensorShape guesses_shape ({times_to_run.flat<int>()(0)*BATCH_SIZE});

    // Create an output tensor
    Tensor* guesses = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, guesses_shape , &guesses));

    vgg16_main_infer( 
                      (float*)(image_input.flat<float>().data()),
                      (float*)(weights.flat<float>().data()),
                      (float*)(bias.flat<float>().data()),
                      (int*)(guesses->flat<int>().data()),
                      (times_to_run.flat<int>()(0))
                    );
  }
};



class Vgg16CustomTrainNormalOp : public OpKernel {
 public:
  explicit Vgg16CustomTrainNormalOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& image_input = context->input(0);
    const Tensor& truth_table = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& weight_velocity = context->input(3);
    const Tensor& bias = context->input(4);
    const Tensor& bias_velocity = context->input(5);
    const Tensor& regularization = context->input(6);
    const Tensor& momentum = context->input(7);
    const Tensor& learning_rate = context->input(8);
    const Tensor& times_to_run = context->input(9);
    const Tensor& output_mode = context->input(10);

    TensorShape weight_shape({WEIGHT_SIZE_ORGNL});
    TensorShape bias_shape({BIAS_SIZE});

    auto weights_out = MakeUnique<Tensor>();
    auto weight_velocity_out = MakeUnique<Tensor>();
    auto bias_out = MakeUnique<Tensor>();
    auto bias_velocity_out = MakeUnique<Tensor>();
    CHECK(weights_out->CopyFrom(weights, weight_shape));
    CHECK(weight_velocity_out->CopyFrom(weight_velocity, weight_shape));
    CHECK(bias_out->CopyFrom(bias, bias_shape));
    CHECK(bias_velocity_out->CopyFrom(bias_velocity, bias_shape));

    // Create an output tensor
    Tensor* loss_tensor = NULL;
    Tensor* acc_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, {1}, &loss_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(5, {1}, &acc_tensor));

    vgg16_main_normal( 
                (float*)(image_input.flat<float>().data()),
                (char*)(truth_table.flat<int8>().data()),
                (float*)(weights_out->data()),
                (float*)(weight_velocity_out->data()),
                (float*)(bias_out->data()),
                (float*)(bias_velocity_out->data()),
                (double*)&(loss_tensor->flat<double>()(0)),
                (float*)(momentum.flat<float>().data()),
                (float*)(regularization.flat<float>().data()),
                (float*)(learning_rate.flat<float>().data()),
                (float*)&(acc_tensor->flat<float>()(0)),
                (times_to_run.flat<int>()(0)),
                (output_mode.flat<int>()(0))
              );

    context->set_output(0, *weights_out);
    context->set_output(1, *weight_velocity_out);
    context->set_output(2, *bias_out);
    context->set_output(3, *bias_velocity_out);
  }
};



REGISTER_KERNEL_BUILDER(  Name("Vgg16WeightTrans")
                          .Device(DEVICE_CPU)
                          .HostMemory("vgg16_weights")
                          .HostMemory("vgg16_trans_weights"),
                          Vgg16WeightTransOp
                        );


REGISTER_KERNEL_BUILDER(  Name("Vgg16CustomInfer")
                          .Device(DEVICE_CPU)
                          .HostMemory("vgg16_image_input")
                          .HostMemory("vgg16_weights")
                          .HostMemory("vgg16_bias")
                          .HostMemory("vgg16_guesses")
                          .HostMemory("vgg16_times_to_run"),
                          Vgg16CustomInferOp
                        );


REGISTER_KERNEL_BUILDER(  Name("Vgg16CustomTrainNormal")
                          .Device(DEVICE_CPU)
                          .HostMemory("vgg16_image_input")
                          .HostMemory("vgg16_truth_table")
                          .HostMemory("vgg16_weights")
                          .HostMemory("vgg16_weight_velocity")
                          .HostMemory("vgg16_bias")
                          .HostMemory("vgg16_bias_velocity")
                          .HostMemory("vgg16_regularization")
                          .HostMemory("vgg16_momentum")
                          .HostMemory("vgg16_learning_rate")
                          .HostMemory("vgg16_times_to_run")
                          .HostMemory("vgg16_output_mode")
                          .HostMemory("vgg16_weights_out")
                          .HostMemory("vgg16_weight_velocity_out")
                          .HostMemory("vgg16_bias_out")
                          .HostMemory("vgg16_bias_velocity_out")
                          .HostMemory("vgg16_loss_out"),
                          Vgg16CustomTrainNormalOp
                        );