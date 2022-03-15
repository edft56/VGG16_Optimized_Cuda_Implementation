# I modified TF's randaugment to work with tf.function.
# I also modified randaugment's logic a bit.


# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.
AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
#import tensorflow as tf
from tensorflow_addons import image as contrib_image
import tensorflow as tf


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

@tf.function
def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.
  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.
  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.
  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1,tf.float32)
  image2 = tf.cast(image2,tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1,tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

@tf.function
def cutout(image, pad_size, replace=0):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.
  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.
  Returns:
    An image Tensor that is of type uint8.
  """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height,dtype=tf.int32)

  cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width,dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims, constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace,
      image)
  return image

@tf.function
def cutout2(image,pad_size,replace):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    
    top_left_pad_corner_y = tf.random.uniform(shape=[], minval=pad_size, maxval=image_height,dtype=tf.int32)
    top_left_pad_corner_x = tf.random.uniform(shape=[], minval=0, maxval=image_width-pad_size,dtype=tf.int32)
    
    upper_pad = image_height - top_left_pad_corner_y
    lower_pad = top_left_pad_corner_y - pad_size
    left_pad = top_left_pad_corner_x
    right_pad = image_width - (top_left_pad_corner_x + pad_size)
    
    mask = tf.zeros([pad_size,pad_size],dtype=tf.uint8)
    mask = tf.pad(mask,[[lower_pad,upper_pad],[left_pad,right_pad]],constant_values=1)
    
    s1 = image[...,0]
    s2 = image[...,1]
    s3 = image[...,2]
    
    s1 = tf.where(tf.equal(mask, 0),tf.ones_like(s1, dtype=image.dtype) * replace[0], s1)
    s2 = tf.where(tf.equal(mask, 0),tf.ones_like(s2, dtype=image.dtype) * replace[1], s2)
    s3 = tf.where(tf.equal(mask, 0),tf.ones_like(s3, dtype=image.dtype) * replace[2], s3)
    
    image = tf.stack([s1, s2, s3], 2)
    return image

@tf.function
def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel (invert the pixel).
  return tf.where(image < threshold, image, 255 - image)

@tf.function
def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int32) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)

@tf.function
def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)

@tf.function
def contrast(image, factor):
  """Equivalent of PIL Contrast."""

  #if tf.shape(image)[2] == 3:
  degenerate = tf.image.rgb_to_grayscale(image)
  #else:
    #degenerate = image

  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)

@tf.function
def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)

@tf.function
def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)

@tf.function
def rotate(image, degrees, replace):
  """Rotates the image by degrees either clockwise or counterclockwise.
  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels caused by
      the rotate operation.
  Returns:
    The rotated version of image.
  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians
    
  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.

  image = contrib_image.rotate(wrap(image), radians,interpolation='bilinear')
  return unwrap(image, replace)

@tf.function
def translate_x(image, pixels, replace):
  """Equivalent of PIL Translate in X dimension."""
  image = contrib_image.translate(wrap(image), [-pixels, 0],interpolation='BILINEAR')
  return unwrap(image, replace)

@tf.function
def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  image = contrib_image.translate(wrap(image), [0, -pixels],interpolation='BILINEAR')
  return unwrap(image, replace)

@tf.function
def shear_x(image, level, replace):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = contrib_image.transform(
      wrap(image), [1., level, 0., 0., 1., 0., 0., 0.],interpolation='BILINEAR')
  return unwrap(image, replace)

@tf.function
def shear_y(image, level, replace):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = contrib_image.transform(
      wrap(image), [1., 0., 0., level, 1., 0., 0., 0.],interpolation='BILINEAR')
  return unwrap(image, replace)

@tf.function
def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D uint8 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image),tf.float32)
    hi = tf.cast(tf.reduce_max(image),tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = (- lo) * scale
      im = tf.cast(im,tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  #if tf.shape(image)[2]==3:
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  #else:
    #image = scale_channel(image[:, :, 0])
  return image

@tf.function
def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.

  #if tf.shape(image)[3] == 3:
  kernel = tf.tile(kernel, [1, 1, 3, 1])

  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)

@tf.function
def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  #if tf.shape(image)[2]==3:
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  #else:
  #  image = scale_channel(image,0)

  return image

@tf.function
def invert(image):
  """Inverts the image pixels."""
  #image = tf.convert_to_tensor(image)
  return 255 - image

@tf.function
def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended

@tf.function
def unwrap(image, replace):
  """Unwraps an image produced by wrap.
  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.
  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.
  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.

  #if image_shape[2]==4:
  alpha_channel = flattened_image[:, 3]
  #else:
  #  alpha_channel = flattened_image[:, 1]

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)
  zero_idx = tf.equal(alpha_channel, 0)
  

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.reshape( zero_idx ,[tf.shape(zero_idx)[0],1]),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], image_shape[2]-1])
  return image

@tf.function
def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.equal(tf.floor(tf.random.uniform([],minval=0,maxval=1) + 0.5), 1)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor

@tf.function
def _rotate_level_to_arg(level):
  #level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return level

@tf.function
def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return 1.0 # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return level

@tf.function
def _enhance_level_to_arg(level):
  return (level/_MAX_LEVEL) * 1.8 + 0.1

@tf.function
def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return level

@tf.function
def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return level






@tf.function
def distort_image_with_randaugment(image, magnitude=10):
    """Applies the RandAugment policy to `image`.
    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    Args:
    image: `Tensor` of shape [height, width, 3] representing an image. #uint8 only
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
    Returns:
    The augmented version of `image`.
    """


    #replace_value = [128] * 3 
    replace_value = [124, 117, 104] #imagenet mean
    cutout_const = tf.random.uniform([],minval=50,maxval=70, dtype=tf.int32)
    translate_const = 100
    max_rot_degrees = 30

    #available_ops = [
    #    'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
    #    'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
    #    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

    color_ops_slight = [0,1,2,3,4,5,6]
    color_ops_strong = [7,8,9,10,11]
    geometric_ops = [12,13,14,15,16]
    
    ops_to_use = tf.TensorArray(dtype=tf.int32, size = 3)
    
    random_pick = tf.floor(tf.random.uniform([],minval=0,maxval=1) + 1.0)
    
    
    ops_to_use = ops_to_use.write( 0, tf.random.uniform([], maxval=len(color_ops_slight), dtype=tf.int32) )
    
    if random_pick==1:
        ops_to_use = ops_to_use.write( 1, tf.random.uniform([], minval=len(color_ops_slight), maxval=len(color_ops_slight) + len(color_ops_strong), dtype=tf.int32) )
    else:
        ops_to_use = ops_to_use.write( 1, 17 ) #no op
        
    ops_to_use = ops_to_use.write( 2 , tf.random.uniform([], minval=len(color_ops_slight) + len(color_ops_strong), maxval=len(color_ops_slight) + len(color_ops_strong) + len(geometric_ops), dtype=tf.int32) )
    
    

    for layer_num in range(3):

        op_to_select = ops_to_use.read(layer_num)

        if op_to_select==0:
            image = autocontrast(image)
        elif op_to_select==1:
            image = equalize(image)
        elif op_to_select==2:
            image = invert(image)
        elif op_to_select==3:
            image = sharpness(image, _enhance_level_to_arg(40))
        elif op_to_select==4:
            image = tf.image.random_jpeg_quality(image,16,20)
        elif op_to_select==5:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif op_to_select==6:
            image = color(image, _enhance_level_to_arg(tf.random.uniform([],minval=25,maxval=50, dtype=tf.float32)))
            
        elif op_to_select==7:
            image = solarize_add(image, int((tf.random.uniform([],minval=6,maxval=12, dtype=tf.float32)/_MAX_LEVEL) * 110))
        elif op_to_select==8:
            image=posterize(image,int((5.0/_MAX_LEVEL)*4))
        elif op_to_select==9:
            image = solarize(image,tf.cast((tf.random.uniform([],minval=3,maxval=6, dtype=tf.float32)/_MAX_LEVEL * 256),tf.uint8))
        elif op_to_select==10:
            image = contrast(image, _enhance_level_to_arg(tf.random.uniform([],minval=6,maxval=12, dtype=tf.float32)))
        elif op_to_select==11:
            image = brightness(image, _enhance_level_to_arg(tf.random.uniform([],minval=6,maxval=12, dtype=tf.float32)))
        
            
            
        elif op_to_select==12:
            image = shear_x(image, _shear_level_to_arg(tf.random.uniform([],minval=4,maxval=12, dtype=tf.float32)), replace_value)
        elif op_to_select==13:
            image = shear_y(image, _shear_level_to_arg(tf.random.uniform([],minval=4,maxval=12, dtype=tf.float32)), replace_value)
        elif op_to_select==14:
            image = translate_x(image, _translate_level_to_arg(tf.random.uniform([],minval=0,maxval=2, dtype=tf.float32), translate_const), replace_value) 
        elif op_to_select==15:
            image = translate_y(image, _translate_level_to_arg(tf.random.uniform([],minval=0,maxval=2, dtype=tf.float32), translate_const), replace_value) 
        elif op_to_select==16:
            image = cutout2(image, cutout_const,replace_value) 


    if tf.equal(tf.floor(tf.random.uniform([],minval=0,maxval=1) + 0.9), 1):
        image = rotate(image, _rotate_level_to_arg(tf.random.uniform([], maxval=max_rot_degrees, dtype=tf.float32)), replace_value)
    
    return image