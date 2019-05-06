"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
 
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
#首先定义了一个pp = pprint.PrettyPrinter()，
#以方便打印数据结构信息
pp = pprint.PrettyPrinter()
 
#[-1]读取倒数第一个元素
#定义了get_stddev函数，
#是三个参数乘积后开平方的倒数，
#应该是为了随机化用 
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])
 
#定义show_all_variables()函数。
#首先，tf.trainable_variables返回的是需要训练的变量列表；
#然后用tensorflow.contrib.slim中的model_analyzer.analyze_vars
#打印出所有与训练相关的变量信息
def show_all_variables():
  model_vars = tf.trainable_variables()
  #用法参见slim_model_analyzer_analyze_vars.py
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)
 
#首先根据图像路径参数读取路径，
#根据灰度化参数选择是否进行灰度化。
#然后对图像参照输入的参数进行裁剪
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)
 
#调用imsave(inverse_transform(images), size, image_path)函数
#并返回新图像
def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)
 
#调用cipy.misc.imread()函数，
#判断grayscale参数是否进行范围灰度化，
#并进行类型转换为np.float
def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)
 
#调用inverse_transform(images)函数，并返回新图像
def merge_images(images, size):
  return inverse_transform(images)
 
def merge(images, size):
  h, w = images.shape[1], images.shape[2]#首先获取image的高和宽
  #然后判断image是RGB图还是灰度图，以分别进行不同的处理
  if (images.shape[3] in (3,4)):#是RGB图
    c = images.shape[3] 
    #size是visualize(sess, dcgan, config, option)函数中得到的
    #如果通道数是3或4，
    #则对每一批次（如，batch_size=64）的所有图像，
    #用0初始化一张原始图像放大8*8的图像
    img = np.zeros((h * size[0], w * size[1], c))
    #大概就是将大小为hxw的image
    #填入到(h * size[0])x(w * size[1])的新图像中
    #并且返回这张大图像
    #因此循环次数是(size[0] x size[1])
    for idx, image in enumerate(images):
      i = idx % size[1]#取余,为啥不是size[0]??
      j = idx // size[1]#整除，取整数部分
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:#是灰度图
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      #如果通道数是1，也是一样，
      #只不过填入图像的时候只填一个通道的信息
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')
 
#首先将merge()函数返回的图像，
#用 np.squeeze()函数移除长度为1的轴。
#然后利用scipy.misc.imsave()函数将新图像保存到指定路径中
def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)
 
#对图像的H和W与crop的H和W相减，得到取整的值，
#根据这个值作为下标依据来scipy.misc.resize图像
def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])
 
#对输入的图像进行裁剪，
#如果crop为true，则使用center_crop()函数，
#对图像的H和W与crop的H和W相减，得到取整的值，
#根据这个值作为下标依据来scipy.misc.resize图像；
 
#否则不对图像进行其他操作，
#直接scipy.misc.resize为64*64大小的图像。
#最后返回图像
def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.#使得像素值[0:255]转换为[-1,1]
 
#对图像进行翻转后返回新图像,像素值[-1,1]变为[0,1]
def inverse_transform(images):
  return (images+1.)/2.
###########################
###########################
#总结下来，这几个函数相互调用，
#主要实现了3个图像操作功能：
#1.获取图像get_image()，负责读取图像，返回图像裁剪后的新图像；
#2.保存图像save_images()，负责将一个batch中所有图像
#保存为一张大图像并返回；
#3.图像翻转merge_images()，负责不知道怎么得翻转的，
#返回新图像。
###########################
###########################
 
#应该是获取每一层的权值、偏置值什么的，
#但貌似代码中没有用到这个函数，所以先不管，后面用到再说
def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]
 
      B = b.eval()
 
      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]
 
      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()
 
        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}
 
      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})
 
        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})
 
        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))
 
#利用moviepy.editor模块来制作动图，为了可视化用的。
#函数又定义了一个函数make_frame(t)，
#首先根据图像集的长度和持续的时间做一个除法，
#然后返回每帧图像。最后视频修剪并制作成GIF动画
def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy
 
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]
 
    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
 
  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)
 
#分为0、1、2、3、4种option。
#如果option=0，则之间显示生产的样本
#如果option=1，根据不同数据集不一样的处理，
#并利用前面的save_images()函数将sample保存下来；
#等等。
#本次在main.py中选用option=1
def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))#（如，batch_size=64）则为64的开方（8）
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]
 
      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1
 
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
 
      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]
 
      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1
 
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
 
      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]
 
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)
 
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]
 
      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))
 
    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
 
#首先获取图像数量的开平方后向下取整的h和向上取整的w，
#然后设置一个assert断言，如果h*w与图像数量相等，则返回h和w，
#否则断言错误提示
def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
 
#这就是全部utils.py全部内容，
#主要负责图像的一些基本操作，
#获取图像、
#保存图像、
#图像翻转，
#和利用moviepy模块可视化训练过程