from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from compact_bilinear_pooling import compact_bilinear_pooling_layer

def bp(bottom1, bottom2, sum_pool=True):
    '''
    Return the fusion of bottom1 and bottom2
    方式：逐元素相乘，维度扩大到dim1*dim2

    Return:

    output: ndarray
        output.shape=(N, H, W, dim1*dim2)
    '''
    assert(np.all(bottom1.shape[:3] == bottom2.shape[:3]))
    batch_size, height, width = bottom1.shape[:3]
    output_dim = bottom1.shape[-1] * bottom2.shape[-1]  # output_dim = 2048 x 2048

    bottom1_flat = np.array(bottom1).reshape((-1, 2048))    # 将bottom1变形为(N x H x W, 2048)
    bottom2_flat = np.array(bottom2).reshape((-1, 2048))    # 将bottom2变形为(N x H x W, 2048)

    output = np.empty((batch_size*height*width, output_dim), np.float32) # 生成空矩阵, (N x H x W, 2048*2048)
    for n in range(len(output)):                                         # len(output) = N x H x W
        output[n, ...] = np.outer(bottom1_flat[n], bottom2_flat[n]).reshape(-1) # np.outer()表示output[n,:][i,j] = bottom1_flat[i] * bottom2_flat[j] 
                                                                                # output[n, :].shape=(2048*2048, ); output.shape = (N*H*W, 2048*2048)
    output = output.reshape((batch_size, height, width, output_dim))

    if sum_pool:
        output = np.sum(output, axis=(1, 2))
    return output

# Input and output tensors
# Input channels need to be specified for shape inference
input_dim1 = 2048
input_dim2 = 2048
output_dim = 16000
bottom1 = tf.placeholder(tf.float32, [None, None, None, input_dim1])
bottom2 = tf.placeholder(tf.float32, [None, None, None, input_dim2])

def cbp(bottom1_value, bottom2_value):
    '''Return a 2-factor list containing the gradient of top(bilinear pooling) to bot1 and bot2'''
    sess = tf.InteractiveSession()
    bottom1_value = tf.convert_to_tensor(bottom1_value)
    bottom2_value = tf.convert_to_tensor(bottom2_value)
    bottom1_value = sess.run(bottom1_value)
    bottom2_value = sess.run(bottom2_value)
    top = compact_bilinear_pooling_layer(bottom1_value, bottom2_value, output_dim, sum_pool=True)
    return sess.run(top, feed_dict={bottom1: bottom1_value,
                                    bottom2: bottom2_value})

def cbp_with_grad(bottom1_value, bottom2_value):
    '''Return a 2-factor list containing the derivatives of top+grad to bot1 and bot2'''
    sess = tf.InteractiveSession()
    bottom1_value = tf.convert_to_tensor(bottom1_value)
    bottom2_value = tf.convert_to_tensor(bottom2_value)
    bottom1_value = sess.run(bottom1_value)
    bottom2_value = sess.run(bottom2_value)
    top = compact_bilinear_pooling_layer(bottom1_value, bottom2_value, output_dim, sum_pool=True)
    grad = tf.gradients(top, [bottom1, bottom2])    # 计算top对[bottom1, bottom2]的导数

    return sess.run([top]+grad, feed_dict={bottom1: bottom1_value,
                                           bottom2: bottom2_value})

def test_kernel_approximation(batch_size=2, height=3, width=4):
    print("Testing kernel approximation...")

    # Input values
    # 生成四个随机矩阵，(N, H, W, dim)
    x = np.random.rand(batch_size, height, width, 2048).astype(np.float32)
    y = np.random.rand(batch_size, height, width, 2048).astype(np.float32)

    z = np.random.rand(batch_size, height, width, 2048).astype(np.float32)
    w = np.random.rand(batch_size, height, width, 2048).astype(np.float32)

    # Compact Bilinear Pooling results
    # 计算双线性池化层对x,y,z,w的导数
    cbp_xy = cbp(x, y)
    cbp_zw = cbp(z, w)

    # (Original) Bilinear Pooling results
    # 计算两者相乘得到的矩阵
    bp_xy = bp(x, y)
    bp_zw = bp(z, w)

    # Check the kernel results of Compact Bilinear Pooling
    # against Bilinear Pooling
    cbp_kernel = np.sum(cbp_xy*cbp_zw, axis=1)
    bp_kernel = np.sum(bp_xy*bp_zw, axis=1)
    ratio = cbp_kernel / bp_kernel
    print("ratio between Compact Bilinear Pooling (CBP) and Bilinear Pooling (BP):")
    print(ratio)
    assert(np.all(np.abs(ratio - 1) < 2e-2))
    print("Passed.")

def test_large_input(batch_size=32, height=14, width=14):
    print("Testing large input...")

    # Input values
    x = np.random.rand(batch_size, height, width, input_dim1).astype(np.float32)
    y = np.random.rand(batch_size, height, width, input_dim2).astype(np.float32)

    # Compact Bilinear Pooling results
    _ = cbp_with_grad(x, y)

    # Test passes iff no exception occurs.
    print("Passed.")

def main():
    sess = tf.InteractiveSession()
    test_kernel_approximation(batch_size=2, height=3, width=4)
    test_large_input(batch_size=64, height=14, width=14)
    sess.close()

if __name__ == '__main__':
    main()