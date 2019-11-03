from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from compact_bilinear_pooling import compact_bilinear_pooling_layer

def bp(bottom1, bottom2, sum_pool=True):
    '''
    Return the fusion of bottom1 and bottom2
    方式：逐元素相乘，维度扩大到dim1*dim2

    Return
    ------
    output: ndarray
        if sum_pool=True, output.shape=(N, dim1*dim2)
        if sum_pool=False, output.shape=(N, H, W, dim1*dim2)
    '''
    assert(np.all(bottom1.shape[:3] == bottom2.shape[:3]))
    batch_size, height, width = bottom1.shape[:3]
    output_dim = bottom1.shape[-1] * bottom2.shape[-1]

    bottom1_flat = bottom1.reshape((-1, bottom1.shape[-1]))
    bottom2_flat = bottom2.reshape((-1, bottom2.shape[-1]))

    output = np.empty((batch_size*height*width, output_dim), np.float32)
    for n in range(len(output)):
        output[n, ...] = np.outer(bottom1_flat[n], bottom2_flat[n]).reshape(-1)
    output = output.reshape((batch_size, height, width, output_dim))

    if sum_pool:
        output = np.sum(output, axis=(1, 2))    # (N, dim1*dim2)
    return output

# Input and output tensors
# Input channels need to be specified for shape inference
input_dim1 = 2048
input_dim2 = 2048
output_dim = 16000
bottom1 = tf.placeholder(tf.float32, [None, None, None, input_dim1])
bottom2 = tf.placeholder(tf.float32, [None, None, None, input_dim2])
top = compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, sum_pool=True)
grad = tf.gradients(top, [bottom1, bottom2])        # grad为top对[bottom1, bottom2]的导数, shape = (N, )

def cbp(bottom1_value, bottom2_value):
    sess = tf.get_default_session()
    return sess.run(top, feed_dict={bottom1: bottom1_value,
                                    bottom2: bottom2_value})

def cbp_with_grad(bottom1_value, bottom2_value):
    sess = tf.get_default_session()
    return sess.run([top]+grad, feed_dict={bottom1: bottom1_value,
                                           bottom2: bottom2_value})

def test_kernel_approximation(batch_size, height, width):
    print("Testing kernel approximation...")

    # Input values
    # 生成四个随机矩阵，(N, H, W, input_dim)
    x = np.random.rand(batch_size, height, width, input_dim1).astype(np.float32)
    y = np.random.rand(batch_size, height, width, input_dim2).astype(np.float32)

    z = np.random.rand(batch_size, height, width, input_dim1).astype(np.float32)
    w = np.random.rand(batch_size, height, width, input_dim2).astype(np.float32)

    # Compact Bilinear Pooling results
    # 计算紧密双线性池化层输出结果
    cbp_xy = cbp(x, y)
    cbp_zw = cbp(z, w)

    # (Original) Bilinear Pooling results
    # 计算普通双线性池化层输出结果
    bp_xy = bp(x, y)
    bp_zw = bp(z, w)

    # Check the kernel results of Compact Bilinear Pooling
    # against Bilinear Pooling
    # 计算两者比值
    cbp_kernel = np.sum(cbp_xy*cbp_zw, axis=1)
    bp_kernel = np.sum(bp_xy*bp_zw, axis=1)
    ratio = cbp_kernel / bp_kernel
    print("ratio between Compact Bilinear Pooling (CBP) and Bilinear Pooling (BP):")
    print(ratio)        # shape = (N, )
    assert(np.all(np.abs(ratio - 1) < 2e-2))
    print("Passed.")

def test_large_input(batch_size, height, width):
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