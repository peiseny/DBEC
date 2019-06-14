import scipy.misc
import os
import numpy as np
import tensorflow as tf
import pickle as pk
from tensorflow.contrib.layers.python.layers import utils


import logging

from time import strftime, gmtime

import time


slim = tf.contrib.slim

imagesize = 64      # 图片大小
Train_count = 4000  #训练集大小
Test_count = 100    #测试集大小
Classes = 5         #类别数
Channel = 3         #通道数
BATCH_SIZE = 300    # 批次大小
LR = 0.001   # 学习率
EPOCH = 1000  # 迭代次数
LOAD_MODEL = False  # 是否根据保存的模型继续训练
TRAIN = True  # 是否训练完成
FINETUNE = False
#TRAIN = False
HASHING_BITS = 24  # 编码长度
CURRENT_DIR = os.getcwd()  # 获取当前路径

#卷积
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

#全连接
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation

#最大池化
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


# Todo 以shape形状初始化偏置项biases
def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(
                              bias_start, dtype=dtype))
    return var


# Todo 以shape形状初始化权值参数weights
def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(
                              stddev=stddev, dtype=dtype))
    return var


# Todo 全连接层
def fully_connected(value, output_shape, name='fully_connected', with_w=False):
    value = tf.reshape(value, [BATCH_SIZE, -1])
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases
# Todo 卷积层
def conv2d(value, output_dim, k_h=3, k_w=3,
           strides=[1, 1, 1, 1], name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

# Todo 激活函数
def relu(value, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)



# Todo 归一化处理
def lrn(value, depth_radius=1, alpha=5e-05, beta=0.75, name='lrn1'):
    with tf.variable_scope(name):
        norm1 = tf.nn.lrn(value, depth_radius=depth_radius, bias=1.0, alpha=alpha, beta=beta)
        return norm1


# Todo 最大池化层
def pool(value, k_size=[1, 3, 3, 1],
         strides=[1, 2, 2, 1], name='pool1'):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(value, ksize=k_size, strides=strides, padding='VALID')
        return pool


# Todo 平均池化层
def pool_avg(value, k_size=[1, 3, 3, 1],
             strides=[1, 2, 2, 1], name='pool1'):
    with tf.variable_scope(name):
        pool = tf.nn.avg_pool(value, ksize=k_size, strides=strides, padding='VALID')
        return pool


 

# Todo 网络整体结构
def discriminator(image, hashing_bits, reuse=False, name='discriminator'):
    with tf.name_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # 第一层卷积层
        conv1 = conv2d(image, output_dim=32, name='d_conv1')
        relu1 = relu(pool(conv1, name='d_pool1'), name='d_relu1')
        # 第二层卷积层
        conv2 = conv2d(lrn(relu1, name='d_lrn1'), output_dim=32, name='d_conv2')
        relu2 = relu(pool(conv2, name='d_pool2'), name='d_relu2')
        # 第三层卷积层
        conv3 = conv2d(lrn(relu2, name='d_lrn2'), output_dim=64, name='d_conv3')
        relu3 = relu(pool(conv3, name='d_pool3'), name='d_relu3')
        # 第四层卷积层
        conv4 = conv2d(lrn(relu3, name='d_lrn3'), output_dim=128, name='d_conv4')
        relu4 = relu(pool(conv4, name='d_pool4'), name='d_relu4')
        # 第五层卷积层
        conv5 = conv2d(lrn(relu4, name='d_lrn4'), output_dim=256, name='d_conv5')
        #pool5 = pool(relu(conv5, name='d_relu5'), name='d_pool5')
        pool5 = relu(pool(conv5, name='d_relu5'), name='d_pool5')
        # 全连接层
        relu_ip1 = relu(fully_connected(pool5, output_shape=4000, name='d_ip1'), name='d_reluip1')
        # 哈希层
        ip2 = fully_connected(relu_ip1, output_shape=hashing_bits, name='d_ip2')

        return ip2


# Todo 从二进制文件读取数据集信息
def read_juhua_data():
    data_dir = CURRENT_DIR + '/'
   
    train_name = 'juhua/train_batch'
    test_name = 'juhua/test_batch'
    train_X = None  # 训练集数据
    train_Y = None  # 训练集标签
    test_X = None  # 测试集数据
    test_Y = None  # 测试集标签

    # 训练集数据
   
    file_path = train_name
    with open(file_path, 'rb') as fo:
        dict = pk.load(fo)
        train_X = dict['data']
        train_Y = dict['labels']

    # 测试集数据
    # file_path = data_dir + test_name
    file_path = test_name
    with open(file_path, 'rb') as fo:
        dict = pk.load(fo)  # ,encoding='latin1')
        test_X = dict['data']
        test_Y = dict['labels']
    train_X = train_X.reshape((Train_count, Channel, imagesize, imagesize)).transpose(0, 2, 3, 1).astype(np.float)
    # train_Y = train_Y.reshape((4000)).astype(np.float)
    test_X = test_X.reshape((Test_count, Channel, imagesize, imagesize)).transpose(0, 2, 3, 1).astype(np.float)
    

    # 标签信息转换成数组，5列
    train_y_vec = np.zeros((len(train_Y), Classes), dtype=np.float)
    test_y_vec = np.zeros((len(test_Y), Classes), dtype=np.float)
    for i, label in enumerate(train_Y):
        train_y_vec[i, int(train_Y[i])] = 1.  
    for i, label in enumerate(test_Y):
        test_y_vec[i, int(test_Y[i])] = 1.  

    return train_X / 255., train_y_vec, test_X / 255., test_y_vec


# Todo  损失函数
def hashing_loss(image, label, alpha, m):

    D = discriminator(image, HASHING_BITS)
  
    w_label = tf.matmul(label, label, False, True)

    r = tf.reshape(tf.reduce_sum(D * D, 1), [-1, 1])
    
    p2_distance = r - 2 * tf.matmul(D, D, False, True) + tf.transpose(r)
    
    temp = w_label * p2_distance + (1 - w_label) * tf.maximum(m - p2_distance, 0)

    regularizer = tf.reduce_sum(tf.abs(tf.abs(D) - 1))
    
    d_loss = tf.reduce_sum(temp) / (BATCH_SIZE * (BATCH_SIZE - 1)) + alpha * regularizer / BATCH_SIZE
    
    return d_loss

 


# Todo 训练模型
def train():
     
	 
    train_dir = CURRENT_DIR + '/logs/'  # 模型路径
    # 步数
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 设置图片变量和标签变量
    image = tf.placeholder(tf.float32, [BATCH_SIZE, imagesize, imagesize, Channel], name='image')
    label = tf.placeholder(tf.float32, [BATCH_SIZE, Classes], name='label')

   
    # set alpha=0.01
    alpha = tf.constant(0.01, dtype=tf.float32, name='tradeoff')
    # set m = 2*HASHING_BITS
    m = tf.constant(HASHING_BITS * 2, dtype=tf.float32, name='bi_margin')
    # 计算损失
    d_loss_real = hashing_loss(image, label, alpha, m)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]

    # 保存
    saver = tf.train.Saver()
    # 优化损失
    d_optim = tf.train.AdamOptimizer(LR, beta1=0.5).minimize(d_loss_real, var_list=d_vars, global_step=global_step)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.InteractiveSession(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)
 

    start = 0

    # 加载模型
    if LOAD_MODEL:
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(train_dir, ckpt_name))
            # 获取步数
            global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                .split('-')[-1]
            print('Loading success, global_step is %s' % global_step)

        start = int(global_step)

    train_x, train_y, test_x, test_y = read_juhua_data()

    # 循环迭代次数
    for epoch in range(start, EPOCH):

        batch_idxs = int(Train_count / BATCH_SIZE)

        # 按批次取数据训练
        for idx in range(start, batch_idxs):
            image_idx = train_x[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
            label_idx = train_y[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

            # 启动会话
            sess.run([d_optim], feed_dict={image: image_idx, label: label_idx})
            # writer.add_summary(summary_str, idx + 1)

            # 计算loss
            errD_real = d_loss_real.eval(feed_dict={image: image_idx, label: label_idx})
            # 每10批次输出loss
            if (idx + 1) % 10 == 0:

                timestr = strftime("Time :%Y-%m-%d %H:%M:%S")
                logging.info(timestr)
                logging.info("[%3d/%3d][%4d/%4d] d_loss: %.8f" % ( epoch + 1, EPOCH, idx + 1, batch_idxs, errD_real))
                # 每10次迭代保存模型
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(train_dir, 'my_DSH_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
            print('*********    model saved    *********')

            # 关闭会话
    sess.close()


# Todo 转换成二进制编码
def toHashCode(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0 else '1'   ##  mean, median
        list_string_binary.append(str)
        
    return list_string_binary


# Todo 根据训练好的模型计算二进制编码，保存到txt
def evaluate_Codes():
    checkpoint_dir = CURRENT_DIR + '/logs/'

    image = tf.placeholder(tf.float32, [BATCH_SIZE, imagesize, imagesize, Channel], name='image')
    
    D = discriminator(image, HASHING_BITS)
    # res = tf.sign(D)

    # 读取模型
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession(config=config)

    train_x, train_y, test_x, test_y = read_juhua_data()
    file_res = open('result.txt', 'w')
    # sys.stdout = file_res
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Loading success, global_step is %s' % global_step)

        # 测试集进入模型
        for i in range(int(Test_count / BATCH_SIZE)):
            # 得到网络模型输出
            eval_sess = sess.run(D, feed_dict={image: test_x[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]})
            # print(eval_sess)
            # 将输出转换成二进制
            w_res = toHashCode(eval_sess)
            print(1)
            # 得到标签数组 100*1
            w_label = np.argmax(test_y[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], axis=1)
            # 将二进制码和标签写入txt
            for j in range(BATCH_SIZE):
                file_res.write(w_res[j] + '\t' + str(w_label[j]) + '\n')

                # 训练集进入模型
        for i in range(int(Train_count / BATCH_SIZE)):
            # 得到网络模型输出
            eval_sess = sess.run(D, feed_dict={image: train_x[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]})
            # print(eval_sess)
            # 将输出转换成二进制
            w_res = toHashCode(eval_sess)
            # 得到标签数组  40个100*1
            w_label = np.argmax(train_y[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], axis=1)
            # 将二进制码和标签写入txt
            for j in range(BATCH_SIZE):
                file_res.write(w_res[j] + '\t' + str(w_label[j]) + '\n')

                # 关闭文件流
    file_res.close()
    # 关闭会话
    sess.close()





if __name__ == '__main__':

    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/'
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)


    timestr = strftime("Start time :%Y-%m-%d %H:%M:%S")
    logging.info(timestr)
	#训练模型
    TRAIN = True

    if TRAIN:
        start=time.time()
        train()
        end =time.time()
        logging.info("Train time:%f s"%(end - start))
	#计算二进制编码
    else:
        start = time.time()
        evaluate_Codes()
        end = time.time()
        logging.info("Evaluate Codes time:%f s"%(end - start))

    timestr = strftime("End time :%Y-%m-%d %H:%M:%S")
    logging.info(timestr)
