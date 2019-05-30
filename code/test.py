import os
import time
import random
import numpy as np
import PIL.Image as Image

#gpunumber = 0
#os.environ["CUDA_VISIBLE_DEVICES"]= str(gpunumber)
import tensorflow as tf
import dataset
import model

from coviar import get_num_frames
from coviar import load
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim import nets
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

flags = tf.app.flags

# Loading path
#flags.DEFINE_string('dataset', 'ucf101', 'name of the dataset')
flags.DEFINE_string('dataset', 'hmdb51', 'name of the dataset')

flags.DEFINE_string('valid_list', 'data/datalists/hmdb51_split1_test.txt', 'valid list')
flags.DEFINE_string('data_path', 'data/hmdb51/mpeg4_videos', 'data path')


flags.DEFINE_string('model_path','my_model/coviar/all_net.chkp-1000', 'restore model path')

flags.DEFINE_integer('num_segments', 3, 'segments in each videos')


FLAGS = flags.FLAGS

if (FLAGS.dataset == 'ucf101'):
    N_CLASS = 101
elif (FLAGS.dataset == 'hmdb51'):
    N_CLASS = 51
else:
    raise ValueError('Unknown dataset ')


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def main(_):

    with tf.name_scope('input_placeholder'):
        mv_placeholder = tf.placeholder(tf.float32, 
                    shape=(None, FLAGS.num_segments, 224, 224, 3 ), name = 'mv_frame')
        i_placeholder = tf.placeholder(tf.float32,
                    shape=(None, FLAGS.num_segments, 224, 224, 3 ), name = 'i_frame')
        r_placeholder = tf.placeholder(tf.float32,
                    shape=(None, FLAGS.num_segments, 224, 224, 3 ), name = 'r_frame')

    with tf.name_scope('label_placeholder'):
        label_placeholder = tf.placeholder(tf.int32, shape=(None), name = 'labels')

    with tf.name_scope('accuracy'):
        combine_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        i_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        mv_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        r_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        tf.summary.scalar('combine_acc', combine_value_)
        tf.summary.scalar('i_acc', i_value_)
        tf.summary.scalar('mv_acc', mv_value_)
        tf.summary.scalar('r_acc', r_value_)
        
    print('Finish placeholder.')


    with tf.name_scope('flatten_input'):
        b_size = tf.shape(mv_placeholder)[0]
        flat_mv = tf.reshape(mv_placeholder, [b_size * FLAGS.num_segments, 224, 224, 3]) # Since we have mulitple segments in a single video
        flat_i = tf.reshape(i_placeholder, [b_size * FLAGS.num_segments, 224, 224, 3])
        flat_r = tf.reshape(r_placeholder, [b_size * FLAGS.num_segments, 224, 224, 3])

    with tf.variable_scope('fc_var') as var_scope:
        mv_weights = {
            'w1': _variable_with_weight_decay('wmv1', [2048 , 512 ], 0.0005),
            'w2': _variable_with_weight_decay('wmv2', [512 , N_CLASS], 0.0005)
        }
        mv_biases = {
            'b1': _variable_with_weight_decay('bmv1', [ 512 ], 0.00),
            'b2': _variable_with_weight_decay('bmv2', [ N_CLASS ], 0.00)
        }
        i_weights = {
            'w1': _variable_with_weight_decay('wi1', [2048 , 512 ], 0.0005),
            'w2': _variable_with_weight_decay('wi2', [512 , N_CLASS], 0.0005)
        }
        i_biases = {
            'b1': _variable_with_weight_decay('bi1', [ 512 ], 0.00),
            'b2': _variable_with_weight_decay('bi2', [ N_CLASS ], 0.00)
        }
        r_weights = {
            'w1': _variable_with_weight_decay('wr1', [2048 , 512 ], 0.0005),
            'w2': _variable_with_weight_decay('wr2', [512 , N_CLASS], 0.0005)
        }
        r_biases = {
            'b1': _variable_with_weight_decay('br1', [ 512 ], 0.00),
            'b2': _variable_with_weight_decay('br2', [ N_CLASS ], 0.00)
        }

    with tf.variable_scope('fusion_var'):
        fusion = tf.get_variable('fusion', [3], initializer=tf.contrib.layers.xavier_initializer())
    
    print('Finish Flatten.')
    
    with tf.device('/gpu:0'):

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            i_feature, _ = resnet_v1.resnet_v1_152(flat_mv, num_classes=None, is_training=True, scope='i_resnet')
            mv_feature, _ = resnet_v1.resnet_v1_50(flat_i, num_classes=None, is_training=True, scope='mv_resnet')
            r_feature, _ = resnet_v1.resnet_v1_50(flat_r, num_classes=None, is_training=True, scope='r_resnet')


        with tf.name_scope('reshape_feature'):
            i_feature = tf.reshape(i_feature, [-1, 2048])
            mv_feature = tf.reshape(mv_feature, [-1, 2048])
            r_feature = tf.reshape(r_feature, [-1, 2048])


        with tf.name_scope('inference_model'):

            i_sc, i_pred = model.inference_feature (i_feature, i_weights, i_biases,
                                                      FLAGS.num_segments, N_CLASS, name = 'i_inf')

            mv_sc, mv_pred = model.inference_feature (mv_feature, mv_weights, mv_biases,
                                                      FLAGS.num_segments, N_CLASS, name = 'mv_inf')

            r_sc, r_pred = model.inference_feature (r_feature, r_weights, r_biases,
                                                      FLAGS.num_segments, N_CLASS, name = 'r_inf')

            _, pred_class = model.inference_fusion ( i_sc, mv_sc, r_sc, fusion)

    print('Finish Model.')
    
    
    with tf.name_scope('saver'):
        restore_var = [v for v in tf.trainable_variables() if ('Adam' not in v.name)]
        my_saver = tf.train.Saver(var_list=restore_var, max_to_keep=5)
    
    with tf.name_scope('init_function'):
        init_var = tf.global_variables_initializer()

    with tf.name_scope('video_dataset'):
        test_data = dataset.buildTestDataset(FLAGS.valid_list, FLAGS.data_path, FLAGS.num_segments, 
                                             batch_size = 1, num_threads = 2, buffer = 30)
        
        with tf.name_scope('dataset_iterator'):

            it_test = tf.contrib.data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
            next_test_data = it_test.get_next()
            init_test_data = it_test.make_initializer(test_data)
            
            
    print('Finish Dataset.')
    
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=config)
    

    with tf.name_scope('intialization'):
        sess.run(init_var)
        my_saver.restore(sess, FLAGS.model_path)

    # Testing

    combine_acc = 0
    i_acc = 0
    mv_acc = 0
    r_acc = 0

    combine_classes = []
    mv_classes = []
    i_classes = []
    r_classes = []
    gt_label = []
    sess.run(init_test_data)

    for i in range(1530):
        
        ti_arr, tmv_arr, tr_arr, tlabel = sess.run(next_test_data)
        i_class, mv_class, r_class, com_class = sess.run([i_pred, mv_pred, r_pred, pred_class], 
                            feed_dict={mv_placeholder: tmv_arr, i_placeholder: ti_arr,
                                        r_placeholder: tr_arr , label_placeholder : tlabel })
        combine_classes = np.append(combine_classes, com_class)
        mv_classes = np.append(mv_classes, mv_class)
        i_classes = np.append(i_classes, i_class)
        r_classes = np.append(r_classes, r_class)
        gt_label = np.append(gt_label, tlabel)
        print("Now predicting: ", i, "Label: ",tlabel, "Predictions: ", i_class, mv_class, r_class, com_class)
    
    combine_acc = np.sum((combine_classes == gt_label)) / gt_label.size
    i_acc = np.sum((i_classes == gt_label)) / gt_label.size
    mv_acc = np.sum((mv_classes == gt_label)) / gt_label.size
    r_acc = np.sum((r_classes == gt_label)) / gt_label.size

    print('Finished with accuracy: %f , %f , %f, %f' % ( i_acc, mv_acc, r_acc, combine_acc))
        
        
       
    #writer.close()


if __name__ == '__main__':
    tf.app.run()