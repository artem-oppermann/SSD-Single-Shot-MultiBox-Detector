import tensorflow as tf
from util import _get_training_data
import numpy as np
from scipy.misc import imshow
from collections import namedtuple
from model import Model
from vgg16 import VGG


tf.app.flags.DEFINE_string('tf_records_train_path', 'C:/Users/Admin/Desktop/deep_learning _local_datasets/SSD dataset/tf_records/',
                           'Path of the training data.'
                         )

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.'
                           )
tf.app.flags.DEFINE_integer('batch_size', 1,
                            'Size of the training batch.'
                           )
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          'Learning rate for the optimization'
                          )


tf.app.flags.DEFINE_integer('samples_per_tfrecords_file', 200,
                            'Number of samples in one tf records files.'
                            )
tf.app.flags.DEFINE_integer('number_tfrecords_files', 25,
                            'Number of tfrecords files.'
                            )


tf.app.flags.DEFINE_integer('eval_after_iter', 2,
                            'Evaluate model after number of iterations,')

FLAGS = tf.app.flags.FLAGS



def main(_):
    
    
    num_batches=int((FLAGS.samples_per_tfrecords_file*FLAGS.samples_per_tfrecords_file)/FLAGS.batch_size)
    num_batches=1
    
    with tf.Graph().as_default():
      
        dataset=_get_training_data(FLAGS)
        iterator = dataset.make_initializable_iterator()
        
        image, x_min, x_max, y_min, y_max, labels= iterator.get_next()

        bboxes=tf.stack([y_min,x_min,y_max,x_max],axis=1)
        bboxes=tf.transpose(bboxes,[0,2,1])
        image.set_shape((FLAGS.batch_size,300,300,3))
              
        model=Model(FLAGS)
        
        dboxes=model._build_dboxes()
        glabels, gloc, gscores=model._encode_bboxes(bboxes, labels, dboxes)
        opt=model.optimize(image, glabels, gloc, gscores)
        
        with tf.Session() as sess:
            
            for epoch in range(FLAGS.num_epoch):
                
                sess.run(tf.global_variables_initializer())
                sess.run(iterator.initializer)
                
                loss=0
                loss_pos=0
                loss_neg=0
                loss_loc=0
                
                for batch in range(num_batches):
                    
                    _, loss_, loss_pos_, loss_neg_, loss_loc_=sess.run(opt)
                    
                    loss+=loss_
                    loss_pos+=loss_pos_
                    loss_neg+=loss_neg_
                    loss_loc+=loss_loc_
                    
                    if batch>0 and batch%FLAGS.eval_after_iter==0:
                   
                        print('epoch_nr: %i, batch_nr: %i, loss: %.2f, loss_pos: %.2f, loss_neg: %.2f, loss_loc: %.2f' 
                              %(epoch, batch,(loss/FLAGS.eval_after_iter),(loss_pos/FLAGS.eval_after_iter),
                                (loss_neg/FLAGS.eval_after_iter),(loss_loc/FLAGS.eval_after_iter)))
                        
                        loss=0
                        loss_pos=0
                        loss_neg=0
                        loss_loc=0
         



                    
          
if __name__ == "__main__":
    
    tf.app.run()
            





    