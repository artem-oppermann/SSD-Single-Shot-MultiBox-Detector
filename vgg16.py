
import tensorflow as tf
import util

class VGG:
    
    def _init_(self):
        pass
        
    
    def build_network(self, X):
        
        self.initializer=util._get_initializer()
        end_point={}
        
        with tf.variable_scope('vgg16_network'):
            
            with tf.variable_scope('block1'): 
        
                self.conv1_1 = self._conv_layer(X, filters=64, name='conv1_1')
                self.conv1_2 = self._conv_layer(self.conv1_1, filters=64 , name='conv1_2') #300
                self.pool1 = self._max_pool(self.conv1_2,name='pool_1') 
            
            with tf.variable_scope('block2'): 
                
                self.conv2_1 = self._conv_layer(self.pool1, filters=128, name='conv2_1')
                self.conv2_2 = self._conv_layer(self.conv2_1, filters=128, name='conv2_2') #150
                self.pool2 = self._max_pool(self.conv2_2,name='pool_2') #75
                
            with tf.variable_scope('block3'): 
                
                self.conv3_1 = self._conv_layer(self.pool2, filters=256, name='conv3_1')
                self.conv3_2 = self._conv_layer(self.conv3_1, filters=256, name='conv3_2')
                self.conv3_3 = self._conv_layer(self.conv3_2, filters=256, name='conv3_3') #75
                self.pool3 = self._max_pool(self.conv3_3,name='pool_3') #

            with tf.variable_scope('block4'): 
                
                self.conv4_1 = self._conv_layer(self.pool3, filters=512, name='conv4_1') #38
                self.conv4_2 = self._conv_layer(self.conv4_1, filters=512, name='conv4_2')
                self.conv4_3 = self._conv_layer(self.conv4_2, filters=512, name='conv4_3')
                end_point['block4']=self.conv4_3
                self.pool4 = self._max_pool(self.conv4_3,name='pool_4') 

            with tf.variable_scope('block5'): 
                
                self.conv5_1 = self._conv_layer(self.pool4, filters=512, name='conv5_1')
                self.conv5_2 = self._conv_layer(self.conv5_1, filters=512, name='conv5_2')
                self.conv5_3 = self._conv_layer(self.conv5_2, filters=512, name='conv5_3') #19
                self.pool5 = self._max_pool(self.conv5_3, strides=(1, 1), name='pool_5') 
                
            with tf.variable_scope('block6'): 
                
                self.conv6 = self._conv_layer(self.pool5, filters=1024, name='conv6')  #19
            
            with tf.variable_scope('block7'): 
                
                self.conv7 = self._conv_layer(self.conv6, filters=1024, kernel_size=[1, 1], strides=(1, 1),name='conv7') #19
                end_point['block7']=self.conv7
            
            with tf.variable_scope('block8'): 
                
                self.conv8_1 = self._conv_layer(self.conv7, filters=256, kernel_size=[1, 1], strides=(1, 1), #19
                                                name='conv8_1')
                self.conv8_2 = self._conv_layer(self.conv8_1, filters=512, kernel_size=[3, 3], strides=(2, 2), #10
                                                name='conv8_2') 
                end_point['block8']=self.conv8_2
            
            with tf.variable_scope('block9'):
                
                self.conv9_1 = self._conv_layer(self.conv8_2, filters=128, kernel_size=[1, 1], strides=(1, 1), #10
                                        name='conv9_1')
                self.conv9_2 = self._conv_layer(self.conv9_1, filters=256, kernel_size=[3, 3], strides=(2, 2),  #5
                                        name='conv9_2')
                end_point['block9']=self.conv9_2
            
            with tf.variable_scope('block_10'): 
                self.conv10_1 = self._conv_layer(self.conv9_2, filters=128, kernel_size=[1, 1], strides=(1, 1), #5
                                                 name='conv10_1')
                self.conv10_2 = self._conv_layer(self.conv10_1, filters=256, kernel_size=[3, 3], strides=(2, 2), #3
                                                 name='conv10_2')
                end_point['block10']=self.conv10_2
                
            with tf.variable_scope('block11'): 
                
                self.conv11_1 = self._conv_layer(self.conv10_2, filters=128, kernel_size=[1, 1], strides=(1, 1), #3
                                                 name='conv11_1')
                self.conv11_2 = self._conv_layer(self.conv11_1, filters=256, kernel_size=[3, 3], strides=(2, 2), #1
                                                 name='conv11_2', padding='valid')
                end_point['block11']=self.conv11_2
            
            return end_point
                

    def _conv_layer(self, x, filters, name, kernel_size=(3,3), strides=(1,1), padding='same'):
        
        return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, 
                                padding=padding,
                                kernel_initializer=self.initializer,
                                name=name)
    
    def _max_pool(self, x, name, pool_size=(2,2),strides=(2,2),padding='same'):

        return tf.layers.max_pooling2d(x,pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       name=name)
 