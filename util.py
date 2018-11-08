import tensorflow as tf
import os


tf_records_train_path='C:/Users/Admin/Desktop/deep_learning _local_datasets/SSD dataset/tf_records/'

def _get_training_data(FLAGS):
    
    filenames=[FLAGS.tf_records_train_path+f for f in os.listdir(FLAGS.tf_records_train_path)]
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(FLAGS.batch_size, padded_shapes=([None,None,3], [None],[None],[None],[None],[None]))
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset
    


def parse(serialized):
    
    
    
    
    features={'image/encoded':tf.FixedLenFeature([], tf.string),
              'image/height':tf.FixedLenFeature([], tf.int64),
              'image/width':tf.FixedLenFeature([], tf.int64),
              'image/channels':tf.FixedLenFeature([], tf.int64),  
                                                 
              'image/object/bbox/xmin':tf.VarLenFeature(tf.float32),
              'image/object/bbox/xmax':tf.VarLenFeature(tf.float32),
              'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
              'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
              'image/object/bbox/label':tf.VarLenFeature(tf.int64),
              }
    
    
    parsed_example=tf.parse_single_example(serialized,
                                           features=features,
                                           )
 
    height = tf.cast(parsed_example['image/height'], tf.int32)
    width = tf.cast(parsed_example['image/width'], tf.int32)  
    num_channels = tf.cast(parsed_example['image/channels'], tf.int32)

    image = tf.cast(tf.decode_raw(parsed_example['image/encoded'], tf.uint8), tf.int32)
    image_reshaped=tf.reshape(image,(height,width,num_channels))
    image=tf.image.resize_images(image_reshaped,(300,300))
    
    x_min=tf.sparse_tensor_to_dense(parsed_example['image/object/bbox/xmin'])
    x_max=tf.sparse_tensor_to_dense(parsed_example['image/object/bbox/xmax'])
    y_min=tf.sparse_tensor_to_dense(parsed_example['image/object/bbox/ymin'])
    y_max=tf.sparse_tensor_to_dense(parsed_example['image/object/bbox/ymax'])
    
    labels=tf.sparse_tensor_to_dense(parsed_example['image/object/bbox/label'])
    
    
    return image, x_min, x_max, y_min, y_max, labels




def _get_initializer():
    return tf.random_normal_initializer(0.0,0.01)
    





