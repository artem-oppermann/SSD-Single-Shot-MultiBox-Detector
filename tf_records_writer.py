import os
import sys
import tensorflow as tf
import xml.etree.ElementTree as ET
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR='C:/Users/Admin/Desktop/deep_learning _local_datasets/SSD dataset/training/'
OUTPUT_DIR='C:/Users/Admin/Desktop/deep_learning _local_datasets/SSD dataset/tf_records'
    
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = '/JPEGImages/'

SAMPLES_PER_FILES = 200


VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}



def _process_image(directory, name):
   
    # Read the image file.
    filename = directory +DIRECTORY_IMAGES + name + '.jpg'
    #image_data = tf.gfile.FastGFile(filename, 'rb').read()

    image_data= imread(filename)
    
    image_data=image_data.tostring()
   
    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')

    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
        
        
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape):

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    
    for b in bboxes:
        assert len(b) == 4
       
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        

        
    #xmin=np.array(xmin).tostring()
    #ymin=np.asarray(ymin, np.uint8)
    #ymin=np.array(ymin).tostring()
    #xmax=np.asarray(xmax, np.uint8)
    #xmax=np.array(xmax).tostring()
    #ymax=np.asarray(ymax, np.uint8)
    #ymax=np.array(ymax).tostring()
    
    #labels=np.asarray(labels, np.uint8)
    #labels=labels.tostring()
    
    height=shape[0]
    width=shape[1]
    channels=shape[2]

    example = tf.train.Example(features=tf.train.Features(feature={ 'image/encoded': bytes_feature(image_data),
                                                                    'image/height': int64_feature(height),
                                                                    'image/width': int64_feature(width),
                                                                    'image/channels': int64_feature(channels),
                                                                    'image/shape': int64_feature(shape),
                                                                    'image/object/bbox/xmin': float_feature(xmin),
                                                                    'image/object/bbox/xmax': float_feature(xmax),
                                                                    'image/object/bbox/ymin': float_feature(ymin),
                                                                    'image/object/bbox/ymax': float_feature(ymax),
                                                                    'image/object/bbox/label': int64_feature(labels),
                                                                    'image/object/bbox/label_text': bytes_feature(labels_text),}))
    
    
    
    
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, idx, name):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def run(dataset_dir, output_dir):

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)  
    filenames = sorted(os.listdir(path))

    i = 0
    fidx = 1
    
    while i < len(filenames):
       
        tf_filename = _get_output_filename(output_dir, fidx,  name='voc_train')
        
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the Pascal VOC dataset!')
    
    
    
if __name__ == "__main__":
    
    run(dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR)
            
    