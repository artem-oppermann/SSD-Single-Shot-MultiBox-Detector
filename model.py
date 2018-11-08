import tensorflow as tf
import numpy as np
from collections import namedtuple
import math
from vgg16 import VGG
import util

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])



class Model:
    
    
    default_params = SSDParams(img_shape=(300, 300),
                               num_classes=21,
                               no_annotation_label=21,
                               feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
                               feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                               anchor_size_bounds=[0.15, 0.90],
                               anchor_sizes=[(21., 45.),
                                             (45., 99.),
                                             (99., 153.),
                                             (153., 207.),
                                             (207., 261.),
                                             (261., 315.)],
                                
                               anchor_ratios=[ [2, .5],
                                               [2, .5, 3, 1./3],
                                               [2, .5, 3, 1./3],
                                               [2, .5, 3, 1./3],
                                               [2, .5],
                                               [2, .5]],
                               anchor_steps=[8, 16, 32, 64, 100, 300],
                               anchor_offset=0.5,
                               normalizations=[20, -1, -1, -1, -1, -1],
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]
                               )
    
    
    
    def __init__(self, FLAGS):
        
        self.FLAGS=FLAGS
        self.params = Model.default_params
        self.initializer=util._get_initializer()
        self.vgg_net=VGG()
    
    def _inference(self, image):

        conv_nets=self.vgg_net.build_network(image)

        logits = []
        predictions=[]
        localisations = []
        
        for i, layer in enumerate(self.params.feat_layers):

            with tf.variable_scope(layer + '_box'):
                
                num_dboxes=len(self.params.anchor_sizes[i])+len(self.params.anchor_ratios[i])
              
                # Location.
                with tf.variable_scope('localisation_prediction'):
                    num_loc_pred = num_dboxes * 4  
                    loc_pred = self._conv_layer(conv_nets[layer], filters=num_loc_pred, name='localisation_conv')
                    loc_pred = tf.reshape(loc_pred, self._get_tensor_shape(loc_pred)[:-1]+[num_dboxes, 4])
 
                # Class prediction.
                with tf.variable_scope('class_prediction'):
                    num_cls_pred = num_dboxes * self.params.num_classes
                    cls_pred =  self._conv_layer(conv_nets[layer], filters=num_cls_pred, name='class_conv')
                    cls_pred = tf.reshape(cls_pred,self._get_tensor_shape(cls_pred)[:-1]+[num_dboxes, self.params.num_classes])
          
            logits.append(cls_pred)
            localisations.append(loc_pred)
            predictions.append(tf.nn.softmax(cls_pred))
           
        return logits, localisations, predictions
    
    
    
    
    def optimize(self, image, glabels, gloc, gscores):
        
        with tf.variable_scope('inference'):
            logits, loc, _=self._inference(image)
             
        with tf.name_scope('ssd_loss'):
            loss,loss_pos,loss_neg,loss_loc=self._calculate_loss(logits, loc, glabels, gloc, gscores)
            
        return tf.train.RMSPropOptimizer(self.FLAGS.learning_rate).minimize(loss),loss,loss_pos,loss_neg,loss_loc
    
    
    def _calculate_loss(self, logits, loc,  glabels, gloc, gscores, match_threshold=0.5, negative_ratio=3.0, alpha=1.0):
        
        flat_logits = []
        flat_loc =[]
        
        flat_glabels=[]
        flat_gloc=[]
        flat_gscores=[]

        for i in range(len(logits)):
            
            flat_logits.append(tf.reshape(logits[i], [-1, self.params.num_classes]))
            flat_loc.append(tf.reshape(loc[i], [-1, 4]))

            flat_glabels.append(tf.reshape(glabels[i], [-1]))
            flat_gloc.append(tf.reshape(gloc[i], [-1, 4]))
            flat_gscores.append(tf.reshape(gscores[i], [-1]))
            
        logits = tf.concat(flat_logits, axis=0) 
        loc = tf.concat(flat_loc, axis=0) 
        
        glabels=tf.concat(flat_glabels, axis=0) 
        gloc=tf.concat(flat_gloc, axis=0) 
        gscores=tf.concat(flat_gscores, axis=0) 
        
        dtype = logits.dtype
        
        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)
        
        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = tf.nn.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,predictions[:, 0],1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + self.FLAGS.batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)
        
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=glabels)
            loss_pos = tf.div(tf.reduce_sum(loss * fpmask), self.FLAGS.batch_size, name='value')

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=no_classes)
            loss_neg = tf.div(tf.reduce_sum(loss * fnmask), self.FLAGS.batch_size, name='value')
            
        with tf.name_scope('localization'):
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = self.smooth_l1_loss(loc - gloc)
            loss_loc = tf.div(tf.reduce_sum(loss * weights), self.FLAGS.batch_size, name='value')
        
        loss=loss_pos+loss_neg+loss_loc
        
        return loss,loss_pos,loss_neg,loss_loc
            

        
    
    def _conv_layer(self, x, filters, name, kernel_size=(3,3), strides=(1,1), padding='same'):
        
        return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, 
                                padding=padding,
                                kernel_initializer=self.initializer,
                                name=name,
                                data_format='channels_last',)
        
    
    def smooth_l1_loss(self, x):
        square_loss   = 0.5*x**2
        absolute_loss = tf.abs(x)
        return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)
    
    def _get_tensor_shape(self, x):
        return x.get_shape().as_list()
    
    
    
    def _build_dboxes(self):
        
        layers_anchors=[]
        
        for i, feat_layer in enumerate(self.params.feat_shapes):

            anchor_bboxes=self._build_dboxes_per_layer(self.params.img_shape, 
                                                       feat_layer,
                                                       self.params.anchor_sizes[i],
                                                       self.params.anchor_ratios[i],
                                                       self.params.anchor_steps[i],)
            
            layers_anchors.append(anchor_bboxes)
            
        return layers_anchors
    

    def _build_dboxes_per_layer(self, img_shape,feat_layer,sizes,ratios,step,offset=0.5,dtype=np.float32):
   
        y, x = np.mgrid[0:feat_layer[0], 0:feat_layer[1]]
        y = (y.astype(dtype=np.float32) + offset) / feat_layer[0]
        x = (x.astype(dtype=np.float32) + offset) / feat_layer[0]
    
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)
    
        num_anchors = len(sizes) + len(ratios)
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        
        di = 1
        if len(sizes) > 1:
            h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
            w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
            di += 1
        for i, r in enumerate(ratios):
            h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
            
        return y, x, h, w
    
    
    def _encode_bboxes(self, bboxes_, labels_, dboxes):
        
        target_labels_batch = []
        target_localization_batch = []
        target_score_batch = []
        
        for i, layer in enumerate(dboxes):
            
             target_labels=[]
             target_localizations=[]
             target_scores=[]
            
             for batch_nr in range(self.FLAGS.batch_size):
                 
                 bboxes=bboxes_[batch_nr]
                 labels=labels_[batch_nr]
            
                 t_labels, t_loc, t_scores = self._encode_per_layer(labels, bboxes, layer)
                 
                 target_labels.append(t_labels)
                 target_localizations.append(t_loc)
                 target_scores.append(t_scores)
             
          
             target_labels=tf.stack([target_labels[i] for i in range(self.FLAGS.batch_size)])
             target_localizations=tf.stack([target_localizations[i] for i in range(self.FLAGS.batch_size)])
             target_scores=tf.stack([target_scores[i] for i in range(self.FLAGS.batch_size)])
             
             target_labels_batch.append(target_labels)
             target_localization_batch.append(target_localizations)
             target_score_batch.append(target_scores)
                
        return target_labels_batch, target_localization_batch, target_score_batch
    
    
    
    def _encode_per_layer(self, labels, bboxes, layer):
    
    
        def jaccard_with_anchors(bbox):
    
            int_ymin = tf.maximum(y_min, bbox[0])
            int_xmin = tf.maximum(x_min, bbox[1])
            int_ymax = tf.minimum(y_max, bbox[2])
            int_xmax = tf.minimum(x_max, bbox[3])
            
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            
            inter_vol = h * w
            union_vol = vol_anchors - inter_vol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard
        
        def condition(i, feat_labels, feat_scores,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            
            r = tf.less(i, tf.shape(labels))
            return r[0]
        
        def body(i, feat_labels, feat_scores,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            
            label = labels[i]
            bbox = bboxes[i]
            jaccard = jaccard_with_anchors(bbox)
            
            mask = tf.greater(jaccard, feat_scores)
            mask = tf.logical_and(mask, feat_scores > -0.5)
            mask = tf.logical_and(mask, label < self.params.num_classes)
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, dtype=tf.float32)
            
            feat_labels = imask * label + (1 - imask) * feat_labels
            feat_scores = tf.where(mask, jaccard, feat_scores)
    
            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
                                    
            return [i+1, feat_labels, feat_scores,
                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]
            
        
        # min and max coordinates of the default boxes
        y, x, h, w = layer
        y_min = y - h / 2.
        x_min = x - w / 2.
        y_max = y + h / 2.
        x_max = x + w / 2.
        
        
        vol_anchors = (x_max - x_min) * (y_max - y_min)
    
        shape = (y.shape[0], y.shape[1], h.size)
        
        feat_labels = tf.zeros(shape, dtype=tf.int64)
        feat_scores = tf.zeros(shape, dtype=tf.float32)
    
        feat_ymin = tf.zeros(shape, dtype=tf.float32)
        feat_xmin = tf.zeros(shape, dtype=tf.float32)
        feat_ymax = tf.ones(shape, dtype=tf.float32)
        feat_xmax = tf.ones(shape, dtype=tf.float32)
    
       
        i = 0
        [i, feat_labels, feat_scores, 
         feat_ymin, feat_xmin,feat_ymax, feat_xmax] = tf.while_loop(condition, body,[i, feat_labels, feat_scores,feat_ymin, feat_xmin,feat_ymax, feat_xmax],
         shape_invariants=[tf.TensorShape([]),tf.TensorShape([None, None,None]),
                           tf.TensorShape([None,None,None]),
                           tf.TensorShape([None,None,None]),
                           tf.TensorShape([None,None,None]),
                           tf.TensorShape([None,None,None]),
                           tf.TensorShape([None,None,None])])
                
                
        # Transform to center / size.
        feat_cy = (feat_ymax + feat_ymin) / 2.
        feat_cx = (feat_xmax + feat_xmin) / 2.
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin
            
        # Encode features.
        feat_cy = (feat_cy - y) / h / self.params.prior_scaling[0]
        feat_cx = (feat_cx - x) / w / self.params.prior_scaling[1]
        feat_h = tf.log(feat_h / h) / self.params.prior_scaling[2]
        feat_w = tf.log(feat_w / w) / self.params.prior_scaling[3]
    
        feat_loc = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
           
        return feat_labels, feat_loc, feat_scores
        


  