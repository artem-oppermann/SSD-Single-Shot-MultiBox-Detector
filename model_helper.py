def _build_dboxes():
        
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
    
    
    def _encode_bboxes(self, bboxes, labels, dboxes):
        
        target_labels = []
        target_localizations = []
        target_scores = []
            
        for i, layer in enumerate(dboxes):

            t_labels, t_loc, t_scores = self._encode_per_layer(labels, bboxes, layer)
        
            target_labels.append(t_labels)
            target_localizations.append(t_loc)
            target_scores.append(t_scores)
            
        return target_labels, target_localizations, target_scores
    
    
    
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
            
           # print(labels.shape)
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
                           tf.TensorShape([None, None,None]),
                           tf.TensorShape([None, None,None]),
                           tf.TensorShape([None, None,None]),
                           tf.TensorShape([None, None,None]),
                           tf.TensorShape([None, None,None])])
                
                
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