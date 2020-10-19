

rpn_fg_iou_thresh = 0.7
rpn_bg_iou_thresh = 0.3
rpn_num_samples = 256
rpn_positive_fraction = 0.5
rpn_reg_weights = (1., 1., 1., 1.)
rpn_pre_nms_top_n_train = 2000
rpn_pre_nms_top_n_test = 1000
rpn_post_nms_top_n_train = 2000
rpn_post_nms_top_n_test = 1000
rpn_nms_thresh = 0.7
box_fg_iou_thresh = 0.5
box_bg_iou_thresh = 0.5
box_num_samples = 512
box_positive_fraction = 0.25
box_reg_weights = (10., 10., 5., 5.)
box_score_thresh = 0.1
box_nms_thresh = 0.6
box_num_detections = 100

anchor_sizes = (128, 256, 512)
anchor_ratios = (0.5, 1, 2)

layers = (256, 256, 256, 256)
dim_reduced = 256
