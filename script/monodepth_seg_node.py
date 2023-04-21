#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from cv_bridge import CvBridge
from ros_util import depth_image_to_point_cloud_array, publish_point_cloud, depth_image_to_point_cloud_tensor


from labels import PALETTE, PRESERVED
from numba import jit

@jit(nopython=True, cache=True)
def HandelSegResults(pred_seg, rgb_image, opacity=1.0, palette=PALETTE, preserved=PRESERVED):
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    mask = np.zeros((pred_seg.shape[0], pred_seg.shape[1]), dtype=np.uint8)
    h, w = pred_seg.shape
    for i in range(h):
        for j in range(w):
            color_seg[i, j] = palette[pred_seg[i, j]]
            mask[i, j] = preserved[pred_seg[i, j]]
    new_image = rgb_image * (1 - opacity) + color_seg * opacity
    new_image = new_image.astype(np.uint8)
    return new_image, mask

def normalize_image(image):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std  = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - rgb_mean
    image = image / rgb_std
    return image

def collate_fn(batch):
    collated_data = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated_data[key] = torch.stack([item[key] for item in batch], dim=0)
        elif isinstance(batch[0][key], np.ndarray):
            collated_data[key] = torch.stack([torch.from_numpy(item[key]) for item in batch], dim=0)

    return collated_data

class RosNode:
    def __init__(self):
        rospy.init_node("ros_node")
        rospy.loginfo("Starting RosNode.")

        self._read_params()
        self._init_model()
        self._init_static_memory()
        self._init_topics()

    def _read_params(self):
        rospy.loginfo("Reading params.")
        monodepth_path = rospy.get_param("~MONODEPTH_PATH", "/home/yxliu/multi_cam/monodepth")
        import sys
        sys.path.append(monodepth_path)
        from lib.utils.utils import cfg_from_file

        cfg_file = rospy.get_param("~CFG_FILE", "/home/yxliu/multi_cam/monodepth/configs/kitti360_distillwpose.py")
        self.cfg = cfg_from_file(cfg_file)
        self.cfg.meta_arch.depth_backbone_cfg.pretrained=False
        # self.cfg.meta_arch.pose_backbone_cfg.pretrained=False

        self.onnx_path = rospy.get_param("~SEG_ONNX_PATH", "/home/yxliu/test_ws/src/segmentation/model/server10_unet_49.onnx")

        self.weight_path = rospy.get_param("~WEIGHT_PATH", "/home/yxliu/multi_cam/monodepth/workdirs/KITTI360_WPoseDistill/checkpoint/lib.networks.models.meta_archs.monodepth2_model.DistillWPoseMeta_19.pth")

        self.inference_w   = int(rospy.get_param("~INFERENCE_W",  640))
        self.inference_h   = int(rospy.get_param("~INFERENCE_H",  192))
        self.inference_scale = float(rospy.get_param("~INFERENCE_SCALE", 1.0))
        self.max_depth = float(rospy.get_param("~MAX_DEPTH", 50))
        self.min_y = float(rospy.get_param("~MIN_Y", -2.0))
        self.max_y = float(rospy.get_param("~MAX_Y", 4.5))

        self.cfg.val_dataset.augmentation.cfg_list[1].size = (self.inference_h, self.inference_w)

    def _init_model(self):
        rospy.loginfo("Loading model.")
        import lib.networks
        import lib.data
        import lib.pipeline_hooks
        import lib.evaluation
        from lib.utils.builder import build
        self.meta_arch = build(**self.cfg.meta_arch)
        self.meta_arch = self.meta_arch.cuda()
        state_dict = torch.load(
            self.weight_path, map_location='cuda:{}'.format(self.cfg.trainer.gpu)
        )
        self.meta_arch.load_state_dict(state_dict['model_state_dict'], strict=False)
        self.meta_arch.eval()
        # self.ort_session = ort.InferenceSession(self.onnx_path, providers=['CUDAExecutionProvider'])
        self.seg_ort_session = ort.InferenceSession(self.onnx_path, providers=['CUDAExecutionProvider'])

        self.transform = build(**self.cfg.val_dataset.augmentation)
        self.test_pipeline = build(**self.cfg.trainer.evaluate_hook.test_run_hook_cfg)
        rospy.loginfo("Done loading model.")

    def _init_static_memory(self):
        self.frame_id = None
        self.P = None

    def _init_topics(self):
        self.pc_publisher = rospy.Publisher("/point_cloud", PointCloud2, queue_size=1)
        rospy.Subscriber("/image_raw", Image, self.camera_callback, buff_size=2**24, queue_size=1)
        rospy.Subscriber("/compressed_image", CompressedImage, self.compressed_camera_callback, buff_size=2**24, queue_size=1)
        rospy.Subscriber("/camera_info", CameraInfo, self.camera_info_callback)

    def _predict_seg(self, image):
        """rgb in rgb out"""
        h0, w0, _ = image.shape
        resized_image = cv2.resize(image, (self.inference_w, self.inference_h), interpolation=cv2.INTER_LINEAR)
        input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
        outputs = self.seg_ort_session.run(None, {'input': input_numpy})
        pred_seg = outputs[0][0]
        colorized_seg, mask = HandelSegResults(pred_seg, resized_image, 1.0)
        return colorized_seg, mask

    def _predict_depth(self, image, colorized_seg, mask):
        data = dict()
        h0, w0, _ = image.shape # (h, w, 3)
        data[('image', 0)] = image.copy()
        data['P2'] = self.P.copy()

        data = self.transform(data)
        h_eff, w_eff = data[('image_resize', 'effective_size')]
        data = collate_fn([data])
        
        with torch.no_grad():
            
            output_dict = self.test_pipeline(data, self.meta_arch)
            depth = output_dict["depth"] * self.inference_scale
            depth = depth[0, 0]
            #h_depth, w_detph = depth.shape
            
            #depth = depth[0:h_eff, 0:w_eff]
            #resize_rgb = cv2.resize(image, (w_detph, h_depth))[0:h_eff, 0:w_eff]

            # torch.clip(depth, 0, self.max_depth, out=depth)

            point_cloud = depth_image_to_point_cloud_tensor(depth, data['P2'][0][0:3, 0:3].cpu().numpy(), colorized_seg, mask)
            mask2 = (point_cloud[:, 1] > self.min_y) * (point_cloud[:, 1] < self.max_y)  * (point_cloud[:, 2] < self.max_depth)

            point_cloud = point_cloud[mask2].cpu().numpy()
            
        return depth, point_cloud

    def camera_callback(self, msg):
        height = msg.height
        width  = msg.width
        if self.P is not None:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape([height, width, 3]) #[BGR]
            colorized_seg, mask = self._predict_seg(image[:, :, ::-1].copy()) 
            depth, point_cloud = self._predict_depth(image[:, :, ::-1].copy(), colorized_seg, mask) # BGR -> RGB
            publish_point_cloud(point_cloud, self.pc_publisher, self.frame_id, 'xyzrgb')
    
    def compressed_camera_callback(self, msg):
        if self.P is not None:
            image = CvBridge().compressed_imgmsg_to_cv2(msg) #[BGR]
            colorized_seg, mask = self._predict_seg(image[:, :, ::-1].copy()) 
            depth, point_cloud = self._predict_depth(image[:, :, ::-1].copy(), colorized_seg, mask) # BGR -> RGB
            publish_point_cloud(point_cloud, self.pc_publisher, self.frame_id, 'xyzrgb')

    def camera_info_callback(self, msg):
        self.P = np.zeros((3, 4))
        self.P[0:3, 0:3] = np.array(msg.K).reshape((3, 3))
        self.frame_id = msg.header.frame_id

if __name__ == "__main__":
    ros_node = RosNode()
    rospy.spin()