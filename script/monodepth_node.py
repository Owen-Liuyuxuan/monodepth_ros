#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from cv_bridge import CvBridge
from ros_util import depth_image_to_point_cloud_array, publish_point_cloud, depth_image_to_point_cloud_tensor


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
        monodepth_path = rospy.get_param("~MONODEPTH_PATH", "/home/FSNet")
        import sys
        sys.path.append(monodepth_path)
        from vision_base.utils.utils import cfg_from_file

        cfg_file = rospy.get_param("~CFG_FILE", "/home/yxliu/multi_cam/monodepth/configs/kitti360_gtpose_config.py")
        self.cfg = cfg_from_file(cfg_file)
        self.cfg.meta_arch.depth_backbone_cfg.pretrained=False

        self.weight_path = rospy.get_param("~WEIGHT_PATH", "/home/yxliu/multi_cam/monodepth/workdirs/MonoDepth2WPose/checkpoint/MonoDepthWPose_ss11.pth")

        self.inference_w   = int(rospy.get_param("~INFERENCE_W",  640))
        self.inference_h   = int(rospy.get_param("~INFERENCE_H",  192))
        self.inference_scale = float(rospy.get_param("~INFERENCE_SCALE", 1.0))
        self.max_depth = float(rospy.get_param("~MAX_DEPTH", 50))
        self.min_y = float(rospy.get_param("~MIN_Y", -2.0))
        self.max_y = float(rospy.get_param("~MAX_Y", 4.5))

        self.cfg.val_dataset.augmentation.cfg_list[1].size = (self.inference_h, self.inference_w)

    def _init_model(self):
        rospy.loginfo("Loading model.")
        from vision_base.utils.builder import build
        self.meta_arch = build(**self.cfg.meta_arch)
        self.meta_arch = self.meta_arch.cuda()
        state_dict = torch.load(
            self.weight_path, map_location='cuda:{}'.format(self.cfg.trainer.gpu)
        )
        self.meta_arch.load_state_dict(state_dict['model_state_dict'], strict=False)
        self.meta_arch.eval()

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

    def _predict_depth(self, image):
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
            h_depth, w_detph = depth.shape
            
            depth = depth[0:h_eff, 0:w_eff]
            resize_rgb = cv2.resize(image, (w_eff, h_eff))

            crop_top = 0
            croped_depth = depth[crop_top:, :]
            croped_image = resize_rgb[crop_top:, :, :]
            croppred_P = data['P2'][0, 0:3, 0:3].cpu().numpy()
            croppred_P[1, 2] -= crop_top

            point_cloud = depth_image_to_point_cloud_tensor(croped_depth, croppred_P, croped_image)
            mask = (point_cloud[:, 1] > self.min_y) * (point_cloud[:, 1] < self.max_y) * (point_cloud[:, 0] < self.max_depth)

            point_cloud = point_cloud[mask].cpu().numpy()
            

        return depth, point_cloud

    def camera_callback(self, msg):
        height = msg.height
        width  = msg.width
        if self.P is not None:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape([height, width, 3]) #[BGR]
            depth, point_cloud = self._predict_depth(image[:, :, ::-1].copy()) # BGR -> RGB
            publish_point_cloud(point_cloud, self.pc_publisher, self.frame_id, 'xyzrgb')
    
    def compressed_camera_callback(self, msg):
        if self.P is not None:
            image = CvBridge().compressed_imgmsg_to_cv2(msg) #[BGR]
            depth, point_cloud = self._predict_depth(image[:, :, ::-1].copy()) # BGR -> RGB
            publish_point_cloud(point_cloud, self.pc_publisher, self.frame_id, 'xyzrgb')

    def camera_info_callback(self, msg):
        self.P = np.zeros((3, 4))
        self.P[0:3, 0:3] = np.array(msg.K).reshape((3, 3))
        self.frame_id = msg.header.frame_id

if __name__ == "__main__":
    ros_node = RosNode()
    rospy.spin()