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
        rospy.init_node("right_fisheye_monodepth_ros_node")
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
        from vision_base.utils.utils import cfg_from_file

        cfg_file = rospy.get_param("~CFG_FILE", "/home/yxliu/multi_cam/monodepth/configs/kitti360_fisheye.py")
        self.cfg = cfg_from_file(cfg_file)
        self.cfg.meta_arch.depth_backbone_cfg.pretrained=False
        # self.cfg.meta_arch.pose_backbone_cfg.pretrained=False

        self.onnx_path = rospy.get_param("~ONNX_PATH", "/home/yxliu/test_ws/model/monodepth.onnx")
        self.weight_path = rospy.get_param("~WEIGHT_PATH", "/home/yxliu/multi_cam/monodepth/workdirs/KITTI360_fisheye/checkpoint/lib.networks.models.meta_archs.monodepth2_model.MonoDepthWPose_19.pth")

        self.fish_eye_mask = torch.from_numpy(cv2.resize(cv2.imread('/home/yxliu/test_ws/src/monodepth/model/fisheye_mask.png', -1), (384, 384), cv2.INTER_NEAREST)).bool().cuda()

        self.inference_w   = int(rospy.get_param("~INFERENCE_W",  384))
        self.inference_h   = int(rospy.get_param("~INFERENCE_H",  384))
        self.inference_scale = float(rospy.get_param("~INFERENCE_SCALE", 1.0))
        self.max_depth = float(rospy.get_param("~MAX_DEPTH", 50))
        self.min_y = float(rospy.get_param("~MIN_Y", -0.0))
        self.max_y = float(rospy.get_param("~MAX_Y", 2.5))

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
        # self.ort_session = ort.InferenceSession(self.onnx_path, providers=['CUDAExecutionProvider'])
        self.projector = build(**dict(name="lib.networks.utils.mei_fisheye_utils.MeiCameraProjection"))
        self.transform = build(**self.cfg.val_dataset.augmentation)
        self.test_pipeline = build(**self.cfg.trainer.evaluate_hook.test_run_hook_cfg)
        rospy.loginfo("Done loading model.")

    def _init_static_memory(self):
        self.frame_id = None
        self.P = None

    def _init_topics(self):
        self.pc_publisher = rospy.Publisher("/point_cloud_right", PointCloud2, queue_size=1)
        rospy.Subscriber("/kitti360/right_fisheye_camera/image", Image, self.camera_callback, buff_size=2**24, queue_size=1)
        rospy.Subscriber("/compressed_image", CompressedImage, self.compressed_camera_callback, buff_size=2**24, queue_size=1)
        rospy.Subscriber("/kitti360/right_fisheye_camera/camera_info", CameraInfo, self.camera_info_callback)

    def _predict_depth(self, image):
        data = dict()
        h0, w0, _ = image.shape # (h, w, 3)
        data[('image', 0)] = image.copy()
        data['P2'] = self.P.copy()
        
        data = self.transform(data)
        h_eff, w_eff = data[('image_resize', 'effective_size')]
        data = collate_fn([data])
        data['calib_meta'] = self.calib
        with torch.no_grad():
            output_dict = self.test_pipeline(data, self.meta_arch)
            depth = output_dict["norm"] * self.inference_scale
            point_cloud, mask = self.projector.image2cam(depth, data['P2'], self.calib)
            mask = mask * (depth > 1.3) * self.fish_eye_mask
            print(point_cloud.shape, data[('original_image', 0)][None].shape)
            rgb_point_cloud = torch.cat([point_cloud, data[('original_image', 0)][None].contiguous() / 256], dim=-1)
            rgb_point_cloud = rgb_point_cloud[:, :, 210:300, 100:284][mask[:, :, 210:300, 100:284].bool()] #[N, 3]
            print(rgb_point_cloud[:, 2].min(), rgb_point_cloud[:, 2].max())
            mask = (rgb_point_cloud[:, 1] > self.min_y) * (rgb_point_cloud[:, 1] < self.max_y) * (rgb_point_cloud[:, 2] < self.max_depth) * (rgb_point_cloud[:, 2] > 0.1) 


            rgb_point_cloud = rgb_point_cloud[mask].cpu().numpy()
            print(rgb_point_cloud.shape)            

        return depth, rgb_point_cloud

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
        # self.P = np.zeros((3, 4))
        # self.P[0:3, 0:3] = np.array(msg.K).reshape((3, 3))
        self.frame_id = msg.header.frame_id
        self.P = np.array([1.4854388981875156e+03, 0, 6.9888316784030962e+02, 0,
                           0, 1.4849477411748708e+03, 6.9814541887723055e+02, 0,
                           0, 0, 1, 0]).reshape([3, 4])
        calib = dict(
                distortion_parameters=dict(k1=4.9370396274089505e-02, k2=4.5068455478645308e+00),
                mirror_parameters=dict(xi=2.5535139132482758e+00)
            )
        # self.P = np.array([1.3363220825849971e+03, 0, 7.1694323510126321e+02, 0,
        #                    0, 1.3357883350012958e+03, 7.0576498308221585e+02, 0,
        #                    0, 0, 1, 0]).reshape([3, 4])
        # calib = dict(
        #         distortion_parameters=dict(k1=1.6798235660113681e-02, k2=1.6548773243373522e+00),
        #         mirror_parameters=dict(xi=2.2134047507854890e+00)
        #     )
        self.calib = [calib]

if __name__ == "__main__":
    ros_node = RosNode()
    rospy.spin()