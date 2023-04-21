#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort
import message_filters
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

        cfg_file = rospy.get_param("~CFG_FILE", "/home/yxliu/multi_cam/monodepth/configs/nusc_json_288512.py")
        self.cfg = cfg_from_file(cfg_file)
        self.cfg.meta_arch.depth_backbone_cfg.pretrained=False
        # self.cfg.meta_arch.pose_backbone_cfg.pretrained=False

        # self.onnx_path = rospy.get_param("~ONNX_PATH", "/home/yxliu/test_ws/model/monodepth.onnx")
        self.weight_path = rospy.get_param("~WEIGHT_PATH",
             "/home/yxliu/multi_cam/new_stereo/submitted_result/monodepth_unsupervised/Res34WPoseNusc_288_512_latest.pth")

        self.inference_w   = int(rospy.get_param("~INFERENCE_W",  512))
        self.inference_h   = int(rospy.get_param("~INFERENCE_H",  288))
        self.inference_scale = float(rospy.get_param("~INFERENCE_SCALE", 1.0))
        self.max_depth = float(rospy.get_param("~MAX_DEPTH", 50))
        self.min_y = float(rospy.get_param("~MIN_Y", -2.0))
        self.max_y = float(rospy.get_param("~MAX_Y", 5.5))

        self.channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        

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

        self.transform = build(**self.cfg.val_dataset.augmentation)
        self.test_pipeline = build(**self.cfg.trainer.evaluate_hook.test_run_hook_cfg)
        rospy.loginfo("Done loading model.")

    def _init_static_memory(self):
        self.frame_ids = dict()
        self.Ps = dict()
        self.scales = dict()
        for key in self.channels:
            self.Ps[key] = None
            self.frame_ids[key] = None
            self.scales[key] = self.inference_scale
            if key == 'CAM_BACK':
                self.scales[key] *= 809 / 1259

    def _init_topics(self):
        self.pc_publisher = {}
        for cam in self.channels:
            self.pc_publisher[cam] = rospy.Publisher(f"/point_cloud_{cam.lower()}", PointCloud2, queue_size=1)
        
        self.image_sub_per_channel = {}
        self.info_sub = {}
        for cam in self.channels:
            self.image_sub_per_channel[cam] = message_filters.Subscriber(f"/nuscenes/{cam}/image", Image, buff_size=2**24, queue_size=3)
            self.info_sub[cam] = message_filters.Subscriber(f"/nuscenes/{cam}/camera_info", CameraInfo, queue_size=2)
        self.ts_cam = message_filters.ApproximateTimeSynchronizer([item for _,item in self.image_sub_per_channel.items()], 2, slop=0.2)
        self.ts_cam.registerCallback(self.cameras_callback)

        self.ts_info = message_filters.ApproximateTimeSynchronizer([item for _,item in self.info_sub.items()], 2, slop=0.3)
        self.ts_info.registerCallback(self.infos_callback)

    def _predict_depth(self, images):
        data = dict()

        data = collate_fn(
            [self.transform(
                {
                    ('image', 0): images[i].copy(),
                    'P2': self.Ps[self.channels[i]].copy()
                }
            ) for i in range(len(images))]    
        )
        h_eff, w_eff = data[('image_resize', 'effective_size')][0]

        output_pcs = []
        
        with torch.no_grad():

            for key in data:
                data[key] = data[key].cuda().contiguous()
            meta = dict(epoch_num=0, global_step=0, is_training=False)
            output_dict = self.meta_arch(data, meta)
            
            depths = output_dict["depth"]
            _, _, h_depth, w_detph = depths.shape
            #depths = F.adaptive_avg_pool2d(depths, (h0, w0))
            
            for i, cam in enumerate(self.channels):
                depth = depths[i, 0] * self.scales[cam]
                depth = depth[0:h_eff, 0:w_eff]
                h, w = depth.shape
                torch.clip(depth, 0, self.max_depth, out=depth)

                resize_rgb = cv2.resize(images[i], (w_detph, h_depth))[0:h_eff, 0:w_eff]
                point_cloud = depth_image_to_point_cloud_tensor(depth, data['P2'][i, 0:3, 0:3].cpu().numpy(), cv2.resize(images[i], (w, h)))
                mask = (point_cloud[:, 1] > self.min_y) * (point_cloud[:, 1] < self.max_y) * (point_cloud[:, 2] < self.max_depth)

                point_cloud = point_cloud[mask].cpu().numpy()
                output_pcs.append(point_cloud)
        
        return output_pcs

    def cameras_callback(self, *msgs):
        images = []
        for i in range(len(self.channels)):
            channel = self.channels[i]
            height = msgs[i].height
            width  = msgs[i].width
            if self.Ps[channel] is not None:
                image = np.frombuffer(msgs[i].data, dtype=np.uint8).reshape([height, width, 3]) #[BGR]
                images.append(image[:, :, ::-1])
            else:
                return
        point_clouds = self._predict_depth(images) # BGR -> RGB
        for i in range(len(self.channels)):
            channel = self.channels[i]
            publish_point_cloud(point_clouds[i], self.pc_publisher[channel], self.frame_ids[channel], 'xyzrgb')


    def infos_callback(self, *msgs):
        for i in range(len(self.channels)):
            channel = self.channels[i]
            self.Ps[channel] = np.zeros((3, 4))
            self.Ps[channel][0:3, 0:3] = np.array(msgs[i].K).reshape((3, 3))
            self.frame_ids[channel] = msgs[i].header.frame_id

if __name__ == "__main__":
    ros_node = RosNode()
    rospy.spin()
