## MonoDepth ROS Node

This repo contains a Monodepth Ros node. Base on https://github.com/Owen-Liuyuxuan/monodepth

Recommend working with https://git.ram-lab.com/yuxuan/kitti360_visualize / https://git.ram-lab.com/yuxuan/kitti_visualize

All parameters are exposed in the launch file.

![image](example.png)

### Subscribed Topics

image_raw ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html))

A stream of rectifiled image to be predicted using monodepth.

camera_info ([sensor_msgs/CameraInfo](http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html))

Camera calibration information of the rectified image.

### Published Topics

point_cloud ([sensor_msgs/PointCloud2](http://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html))

Predicted point clouds.



Latest update: We also add a segmentation model trained by [segmentation](http://gitlab.ram-lab.com/yuxuan/segmentation) repo exported with onnx. [Model File](http://gofile.me/4jm56/MWwTw1BRE)

