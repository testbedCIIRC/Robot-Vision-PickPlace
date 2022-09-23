# Robust and Easily Deployable Pick and Place Architecture for Industry 4.0

Current industrial robot programming approaches use unfamiliar development environments that require advanced knowledge of industrial robot programming. They limit the scalability and flexibility of industrial environments after the initial project commissioning phase where the robots are hardcoded for a specific task. This complicates the integration of state-of-the-art deep learning methods.

![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/robot_video.gif)
![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/detection_frame.gif)

Our developed pick and place architecture allows an industrial KUKA robot to be easily integrated with 3D computer vision methods and state of the art instance segmentation algorithms such as YOLACT for packet detection, segmentation, tracking and optimal gripping pose estimation. We showcase our solution on packets of different colors and sizes and contents.

## HW setup
Our setup consists of a KUKA KR Cybertech industrial robot, a PLC, a conveyor belt with rotary encoder and an Intel Realsense RGBD camera. We mounted apriltags on the conveyor for robot workspace detection.

![alt text]()

## Robot system and control
The high level control, coded in Python, utilizes an OPC UA Client that runs on the PC. It includes a finite state machine for the trajectory planning based on visual recognition, which reads sensor and robot data from the OPC UA nodes and sends high level control actions and path trajectory updates to the OPC UA Server on the PLC. These inputs are converted into control instructions using the mx Automation package and are sent to the robot controller via Profinet.

![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/HWSetup.png)
![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/StateMachineSimple.png)

## Vision

The vision system uses the YOLACT CNN architecture for packet localization, classification and segmentation. We use the generated mask for rotation estimation. Then the objects are registered and tracked using their nearest neighbors between the frames. The real world coordinates of the packet are obtained with the homography matrix between the detected tags and their real world coordinates.

![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/packet_detection.png)
![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/segmented_point_cloud.png)

## Gripping
During the tracking process the depth map is cropped and averaged. The corresponding point cloud is downsampled and the center of mass from its depth values is estimated. Then, we locate cirlic neighborhood with radius given by the dimensions of the gripper. Afterwards the plane is fitted into those points using Least Squares to get the optimal gripping pose vector.

![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/optimal_pt.png)
![alt text](https://github.com/testbedCIIRC/Robot-Vision-PickPlace/blob/main/readme_imgs/downsampled_point_cloud.png)

## Conclusion and future work
Our system shows potential to be scaled and increase the flexibility of industrial pick and place set ups due to the ease of integration of state of the art models and open source software. It may allow developers to get to a working solution much faster. In the future we would like to implement detection and processing of objects with complex geometries. For this we would like to explore the use of synthetic data generation.

![alt text]()

# Environment set up
<br />
<b>1.</b> Clone this repository
<br/><br/>
<b>2.</b> Create a virtual environment 
<pre>
python -m venv robot_cell_env
</pre> 
<br/>
<b>3.</b> Activate virtual environment
<pre>
source robot_cell_env/bin/activate # Linux
.\robot_cell_env\Scripts\activate # Windows 
</pre>
<br/>
<b>4.</b> Add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=robot_cell_env
</pre>
<br/>
<b>5.</b> Run cells in rob_env_create.ipynb. Make sure environment kernel is selected.
