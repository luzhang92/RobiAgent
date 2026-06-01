from typing import Optional, Any
from dataclasses import dataclass
import numpy as np
import time
from lerobot.utils.robot_utils import precise_sleep
import cv2

@dataclass
class PhoneTouchInternalParams:
    cam_to_ik_rot: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )

    ee_target_pos = {
            'ee.x': 0.1579751324680724,
            'ee.y': 0.0,
            'ee.z': 0.1302531730019664641,
            'ee.wx': 1.4786710303797914,
            'ee.wy': 1.5046752548204325,
            'ee.wz': 0.675355774095353,
            'ee.gripper_pos': 5.677154582763338
        }

    ik_target_frame_name: str = "gripper_frame_link"
    urdf_path: str = ""

@dataclass
class ScrollDirection:
    motors: str
    move: int = -1
    range: float = 1.0


class RightSO101Controller:
    def __init__(
        self,
        port: str,
        robot_id: str,
        task_config,
        internal: PhoneTouchInternalParams
    ):
        self.port = port
        self.robot_id = robot_id
        self.internal = internal
        self.robot = None
        self.ee_to_joints = None
        self.joints_to_ee = None
        self.action_keys: list[str] = []
        self.home: dict[str, float] | None = None
        self.home_ik: dict[str, float] | None = None
        self.num_steps = task_config.interpolation_steps
        self.move_epoch = task_config.move_epoch
        self.scroll_epoch = task_config.scroll_epoch
        self.ee_act = internal.ee_target_pos
        self.use_weighted_interpolation = task_config.use_weighted_interpolation
        self.touch_step = task_config.touch_step
        self.direction_action_dict = {
            "up" : ScrollDirection(motors="elbow_flex.pos", move=-1, range=1),
            "down" : ScrollDirection(motors="elbow_flex.pos", move=1, range=1),
            "right" : ScrollDirection(motors="shoulder_pan.pos", move=1, range=0.5),
            "left" : ScrollDirection(motors="shoulder_pan.pos", move=-1, range=0.5),
            }

    def connect(self):
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.processor import RobotProcessorPipeline
        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            transition_to_robot_action,
            observation_to_transition,
            transition_to_observation,
        )
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            ForwardKinematicsJointsToEE,
            InverseKinematicsEEToJoints,
        )

        # Keep holding torque even after normal disconnect / program exit.
        # lerobot's SOFollower.disconnect() can disable torque depending on this flag.
        self.robot = SOFollower(
            SOFollowerRobotConfig(
                id=self.robot_id,
                port=self.port,
                use_degrees=True,
                # disable_torque_on_disconnect=False,
            )
        )
        self.robot.connect(calibrate=False)
        self.action_keys = list(self.robot.action_features.keys())
        self.home = self.get_pose()

        motor_names = list(self.robot.bus.motors.keys())
        kinematics_solver = RobotKinematics(
            urdf_path=self.internal.urdf_path,
            target_frame_name=self.internal.ik_target_frame_name,
            joint_names=motor_names,
        )
        self.ee_to_joints = RobotProcessorPipeline(
            [
                InverseKinematicsEEToJoints(
                    kinematics=kinematics_solver,
                    motor_names=motor_names,
                    initial_guess_current_joints=False,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        self.joints_to_ee = RobotProcessorPipeline(
            [
                ForwardKinematicsJointsToEE(
                    kinematics=kinematics_solver, motor_names=motor_names
                )
            ],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )
        self.home_ik = self.get_ik()

    def disconnect(self):
        if self.robot is None:
            return
        # Intentionally do NOT disable torque here.
        # For demo / tracking, we want motors to keep holding their last pose
        # after normal program exit (as long as they remain powered).
        self.robot.disconnect()
        self.robot = None

    def get_pose(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {k: float(obs[k]) for k in self.action_keys}

    def get_observation(self) -> dict[str, Any]:
        return self.robot.get_observation()

    def get_ik(self):
        obs = self.robot.get_observation()
        ee_pos = self.joints_to_ee(obs)
        return ee_pos

    def go_home(self):
        if self.use_weighted_interpolation:
            ik_origin = self.get_ik()
            home_ik = self.home_ik
            only_ik_interpolation = False
            if only_ik_interpolation:
                origin_loc = np.array([ik_origin['ee.x'], ik_origin['ee.y'], ik_origin['ee.z']])
                home_loc = np.array([home_ik['ee.x'], home_ik['ee.y'], home_ik['ee.z']])
                ik_list = self.weighted_interpolation(origin_loc, home_loc, self.num_steps)
                ee_act = ik_origin
                for ik in ik_list:
                    ee_act['ee.x'] = ik[0]
                    ee_act['ee.y'] = ik[1]
                    ee_act['ee.z'] = ik[2]
                    self.move_arm(ee_act, 1)
            else:
                ik_list = self.interpolate_pose(ik_origin, home_ik, self.num_steps)
                for ik_act in ik_list:
                    self.move_arm(ik_act, 1)
        else:
            self.robot.send_action(dict(self.home))

    def slerp_quat(self, q_start, q_end, t):

        """四元数球面线性插值"""
        # 确保取最短路径
        dot = np.dot(q_start, q_end)
        if dot < 0.0:
            q_end = -q_end
            dot = -dot
        
        # 防止除零
        if dot > 0.9995:
            result = q_start + t * (q_end - q_start)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(dot)  # 两四元数夹角
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s1 = np.sin(theta_0 - theta) / sin_theta_0
        s2 = sin_theta / sin_theta_0
        return s1 * q_start + s2 * q_end 
    
    def interpolate_pose(self, start_pose, end_pose, num_steps):
        """
        start_pose: dict with keys 'ee.x', 'ee.y', 'ee.z', 'ee.wx', 'ee.wy', 'ee.wz'
        end_pose: same format
        num_steps: 插值步数
        """
        # lazy import
        from lerobot.utils.rotation import Rotation

        # 提取位置和姿态
        start_pos = np.array([start_pose['ee.x'], start_pose['ee.y'], start_pose['ee.z']])
        end_pos = np.array([end_pose['ee.x'], end_pose['ee.y'], end_pose['ee.z']])
        
        start_aa = np.array([start_pose['ee.wx'], start_pose['ee.wy'], start_pose['ee.wz']])
        end_aa = np.array([end_pose['ee.wx'], end_pose['ee.wy'], end_pose['ee.wz']])
        
        # 姿态转四元数
        q_start = Rotation.from_rotvec(start_aa).as_quat()
        q_end = Rotation.from_rotvec(end_aa).as_quat()
        
        trajectory = []
        # 生成 0 到 1 之间的线性序列
        linear_t = np.linspace(0, 1, num_steps)

        # 使用 sin 函数将线性序列映射为非线性权重，实现缓入缓出效果
        # 这个权重序列会先慢后快再慢
        weights = np.sin(linear_t * np.pi / 2) ** 2
        for t in weights:        
            # 位置 LERP
            pos = (1 - t) * start_pos + t * end_pos
            
            # 姿态 SLERP
            q = self.slerp_quat(q_start, q_end, t)
            aa = Rotation.from_quat(q).as_rotvec()
            
            trajectory.append({
                'ee.x': pos[0],
                'ee.y': pos[1],
                'ee.z': pos[2],
                'ee.wx': aa[0],
                'ee.wy': aa[1],
                'ee.wz': aa[2],
                'ee.gripper_pos': start_pose['ee.gripper_pos']
            })
        
        return trajectory

    def weighted_interpolation(self, point1, point2, num_steps):
        """
        生成带有缓入缓出效果的非线性插值点序列。
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        diff = p2 - p1

        # 生成 0 到 1 之间的线性序列
        linear_t = np.linspace(0, 1, num_steps)

        # 使用 sin 函数将线性序列映射为非线性权重，实现缓入缓出效果
        # 这个权重序列会先慢后快再慢
        weights = np.sin(linear_t * np.pi / 2) ** 2

        # 使用非线性权重计算插值点
        return np.array([p1 + diff * w for w in weights])

    def linear_interpolation(self, point1, point2, num_steps):
        """
        生成点 point1 到 point2 的线性插值序列（匀速，等间距）。

        参数:
            point1: 起始点（标量、列表或 numpy 数组）
            point2: 终点（同上）
            num_steps: 插值步数（包含起点和终点）

        返回:
            numpy 数组，形状为 (num_steps, dim)，dim 为点的维度
        """
        p1 = np.asarray(point1)
        p2 = np.asarray(point2)
        diff = p2 - p1

        # 生成 [0, 1] 上均匀分布的权重序列，长度 = num_steps
        weights = np.linspace(0, 1, num_steps)

        # 向量化计算所有插值点：p1 + weights * diff
        # 使用 outer 或广播：weights 形状 (num_steps,)，diff 形状 (dim,)
        return p1 + np.outer(weights, diff)

    def move_arm(self, ee_act, move_epoch: int):
        for i in range(move_epoch):
            t0 = time.perf_counter()
            robot_obs = self.robot.get_observation()
            joints_act = self.ee_to_joints((ee_act, robot_obs))
            sended_act = self.robot.send_action(joints_act)
            precise_sleep(max(1.0 / 30 - (time.perf_counter() - t0), 0.0))

    def touch_phone(self, phone_ik: np.ndarray, touch: bool = True):
        self.ee_act['ee.x'] = phone_ik[0] - 0.03
        self.ee_act['ee.y'] = phone_ik[1]
        self.ee_act['ee.z'] = phone_ik[2]
        only_ik_interpolation = False
        if self.use_weighted_interpolation:
            robot_obs = self.robot.get_observation()
            ee_obs = self.joints_to_ee(robot_obs)
            # ee_list = self.interpolate_pose(ee_obs, self.ee_act, self.num_steps)
            # print(ee_list)
            if only_ik_interpolation:
                origin_loc = np.array([ee_obs['ee.x'], ee_obs['ee.y'], ee_obs['ee.z']])
                target_loc = np.array([self.ee_act['ee.x'], self.ee_act['ee.y'], self.ee_act['ee.z']])
                ee_list = self.weighted_interpolation(origin_loc, target_loc, self.num_steps)
                ee_act = self.ee_act
                for ee in ee_list:
                    ee_act['ee.x'] = ee[0]
                    ee_act['ee.y'] = ee[1]
                    ee_act['ee.z'] = ee[2]
                    self.move_arm(ee_act, 1)
            else:
                ee_list = self.interpolate_pose(ee_obs, self.ee_act, self.num_steps)
                for ee_act in ee_list:
                    self.move_arm(ee_act, 1)
        else:
            self.move_arm(self.ee_act, self.move_epoch)

    def verify_position(self):
        obs = self.robot.get_observation()
        last_ee = self.joints_to_ee(obs)
        print(f"last_ee: {last_ee}")

    def scroll_phone(self, direction: str = "up"):
        ###########################################################
        #  direction: up(上滑), down(下滑), right(右滑), left(左滑)  #
        ###########################################################

        #往前启动一下，顶到手机
        self.ee_act['ee.x'] += 0.03
        # self.ee_act['ee.z'] += 0.03
        if self.use_weighted_interpolation:
            robot_obs = self.robot.get_observation()
            ee_obs = self.joints_to_ee(robot_obs)
            only_ik_interpolation = False
            if only_ik_interpolation:
                origin_loc = np.array([ee_obs['ee.x'], ee_obs['ee.y'], ee_obs['ee.z']])
                target_loc = np.array([self.ee_act['ee.x'], self.ee_act['ee.y'], self.ee_act['ee.z']])
                ee_act_list = self.weighted_interpolation(origin_loc, target_loc, 10)
                ee_act = self.ee_act
                for ee_act_array in ee_act_list:
                    ee_act['ee.x'] = ee_act_array[0]
                    ee_act['ee.y'] = ee_act_array[1]
                    ee_act['ee.z'] = ee_act_array[2]
                    self.move_arm(ee_act,1)
            else:
                ee_list = self.interpolate_pose(ee_obs, self.ee_act, self.num_steps)
                for ee_act in ee_list:
                    self.move_arm(ee_act, 1)
        else:
            self.move_arm(self.ee_act, self.move_epoch)
        time.sleep(0.5)

        motor = self.direction_action_dict[direction].motors
        move_direction = self.direction_action_dict[direction].move
        move_range = self.direction_action_dict[direction].range
        obs = self.robot.get_observation()
        action = obs

        for _ in range(self.scroll_epoch):
            t0 = time.perf_counter()
            action[motor] += (move_direction*move_range)
            self.robot.send_action(action)
            precise_sleep(max(1.0 / 45 - (time.perf_counter() - t0), 0.0))


    def tap_phone(self):
        ee_act = self.get_ik()
        for i in (1, -1):
            ee_act['ee.x'] += i * 0.03
            if self.use_weighted_interpolation:
                robot_obs = self.robot.get_observation()
                ee_obs = self.joints_to_ee(robot_obs)
                origin_loc = np.array([ee_obs['ee.x'], ee_obs['ee.y'], ee_obs['ee.z']])
                target_loc = np.array([ee_act['ee.x'], ee_act['ee.y'], ee_act['ee.z']])
                ee_act_list = self.weighted_interpolation(origin_loc, target_loc, 20)
                # ee_act = self.ee_act
                for ee_act_array in ee_act_list:
                    ee_act['ee.x'] = ee_act_array[0]
                    ee_act['ee.y'] = ee_act_array[1]
                    ee_act['ee.z'] = ee_act_array[2]
                    self.move_arm(ee_act,1)
            else:
                self.move_arm(self.ee_act, self.move_epoch)

    def operate_phone(self, phone_ik: np.ndarray, direction: str = "up"):
        if direction.lower() not in self.direction_action_dict.keys():
            print("direction can only support for: up, down, right, left! Please check the direction!")
            return
        phone_ik_temp = phone_ik - np.array([0.03, 0.0, 0.00])
        for _ in range(self.touch_step):
            self.touch_phone(phone_ik)
            time.sleep(1)
            self.scroll_phone(direction)
            time.sleep(0.5)
            self.touch_phone(phone_ik_temp,False)
            time.sleep(0.5)

class Camera:
    def __init__(
        self,
        camera_config
    ):
        self.camera_config = camera_config
    def connect(self):
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
        config = RealSenseCameraConfig(
            serial_number_or_name= self.camera_config.serial_number_or_name,
            fps=self. camera_config.fps,
            width=self.camera_config.width,
            height=self.camera_config.height,
            color_mode=ColorMode.RGB,
            use_depth=True,
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=self.camera_config.warmup_s
        )
        self.camera = RealSenseCamera(config)
        self.camera.connect()

    def disconnect(self):
        self.camera.disconnect()
        self.camera = None

    def capture(self, save_path = "./phone.jpg"):
        color_frame = self.camera.read()
        color_frame_BGR = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, color_frame_BGR)
        depth_map = self.camera.read_depth()
        return color_frame_BGR, depth_map

class PhoneDectector:
    def __init__(
        self,
        task_config,
        camera_config,
        internal_config: Optional[PhoneTouchInternalParams]
    ):
        from ultralytics import YOLO
        self.model = YOLO(task_config.model_path)
        self.camera_config = camera_config
        self.internal_config = internal_config
        self.R_ik_cam = np.array(internal_config.cam_to_ik_rot, dtype=np.float64)
        self.t_ik_cam = np.array(task_config.geometry.camera_origin_in_ik_m, dtype=np.float64)

    def detect(self, source="./phone.jpg"):
        results = self.model.predict(source=source, conf=0.1, classes=67, save=False)
        return results

    def get_phone_axis(self, depth_map, source="./phone.jpg", loc: str = "center"):
        status: str = "success"
        results = self.detect(source)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            status = "no phone detection"
            return status, None
        rgb_frame = results[0].plot()
        # rgb_frame = cv2.cvtColor(annotate_frame, cv2.COLOR_RGB2BGR)

        # 手机中心点像素坐标
        x_px = results[0].boxes.xywh[0][0].item()
        y_px = results[0].boxes.xywh[0][1].item()
        width_px = results[0].boxes.xywh[0][2].item()
        height_px = results[0].boxes.xywh[0][3].item()
        width_px_step = width_px / 10
        height_px_step = height_px / 10
        print(f"width_px_step: {width_px_step}, height_px_step: {height_px_step}")
        if loc == "left top":
            # 手机左上角像素坐标
            x_px = results[0].boxes.xyxy[0][0].item() + width_px_step
            y_px = results[0].boxes.xyxy[0][1].item() + height_px_step
        elif loc == "right top":
            # 手机右上角像素坐标
            x_px = results[0].boxes.xyxy[0][2].item() - width_px_step
            y_px = results[0].boxes.xyxy[0][1].item() + height_px_step
        elif loc == "left bottom":
            # 手机左下角像素坐标
            x_px = results[0].boxes.xyxy[0][0].item() + width_px_step
            y_px = results[0].boxes.xyxy[0][3].item() - height_px_step
        elif loc == "right bottom":
            # 手机右下角像素坐标
            x_px = results[0].boxes.xyxy[0][2].item() - width_px_step
            y_px = results[0].boxes.xyxy[0][3].item() - height_px_step
        elif loc == "photo button center":
            # x_px = results[0].boxes.xyxy[0][2].item() - width_px_step
            y_px = results[0].boxes.xyxy[0][3].item() - 2 * height_px_step

        cv2.circle(rgb_frame, (int(x_px), int(y_px)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow("phone detection", rgb_frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        print(f"像素坐标是: ({x_px} , {y_px})")
        phone_z_m = depth_map[int(y_px), int(x_px)] / 10000

        if phone_z_m < 0.01:
            # TODO: fix this
            status = "no depth information"
            print(status)
            phone_z_m = 0.3898
            # return status, None
        
        if phone_z_m > 0.4:
            status = "wrong depth information, phone is too far away"
            print(status)
            phone_z_m = 0.3898
            # return status, None

        phone_x_m = (x_px - self.camera_config.cx) * phone_z_m / self.camera_config.fx
        phone_y_m = (y_px - self.camera_config.cy) * phone_z_m / self.camera_config.fy
        print(phone_x_m, phone_y_m, phone_z_m)
        phone_cam = np.array([phone_x_m, phone_y_m, phone_z_m], dtype=np.float64)
        phone_ik = self.R_ik_cam @ phone_cam + self.t_ik_cam

        return status, phone_ik

class PhoneTouch:
    def __init__(
        self,
        task_config,
        body_config,
        camera_config,
        internal_config: Optional[PhoneTouchInternalParams] = None
    ):
        self.task_config = task_config
        self.body_config = body_config
        self.camera_config = camera_config
        self.internal_config = internal_config or PhoneTouchInternalParams()
        self.internal_config.urdf_path = body_config.urdf_path

        self.arm = RightSO101Controller(body_config.port, body_config.id, task_config, self.internal_config)
        self.camera = Camera(camera_config)
        self.phone_detector = PhoneDectector(task_config, camera_config, self.internal_config)

    def connect(self):
        self.arm.connect()
        self.camera.connect()

    def disconnect(self):
        self.arm.disconnect()
        self.camera.disconnect()

    def run_forever(self, return_on_finish=True, args=None, execute=True):
        try:
            color_frame, depth_map = self.camera.capture()
            status,phone_axis = self.phone_detector.get_phone_axis(depth_map, color_frame, "center")
            if phone_axis is None:
                print(status)
                return
            print(phone_axis)
            self.arm.operate_phone(phone_axis, "up")
            time.sleep(1)
            print(self.arm.get_ik())
            self.arm.go_home()
            time.sleep(2)

        except Exception as e:
            print(e)
