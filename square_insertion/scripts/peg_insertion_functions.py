import franka_cartesian_controller.motion_generator as motion_generator
import franka_cartesian_controller.transform_util as T
import random
import math

from torchvision import transforms
import cv2
import numpy as np
from utils import normalize_q
from datetime import datetime
import os.path
import time as mytime

import cv2
import time
import os.path

file = "/../logs_ins_exe/log_msgs_{}.txt".format(datetime.now())
i = 0
# CAMERA_ID = 0
CAMERA_ID = 2

class InsertionMotions:
    def __init__(self) -> None:
        self.robot = motion_generator.MotionGenerator(verbose="debug")

    def move_to_random_pose(self, T_0A_pos_ref, T_0A_quat_ref, pose_id):
        utils = Utilities()
        utils.print_log("Moving to random position: T_0{}".format(pose_id))

        r = random.uniform(math.radians(-5), math.radians(5))
        p = random.uniform(math.radians(-5), math.radians(5))
        y = random.uniform(math.radians(-10), math.radians(10))
        eef_TAEi_rot_mat = T.euler2mat([r, p, y])
        eef_TAEi_quat = T.mat2quat(eef_TAEi_rot_mat)
        desired_T0Ei_quat = T.quat_multiply(T_0A_quat_ref, eef_TAEi_quat)
        utils.print_log("\nroll: {}". format(math.degrees(r)) +
                "  pitch: {}".format(math.degrees(p)) + "  yaw: {}".format(math.degrees(y)))

        x = random.uniform(-0.005, 0.005)
        y = random.uniform(-0.005, 0.005)
        z = random.uniform(-0.005, 0.005)
        utils.print_log("\nx: {}".format(x) + "  y: {}".format(y) + "  z: {}".format(z))

        self.robot.move_to_pose_request(
            # random eef position, with new pose origin within a vertical cylinder with r=5 mm & h=10 mm, with the default pose T_0 frame origin at the center of the bottom of the cylinder
            target_pos=T_0A_pos_ref + [x, y, z],
            target_quat=desired_T0Ei_quat,
            linear_speed=0.1,
            rotation_speed=0.4,
            max_duration=5
        )

        random_T = []
        random_T[:3] = self.robot.eef_pos
        random_T[3:] = self.robot.eef_quat
        utils.print_log("\nend-effector T_0{} random translation: \n{}".format(pose_id, random_T[:3]))
        utils.print_log("\nend-effector T_0{} random rot_matrix: \n{}".format(pose_id, T.quat2rotation(random_T[3:])))
        
        pre_align_start = mytime.time()

        utils.print_log("Taking picture {}...".format(pose_id))
        image = utils.capture_image(pose_id)

        utils.print_log("T_0{}\n".format(pose_id) + str(random_T[:-1]))
        file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/" + pose_id + ".text", "w")

        if pose_id[0] == "A":
            T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
            print(T_0A, file=file1)
            return image, random_T[0:3], random_T[3:7], pre_align_start
        elif pose_id[0] == "B":
            T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
            print(T_0B, file=file1) 
            return image, random_T[0:3], random_T[3:7], pre_align_start 

    def move_to_random_pose_wo_image(self, T_0A_pos_ref, T_0A_quat_ref, pose_id):
        utils = Utilities()
        utils.print_log("Moving to random position: T_0{}".format(pose_id))

        r = random.uniform(math.radians(-5), math.radians(5))
        p = random.uniform(math.radians(-5), math.radians(5))
        y = random.uniform(math.radians(-10), math.radians(10))
        eef_TAEi_rot_mat = T.euler2mat([r, p, y])
        eef_TAEi_quat = T.mat2quat(eef_TAEi_rot_mat)
        desired_T0Ei_quat = T.quat_multiply(T_0A_quat_ref, eef_TAEi_quat)
        utils.print_log("\nroll: {}". format(math.degrees(r)) +
                "  pitch: {}".format(math.degrees(p)) + "  yaw: {}".format(math.degrees(y)))

        x = random.uniform(-0.005, 0.005)
        y = random.uniform(-0.005, 0.005)
        z = random.uniform(-0.005, 0.005)
        utils.print_log("\nx: {}".format(x) + "  y: {}".format(y) + "  z: {}".format(z))

        self.robot.move_to_pose_request(
            # random eef position, with new pose origin within a vertical cylinder with r=5 mm & h=10 mm, with the default pose T_0 frame origin at the center of the bottom of the cylinder
            target_pos=T_0A_pos_ref + [x, y, z],
            target_quat=desired_T0Ei_quat,
            linear_speed=0.1,
            rotation_speed=0.4,
            max_duration=5
        )

        random_T = []
        random_T[:3] = self.robot.eef_pos
        random_T[3:] = self.robot.eef_quat
        utils.print_log("\nend-effector T_0{} random translation: \n{}".format(pose_id, random_T[:3]))
        utils.print_log("\nend-effector T_0{} random rot_matrix: \n{}".format(pose_id, T.quat2rotation(random_T[3:])))
        
        utils.print_log("T_0{}\n".format(pose_id) + str(random_T[:-1]))
        file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/" + pose_id + ".text", "w")

        if pose_id[0] == "A":
            T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
            print(T_0A, file=file1)
            return random_T[0:3], random_T[3:7]
        elif pose_id[0] == "B":
            T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
            print(T_0B, file=file1) 
            return random_T[0:3], random_T[3:7]   

    def move_to_big_sampling_error(self, T_0A_pos_ref, T_0A_quat_ref, pose_id):
        utils = Utilities()
        utils.print_log("Moving to random position: T_0{}".format(pose_id))

        # r = random.uniform(math.radians(-5), math.radians(5))
        # p = random.uniform(math.radians(-5), math.radians(5))
        # y = random.uniform(math.radians(-10), math.radians(10))
        r = math.radians(-20)
        p = math.radians(-20)
        y = math.radians(-40)
        eef_TAEi_rot_mat = T.euler2mat([r, p, y])
        eef_TAEi_quat = T.mat2quat(eef_TAEi_rot_mat)
        desired_T0Ei_quat = T.quat_multiply(T_0A_quat_ref, eef_TAEi_quat)
        utils.print_log("\nroll: {}". format(math.degrees(r)) +
                "  pitch: {}".format(math.degrees(p)) + "  yaw: {}".format(math.degrees(y)))

        # x = random.uniform(-0.005, 0.005)
        # y = random.uniform(-0.005, 0.005)
        # z = random.uniform(-0.005, 0.005)
        x = -0.02
        y = -0.02
        z = -0.02
        utils.print_log("\nx: {}".format(x) + "  y: {}".format(y) + "  z: {}".format(z))

        self.robot.move_to_pose_request(
            # random eef position, with new pose origin within a vertical cylinder with r=5 mm & h=10 mm, with the default pose T_0 frame origin at the center of the bottom of the cylinder
            target_pos=T_0A_pos_ref + [x, y, z],
            target_quat=desired_T0Ei_quat,
            linear_speed=0.1,
            rotation_speed=0.4,
            max_duration=5
        )

        random_T = []
        random_T[:3] = self.robot.eef_pos
        random_T[3:] = self.robot.eef_quat
        utils.print_log("\nend-effector T_0{} random translation: \n{}".format(pose_id, random_T[:3]))
        utils.print_log("\nend-effector T_0{} random rot_matrix: \n{}".format(pose_id, T.quat2rotation(random_T[3:])))
        
        pre_align_start = mytime.time()

        utils.print_log("Taking picture {}...".format(pose_id))
        image = utils.capture_image(pose_id)

        utils.print_log("T_0{}\n".format(pose_id) + str(random_T[:-1]))
        file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/" + pose_id + ".text", "w")

        if pose_id[0] == "A":
            T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
            print(T_0A, file=file1)
            return image, random_T[0:3], random_T[3:7], pre_align_start
        elif pose_id[0] == "B":
            T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
            print(T_0B, file=file1) 
            return image, random_T[0:3], random_T[3:7], pre_align_start

    def move_to_T_0A_est(self, est_pos, est_quat):
        self.robot.move_to_pose_request(
            target_pos=est_pos,
            target_quat=est_quat,
            linear_speed=0.1,
            rotation_speed=0.4,
            max_duration=5
        )
        achieved_T_0A_pos_est = self.robot.eef_pos
        achieved_T_0A_quat_est = self.robot.eef_quat
        return achieved_T_0A_pos_est, achieved_T_0A_quat_est 


class ModelEstimation:
    def get_relative_pose_T_BA(self, model, img_A_name, img_B_name, device, img_A, img_B):

        # Find T_BA
        
        img_A = cv2.imread(os.path.dirname(__file__) + '/../ins_exe_data/img/' + img_A_name + '.png')
        img_B = cv2.imread(os.path.dirname(__file__) + '/../ins_exe_data/img/' + img_B_name + '.png')
            
        # Convert OpenCV image to Tensor of type float32
        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_A = img_transform(img_A)
        img_B = img_transform(img_B)

        # add batch dimension
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)    

        img_A = img_A.to(device)
        img_B = img_B.to(device)

        # Find output
        output = model(img_A, img_B)
        output = output.cpu().detach().numpy()

        return output[0][:3], normalize_q(output[0][3:])


class Utilities:
    def print_log(self, msg):
        print(msg)
        print(msg, file=open(os.path.dirname(__file__) + file, "a"))

    def find_errors_between_2_poses(self, pose1_pos, pose1_quat, pose2_pos, pose2_quat):   
        # Find position errors between pose 1 and pose 2
        x_error = np.abs(pose1_pos[0] - pose2_pos[0]) * 1000
        y_error = np.abs(pose1_pos[1] - pose2_pos[1]) * 1000
        z_error = np.abs(pose1_pos[2] - pose2_pos[2]) * 1000

        # Find rotation errors (roll, pitch, yaw) between pose 1 and pose 2
        quat_error, axis_angle_error = T.quat_error(pose1_quat, pose2_quat)
        rot_mat_error = T.quat2rotation(quat_error)
        euler_error = T.mat2euler(rot_mat_error)
        roll_error = np.rad2deg(euler_error[0])
        pitch_error = np.rad2deg(euler_error[1])
        yaw_error = np.rad2deg(euler_error[2])

        return x_error, y_error, z_error, np.abs(roll_error), np.abs(pitch_error), np.abs(yaw_error)

    def capture_image(self, image_id):
        camera = cv2.VideoCapture(CAMERA_ID)
        delta = 0
        previous = time.time()
        while True:
            current  = time.time()
            delta += current - previous
            previous = current

            # Show the image and keep streaming
            _, img = camera.read()
            cv2.imshow("Frame", img)
            cv2.waitKey(1)

            # Check if 1 (or some other value) seconds passed
            if delta > 1:
                # Operations on image
                # Reset the time counter
                cv2.imwrite(os.path.dirname(__file__) + '/../ins_exe_data/img/'+ str(image_id) +'.png', img)
                delta = 0
                camera.release()
                cv2.destroyAllWindows() 
                # break
                return img