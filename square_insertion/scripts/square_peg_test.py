import franka_cartesian_controller.motion_generator as motion_generator
import franka_cartesian_controller.transform_util as T
import random
import math
import capture_image as capture

import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from model import VSNet
from transformations import euler_from_quaternion
from utils import normalize_q
from datetime import datetime
import os.path
import time

robot = motion_generator.MotionGenerator(verbose="debug")
file = "/../logs/log_msgs_{}.txt".format(datetime.now())


def print_log(msg):
    print(msg)
    print(msg, file=open(os.path.dirname(__file__) + file, "a"))


def move_to_random_pose(pose_id):
    print_log("Moving to random position: T_0{}".format(pose_id))

    r = random.uniform(math.radians(-5), math.radians(5))
    p = random.uniform(math.radians(-5), math.radians(5))
    y = random.uniform(math.radians(-10), math.radians(10))
    eef_TCEi_rot_mat = T.euler2mat([r, p, y])
    eef_TCEi_quat = T.mat2quat(eef_TCEi_rot_mat)
    desired_T0Ei_quat = T.quat_multiply(eef_T0C_quat, eef_TCEi_quat)
    print_log("\nroll: {}". format(math.degrees(r)) +
            "  pitch: {}".format(math.degrees(p)) + "  yaw: {}".format(math.degrees(y)))

    x = random.uniform(-0.005, 0.005)
    y = random.uniform(-0.005, 0.005)
    z = random.uniform(0, 0.01)
    print_log("\nx: {}".format(x) + "  y: {}".format(y) + "  z: {}".format(z))

    robot.move_to_pose_request(
        # random eef position, with new pose origin within a vertical cylinder with r=5 mm & h=10 mm, with the default pose T_0 frame origin at the center of the bottom of the cylinder
        target_pos=eef_T0C_pos + [x, y, z],
        target_quat=desired_T0Ei_quat,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
    )

    random_T = []
    random_T[:3] = robot.eef_pos
    random_T[3:] = robot.eef_quat
    print_log("\nend-effector T_0{} random translation: \n{}".format(pose_id, random_T[:3]))
    print_log("\nend-effector T_0{} random rot_matrix: \n{}".format(pose_id, T.quat2rotation(random_T[3:])))

    print_log("Taking picture {}...".format(pose_id))
    image = capture.capture_image(pose_id)

    print_log("T_0{}\n".format(pose_id) + str(random_T[:-1]))
    file1 = open(os.path.dirname(__file__) + "/../model_testing_data/label/" + pose_id + ".text", "w")

    if pose_id[0] == "A":
        T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
        print(T_0A, file=file1)
        return image, random_T[0:3], random_T[3:7]
    elif pose_id[0] == "B":
        T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
        print(T_0B, file=file1) 
        return image, random_T[0:3], random_T[3:7]       


def get_relative_pose_T_BA(model, img_A_name, img_B_name, device, img_A, img_B):
    img_A = cv2.imread(os.path.dirname(__file__) + '/../model_testing_data/img/' + img_A_name + '.png')
    img_B = cv2.imread(os.path.dirname(__file__) + '/../model_testing_data/img/' + img_B_name + '.png')

    # Convert OpenCV image to Tensor of type float32
    img_transform = transforms.Compose([
        # transforms.Resize(size=(480,640)),
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

def move_to_T_0A_est(est_pos, est_quat):
    robot.move_to_pose_request(
        target_pos=est_pos,
        target_quat=est_quat,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
    )
    achieved_T_0A_pos_est = robot.eef_pos
    achieved_T_0A_quat_est = robot.eef_quat
    return achieved_T_0A_pos_est, achieved_T_0A_quat_est


if __name__ == "__main__":

    beginning_pos = robot.eef_pos
    beginning_quat = robot.eef_quat

    print_log("\nend-effector starting translation wrt fixed base frame: \n{}".format(beginning_pos))
    print_log("\nend-effector starting rot_matrix wrt fixed base frame: \n{}".format(T.quat2rotation(beginning_quat)))

    device = torch.device('cpu')

    model2 = torch.load('/home/yeesien/franka_ws/src/square_insertion/VSNet/YS_VSNet-train_10cm_square_28022022_log_everything-200^2*20/model.pth', map_location=device)

    model = VSNet(num_classes=7)

    model.load_state_dict(model2.module.state_dict())

    model.eval()
    model.to(device)

    robot.move_to_pose_request(
        target_pos=[beginning_pos[0], beginning_pos[1], 0.32],    # go to T0 (10 cm above hole),
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=10
    )

    eef_T0C_pos = robot.eef_pos
    eef_T0C_quat = robot.eef_quat
    print_log("\nend-effector T_0C translation: \n{}".format(eef_T0C_pos))
    print_log("\nend-effector T_0C rot_matrix: \n{}".format(T.quat2rotation(eef_T0C_quat)))

    est_errors = np.zeros(6)
    controller_errors = np.zeros(6)

    for i in range(50):
        print_log("\nAttempt {}\n".format(i+1))

        # Move to 2 random poses within the cylinder
        image_T_0A_ref, T_0A_pos_ref, T_0A_quat_ref = move_to_random_pose("A(ref:T_0A)_i_{}".format(i))
        image_T_0B, T_0B_pos, T_0B_quat = move_to_random_pose("B(test:T_0B)_i_{}".format(i))

        T_BA_pos, T_BA_quat = get_relative_pose_T_BA(model, "A(ref:T_0A)_i_{}".format(i), "B(test:T_0B)_i_{}".format(i), device, image_T_0A_ref, image_T_0B)

        T_0A_pos_est, T_0A_quat_est = T.multiply_pose(T_0B_pos, T_0B_quat, T_BA_pos, T_BA_quat)
        print_log("Estimated T_0A_pos: \n{}; \nEstimated T_0A_quat: \n{}".format(T_0A_pos_est, T_0A_quat_est))

        # Find position errors between T_0A_ref and T_0A_est
        
        T_0A_x_error = np.abs(T_0A_pos_est[0] - T_0A_pos_ref[0]) * 1000
        T_0A_y_error = np.abs(T_0A_pos_est[1] - T_0A_pos_ref[1]) * 1000
        T_0A_z_error = np.abs(T_0A_pos_est[2] - T_0A_pos_ref[2]) * 1000

        # Find rotation errors (roll, pitch, yaw) between T_0A_ref and T_0A_est
        T_0A_ref_rpy = np.array(euler_from_quaternion(T_0A_quat_ref))
        T_0A_est_rpy = np.array(euler_from_quaternion(T_0A_quat_est))

        # Convert r, p, y angles to between -pi and pi
        for i in range(len(T_0A_ref_rpy)):
            if T_0A_ref_rpy[i] > math.pi:
                T_0A_ref_rpy[i] -= (2 * math.pi)
            elif T_0A_ref_rpy[i] < -math.pi:
                T_0A_ref_rpy[i] += (2 * math.pi)

        for i in range(len(T_0A_est_rpy)):
            if T_0A_est_rpy[i] > math.pi:
                T_0A_est_rpy[i] -= (2 * math.pi)
            elif T_0A_est_rpy[i] < -math.pi:
                T_0A_est_rpy[i] += (2 * math.pi)

        print("T_0A_ref_rpy: \n{}\n".format(T_0A_ref_rpy))
        print("T_0A_est_rpy: \n{}\n".format(T_0A_est_rpy))

        T_0A_roll_error = np.rad2deg(np.abs(T_0A_est_rpy[0] - T_0A_ref_rpy[0]))
        if T_0A_roll_error > 180:
            T_0A_roll_error = np.abs(T_0A_roll_error - 360)

        T_0A_pitch_error = np.rad2deg(np.abs(T_0A_est_rpy[1] - T_0A_ref_rpy[1]))
        if T_0A_pitch_error > 180:
            T_0A_pitch_error = np.abs(T_0A_pitch_error - 360)

        T_0A_yaw_error = np.rad2deg(np.abs(T_0A_est_rpy[2] - T_0A_ref_rpy[2]))
        if T_0A_yaw_error > 180:
            T_0A_yaw_error = np.abs(T_0A_yaw_error - 360)

        # print_log all the errors above (model error)
        print_log("Model estimation errors: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error))

        # Move to estimated T_0A
        achieved_T_0A_pos_est, achieved_T_0A_quat_est = move_to_T_0A_est(T_0A_pos_est, T_0A_quat_est)

        time.sleep(1)

        # get achieved_T_0A_pos_quat_est and compare it to T_0A_pos_quat_est, the difference is controller error
        # controller position error
        controller_x_error = np.abs(achieved_T_0A_pos_est[0] - T_0A_pos_est[0]) * 1000
        controller_y_error = np.abs(achieved_T_0A_pos_est[1] - T_0A_pos_est[1]) * 1000
        controller_z_error = np.abs(achieved_T_0A_pos_est[2] - T_0A_pos_est[2]) * 1000

        # controller roll, pitch, yaw error
        achieved_T_0A_rpy_est = np.array(euler_from_quaternion(achieved_T_0A_quat_est))

        # Convert r, p, y angles to between -pi and pi
        for i in range(len(achieved_T_0A_rpy_est)):
            if achieved_T_0A_rpy_est[i] > math.pi:
                achieved_T_0A_rpy_est[i] -= (2 * math.pi)
            elif achieved_T_0A_rpy_est[i] < -math.pi:
                achieved_T_0A_rpy_est[i] += (2 * math.pi)

        print("achieved_T_0A_rpy_est: \n{}\n".format(achieved_T_0A_rpy_est))
        controller_roll_error = np.rad2deg(np.abs(achieved_T_0A_rpy_est[0] - T_0A_est_rpy[0]))
        if controller_roll_error > 180:
            controller_roll_error = np.abs(controller_roll_error - 360)

        controller_pitch_error = np.rad2deg(np.abs(achieved_T_0A_rpy_est[1] - T_0A_est_rpy[1]))
        if controller_pitch_error > 180:
            controller_pitch_error = np.abs(controller_pitch_error - 360)
    
        controller_yaw_error = np.rad2deg(np.abs(achieved_T_0A_rpy_est[2] - T_0A_est_rpy[2]))
        if controller_yaw_error > 180:
            controller_yaw_error = np.abs(controller_yaw_error - 360)

        # print_log controller errors
        print_log("Controller errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error))

        # Find average estimated errors across 50 attempts
        est_errors[0] += T_0A_x_error
        est_errors[1] += T_0A_y_error
        est_errors[2] += T_0A_z_error
        est_errors[3] += T_0A_roll_error
        est_errors[4] += T_0A_pitch_error
        est_errors[5] += T_0A_yaw_error

        # Find average controller errors across 50 attempts
        controller_errors[0] += controller_x_error
        controller_errors[1] += controller_y_error
        controller_errors[2] += controller_z_error
        controller_errors[3] += controller_roll_error
        controller_errors[4] += controller_pitch_error
        controller_errors[5] += controller_yaw_error

    ave_est_errors = est_errors/50
        
    print_log("Average estimated errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(*ave_est_errors))

    ave_controller_errors = controller_errors/50
    print_log("Average controller errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(*ave_controller_errors))