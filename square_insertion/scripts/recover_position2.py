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
from utils import normalize_q, tf_to_quat
from datetime import datetime
import os.path
import time

robot = motion_generator.MotionGenerator(verbose="debug")
file = "/../logs/log_msgs_{}.txt".format(datetime.now())
i = 0


def print_log(msg):
    print(msg)
    print(msg, file=open(os.path.dirname(__file__) + file, "a"))


def move_to_pose(pose_id, i):
    print_log("Moving to position: {}".format(pose_id))

    T_CE_label = np.loadtxt('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_04022022/peg_square_x-0.25y-0.25_0_2/label/{}'.format(i) + '.txt', dtype=np.float32)

    T_CE_label_7 = np.array(tf_to_quat(T_CE_label), dtype=np.float32)

    T_0E_pos, T_0E_quat = T.multiply_pose(eef_T0C_pos, eef_T0C_quat, T_CE_label_7[:3], T_CE_label_7[3:])

    robot.move_to_pose_request(
        target_pos=T_0E_pos,
        target_quat=T_0E_quat,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
    )

    print_log("Taking picture {}...".format(pose_id))
    image_T_02E2 = capture.capture_image(pose_id)

    T_02E2_pos = robot.eef_pos
    T_02E2_quat = robot.eef_quat

    return image_T_02E2, T_02E2_pos, T_02E2_quat

    # print_log("T_0{}\n".format(pose_id) + str(random_T[:-1]))
    # file1 = open(os.path.dirname(__file__) + "/../actual_ins_data/label/" + pose_id + ".text", "w")

    # if pose_id[0] == "A":
    #     T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
    #     print(T_0A, file=file1)
    #     return image, random_T[0:3], random_T[3:7]
    # elif pose_id[0] == "B":
    #     T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
    #     print(T_0B, file=file1) 
    #     return image, random_T[0:3], random_T[3:7]       


def get_relative_pose_T_B2A2_hat(model, img_A_name, img_B_name, device):

    # Find T_BA
    img_A = Image.open(os.path.dirname(__file__) + '/../model_testing_data/img/' + img_A_name + '.png')
    img_B = Image.open(os.path.dirname(__file__) + '/../model_testing_data/img/' + img_B_name + '.png')

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


def get_relative_pose_T_B1A1_hat(model, img_A_name, img_B_name, device):

    # Find T_BA
    img_A = Image.open('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_04022022/peg_square_x-0.25y-0.25_0_2/img/{}'.format(img_A_name) + '.png')
    img_B = Image.open('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_04022022/peg_square_x-0.25y-0.25_0_2/img/{}'.format(img_B_name) + '.png')

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


def get_T_B1A1_from_label(img_A_name, img_B_name):

    T_CA_label = np.loadtxt('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_04022022/peg_square_x-0.25y-0.25_0_2/label/{}'.format(img_A_name) + '.txt', dtype=np.float32)
    T_CB_label = np.loadtxt('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_04022022/peg_square_x-0.25y-0.25_0_2/label/{}'.format(img_B_name) + '.txt', dtype=np.float32)

    T_BA_label = np.matmul(np.linalg.inv(T_CB_label), T_CA_label)  # take a as reference
    label = np.array(tf_to_quat(T_BA_label), dtype=np.float32)

    label = torch.from_numpy(label)
    label = label.cpu().detach().numpy()

    return label[:3], label[3:]


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


def find_errors_between_2_poses(pose1_pos, pose1_quat, pose2_pos, pose2_quat):
   
    # Find position errors between pose 1 and pose 2
    x_error = np.abs(pose1_pos[0] - pose2_pos[0]) * 1000
    y_error = np.abs(pose1_pos[1] - pose2_pos[1]) * 1000
    z_error = np.abs(pose1_pos[2] - pose2_pos[2]) * 1000

    # Find rotation errors (roll, pitch, yaw) between pose 1 and pose 2
    ref_rpy = np.array(euler_from_quaternion(pose2_quat))
    est_rpy = np.array(euler_from_quaternion(pose1_quat))

    # Convert r, p, y angles to between -pi and pi
    for i in range(len(ref_rpy)):
        if ref_rpy[i] > math.pi:
            ref_rpy[i] -= (2 * math.pi)
        elif ref_rpy[i] < -math.pi:
            ref_rpy[i] += (2 * math.pi)

    for i in range(len(est_rpy)):
        if est_rpy[i] > math.pi:
            est_rpy[i] -= (2 * math.pi)
        elif est_rpy[i] < -math.pi:
            est_rpy[i] += (2 * math.pi)

    # print("ref_rpy: \n{}\n".format(T_0A_ref_rpy))
    # print("est_rpy: \n{}\n".format(T_0A_est_rpy))

    roll_error = np.rad2deg(np.abs(est_rpy[0] - ref_rpy[0]))
    if roll_error > 180:
        roll_error = np.abs(roll_error - 360)

    pitch_error = np.rad2deg(np.abs(est_rpy[1] - ref_rpy[1]))
    if pitch_error > 180:
        pitch_error = np.abs(pitch_error - 360)

    yaw_error = np.rad2deg(np.abs(est_rpy[2] - ref_rpy[2]))
    if yaw_error > 180:
        yaw_error = np.abs(yaw_error - 360)

    return x_error, y_error, z_error, roll_error, pitch_error, yaw_error    


if __name__ == "__main__":

    beginning_pos = robot.eef_pos
    beginning_quat = robot.eef_quat

    print_log("\nend-effector starting translation wrt fixed base frame: \n{}".format(beginning_pos))
    print_log("\nend-effector starting rot_matrix wrt fixed base frame: \n{}".format(T.quat2rotation(beginning_quat)))

    device = torch.device('cpu')

    # model = torch.load('/home/yeesien/franka_ws/src/round_insertion/VSNet/YS_VSNet-190train/model.pth', map_location=device)
    model2 = torch.load('/home/yeesien/franka_ws/src/square_insertion/VSNet/YS_VSNet-train_10cm_square_11022022-200^2*13/model.pth', map_location=device)

    model = VSNet(num_classes=7)

    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(model2.module.state_dict())

    model.eval()
    model.to(device)

    # T0C_pos = [0.64568156, -0.0022422036, 0.340077]
    # rotm_0C = np.zeros((3, 3), dtype=float)
    # rotm_0C[0, 0] = 0.9626807 
    # rotm_0C[0, 1] = -0.26862115
    # rotm_0C[0, 2] = -0.03299282
    # rotm_0C[1, 0] = -0.2678331  
    # rotm_0C[1, 1] = -0.96310264
    # rotm_0C[1, 2] = 0.026429297
    # rotm_0C[2, 0] = -0.038874943  
    # rotm_0C[2, 1] = -0.016606404
    # rotm_0C[2, 2] = -0.999106
    # T0C_quat = T.mat2quat(rotm_0C)

    robot.move_to_pose_request(
        target_pos=[beginning_pos[0], beginning_pos[1], 0.32],    # go to T0 (10 cm above hole),,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
    )

    eef_T0C_pos = robot.eef_pos
    eef_T0C_quat = robot.eef_quat
    print_log("\nend-effector T_0C translation: \n{}".format(eef_T0C_pos))
    print_log("\nend-effector T_0C rot_matrix: \n{}".format(T.quat2rotation(eef_T0C_quat)))

    T_B1A1_error = np.zeros(6)
    T_B2A2_error = np.zeros(6)
    T_B1A1_B2A2_hat_error = np.zeros(6)
    T_B1A1_B2A2_error = np.zeros(6)
    total_error = [T_B1A1_error, ]

    for i in range(0, 100, 2):
        
        # Move to 1.png (A:ref)
        image_T_02A2_ref, T_02A2_pos, T_02A2_quat = move_to_pose("A(ref:T_0A)_i_{}".format(i), i)
        # Move to 2.png (B:test)
        image_T_02B2_ref, T_02B2_pos, T_02B2_quat = move_to_pose("B(test:T_0B)_i_{}".format(i+1), i+1)


        # Move to 1.png (A:ref)
        # TCA_pos = [0.002441287, 0.0005841404, -0.0042612255]
        # rotm_CA = np.zeros((3, 3), dtype=float)
        # rotm_CA[0, 0] = 0.99805784   
        # rotm_CA[0, 1] = 0.056861326
        # rotm_CA[0, 2] = 0.025441907
        # rotm_CA[1, 0] = -0.05765891    
        # rotm_CA[1, 1] = 0.9978298
        # rotm_CA[1, 2] = 0.0317981
        # rotm_CA[2, 0] = -0.02357861    
        # rotm_CA[2, 1] = -0.033203293
        # rotm_CA[2, 2] = 0.9991704
        # TCA_quat = T.mat2quat(rotm_CA)

        # T_0A_pos, T_0A_quat = T.multiply_pose(eef_T0C_pos, eef_T0C_quat, TCA_pos, TCA_quat)

        # robot.move_to_pose_request(
        #     target_pos=T_0A_pos,
        #     target_quat=T_0A_quat,
        #     linear_speed=0.1,
        #     rotation_speed=0.4,
        #     max_duration=5
        # )

        # T_02A2_pos = robot.eef_pos
        # T_02A2_quat = robot.eef_quat

        # Take picture at 1.txt pose
        # print_log("Taking picture {}...".format("A(ref:T_0A)_i_{}".format(i)))
        # image_T_02A2_ref = capture.capture_image("A(ref:T_0A)_i_{}".format(i))


        # Move to 2.png (B:test)
        # TCB_pos = [-0.0069473386, 0.0010827482, -0.00023329258]
        # rotm_CB = np.zeros((3, 3), dtype=float)
        # rotm_CB[0, 0] = 0.9981678    
        # rotm_CB[0, 1] = 0.05883919
        # rotm_CB[0, 2] = -0.014105698 
        # rotm_CB[1, 0] = -0.05825081     
        # rotm_CB[1, 1] = 0.99753934
        # rotm_CB[1, 2] = 0.039014384 
        # rotm_CB[2, 0] = 0.016366564      
        # rotm_CB[2, 1] = -0.03812123
        # rotm_CB[2, 2] = 0.9991391
        # TCB_quat = T.mat2quat(rotm_CB)

        # T_0B_pos, T_0B_quat = T.multiply_pose(eef_T0C_pos, eef_T0C_quat, TCB_pos, TCB_quat)

        # robot.move_to_pose_request(
        #     target_pos=T_0B_pos,
        #     target_quat=T_0B_quat,
        #     linear_speed=0.1,
        #     rotation_speed=0.4,
        #     max_duration=5
        # )

        # T_02B2_pos = robot.eef_pos
        # T_02B2_quat = robot.eef_quat

        # # Take picture at 2.txt pose
        # print_log("Taking picture {}...".format("B(test:T_0B)_i_{}".format(i)))
        # image_T_02B2 = capture.capture_image("B(test:T_0B)_i_{}".format(i))

        inv_T_02B2 = T.pose_inverse(T_02B2_pos, T_02B2_quat)

        T_B2A2_pos, T_B2A2_quat = T.multiply_pose(inv_T_02B2[0], inv_T_02B2[1], T_02A2_pos, T_02A2_quat)

        T_B2A2_hat_pos, T_B2A2_hat_quat = get_relative_pose_T_B2A2_hat(model, "A(ref:T_0A)_i_{}".format(i), "B(test:T_0B)_i_{}".format(i+1), device)

        T_B1A1_hat_pos, T_B1A1_hat_quat = get_relative_pose_T_B1A1_hat(model, i, i+1, device)

        T_B1A1_pos, T_B1A1_quat = get_T_B1A1_from_label(i, i+1)

        T_B1A1_x_error, T_B1A1_y_error, T_B1A1_z_error, T_B1A1_r_error, T_B1A1_p_error, T_B1A1_yaw_error = find_errors_between_2_poses(T_B1A1_hat_pos, T_B1A1_hat_quat, T_B1A1_pos, T_B1A1_quat)

        T_B2A2_x_error, T_B2A2_y_error, T_B2A2_z_error, T_B2A2_r_error, T_B2A2_p_error, T_B2A2_yaw_error = find_errors_between_2_poses(T_B2A2_hat_pos, T_B2A2_hat_quat, T_B2A2_pos, T_B2A2_quat)

        T_B1A1_B2A2_hat_x_error, T_B1A1_B2A2_hat_y_error, T_B1A1_B2A2_hat_z_error, T_B1A1_B2A2_hat_r_error, T_B1A1_B2A2_hat_p_error, T_B1A1_B2A2_hat_yaw_error = find_errors_between_2_poses(T_B2A2_hat_pos, T_B2A2_hat_quat, T_B1A1_hat_pos, T_B1A1_hat_quat)

        T_B1A1_B2A2_x_error, T_B1A1_B2A2_y_error, T_B1A1_B2A2_z_error, T_B1A1_B2A2_r_error, T_B1A1_B2A2_p_error, T_B1A1_B2A2_yaw_error = find_errors_between_2_poses(T_B2A2_pos, T_B2A2_quat, T_B1A1_pos, T_B1A1_quat)

        # print_log all the errors above (model error)
        print_log("T_B1A1_hat vs T_B1A1: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_B1A1_x_error, T_B1A1_y_error, T_B1A1_z_error, T_B1A1_r_error, T_B1A1_p_error, T_B1A1_yaw_error))

        # print_log all the errors above (model error)
        print_log("T_B2A2_hat vs T_B2A2: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_B2A2_x_error, T_B2A2_y_error, T_B2A2_z_error, T_B2A2_r_error, T_B2A2_p_error, T_B2A2_yaw_error))

        # print_log all the errors above (model error)
        print_log("T_B1A1_hat vs T_B2A2_hat: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_B1A1_B2A2_hat_x_error, T_B1A1_B2A2_hat_y_error, T_B1A1_B2A2_hat_z_error, T_B1A1_B2A2_hat_r_error, T_B1A1_B2A2_hat_p_error, T_B1A1_B2A2_hat_yaw_error))

        # print_log all the errors above (model error)
        print_log("T_B1A1 vs T_B2A2: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_B1A1_B2A2_x_error, T_B1A1_B2A2_y_error, T_B1A1_B2A2_z_error, T_B1A1_B2A2_r_error, T_B1A1_B2A2_p_error, T_B1A1_B2A2_yaw_error))

        # Record the errors for average calculation
        T_B1A1_error[0] += T_B1A1_x_error
        T_B1A1_error[1] += T_B1A1_y_error
        T_B1A1_error[2] += T_B1A1_z_error
        T_B1A1_error[3] += T_B1A1_r_error
        T_B1A1_error[4] += T_B1A1_p_error
        T_B1A1_error[5] += T_B1A1_yaw_error

        T_B2A2_error[0] += T_B2A2_x_error
        T_B2A2_error[1] += T_B2A2_y_error
        T_B2A2_error[2] += T_B2A2_z_error
        T_B2A2_error[3] += T_B2A2_r_error
        T_B2A2_error[4] += T_B2A2_p_error
        T_B2A2_error[5] += T_B2A2_yaw_error

        T_B1A1_B2A2_hat_error[0] += T_B1A1_B2A2_hat_x_error
        T_B1A1_B2A2_hat_error[1] += T_B1A1_B2A2_hat_y_error
        T_B1A1_B2A2_hat_error[2] += T_B1A1_B2A2_hat_z_error
        T_B1A1_B2A2_hat_error[3] += T_B1A1_B2A2_hat_r_error
        T_B1A1_B2A2_hat_error[4] += T_B1A1_B2A2_hat_p_error
        T_B1A1_B2A2_hat_error[5] += T_B1A1_B2A2_hat_yaw_error

        T_B1A1_B2A2_error[0] += T_B1A1_B2A2_x_error
        T_B1A1_B2A2_error[1] += T_B1A1_B2A2_y_error
        T_B1A1_B2A2_error[2] += T_B1A1_B2A2_z_error
        T_B1A1_B2A2_error[3] += T_B1A1_B2A2_r_error
        T_B1A1_B2A2_error[4] += T_B1A1_B2A2_p_error
        T_B1A1_B2A2_error[5] += T_B1A1_B2A2_yaw_error


    T_B1A1_ave_error = T_B1A1_error/50
    T_B2A2_ave_error = T_B2A2_error/50
    T_B1A1_B2A2_hat_ave_error = T_B1A1_B2A2_hat_error/50
    T_B1A1_B2A2_ave_error = T_B1A1_B2A2_error/50

    # print_log average errors
    print_log("T_B1A1_hat vs T_B1A1 average errors: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(*T_B1A1_ave_error))

    print_log("T_B2A2_hat vs T_B2A2 average errors: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(*T_B2A2_ave_error))

    print_log("T_B1A1_hat vs T_B2A2_hat average errors: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(*T_B1A1_B2A2_hat_ave_error))

    print_log("T_B1A1 vs T_B2A2 average errors: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(*T_B1A1_B2A2_ave_error))