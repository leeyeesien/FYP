import franka_cartesian_controller.motion_generator as motion_generator
import franka_cartesian_controller.transform_util as T
import random
import math
import capture_image_exe as capture

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
import time as mytime

robot = motion_generator.MotionGenerator(verbose="debug")
file = "/../logs_ins_exe/log_msgs_{}.txt".format(datetime.now())
i = 0


def print_log(msg):
    print(msg)
    print(msg, file=open(os.path.dirname(__file__) + file, "a"))


def move_to_random_pose(pose_id):
    print_log("Moving to random position: T_0{}".format(pose_id))

    r = random.uniform(math.radians(-5), math.radians(5))
    p = random.uniform(math.radians(-5), math.radians(5))
    y = random.uniform(math.radians(-10), math.radians(10))
    eef_TAEi_rot_mat = T.euler2mat([r, p, y])
    eef_TAEi_quat = T.mat2quat(eef_TAEi_rot_mat)
    desired_T0Ei_quat = T.quat_multiply(T_0A_quat_ref, eef_TAEi_quat)
    print_log("\nroll: {}". format(math.degrees(r)) +
            "  pitch: {}".format(math.degrees(p)) + "  yaw: {}".format(math.degrees(y)))

    x = random.uniform(-0.005, 0.005)
    y = random.uniform(-0.005, 0.005)
    z = random.uniform(0, 0.01)
    print_log("\nx: {}".format(x) + "  y: {}".format(y) + "  z: {}".format(z))

    robot.move_to_pose_request(
        # random eef position, with new pose origin within a vertical cylinder with r=5 mm & h=10 mm, with the default pose T_0 frame origin at the center of the bottom of the cylinder
        target_pos=T_0A_pos_ref + [x, y, z],
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
    file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/" + pose_id + ".text", "w")

    if pose_id[0] == "A":
        T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
        print(T_0A, file=file1)
        return image, random_T[0:3], random_T[3:7]
    elif pose_id[0] == "B":
        T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
        print(T_0B, file=file1) 
        return image, random_T[0:3], random_T[3:7]       


def get_relative_pose_T_BA(model, img_A_name, img_B_name, device, img_A, img_B):

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

    starting_eef_pos = robot.eef_pos
    starting_eef_quat = robot.eef_quat
    print_log("\nend-effector starting translation wrt fixed base frame: \n{}".format(starting_eef_pos))
    print_log("\nend-effector starting rot_matrix wrt fixed base frame: \n{}".format(T.quat2rotation(starting_eef_quat)))

    device = torch.device('cpu')

    model2 = torch.load('/home/yeesien/franka_ws/src/square_insertion/VSNet/YS_VSNet-train_10cm_square_28022022_log_everything-200^2*20/model.pth', map_location=device)

    model = VSNet(num_classes=7)

    model.load_state_dict(model2.module.state_dict())

    model.eval()
    model.to(device)

    robot.move_to_pose_request(
    target_pos=[starting_eef_pos[0], starting_eef_pos[1], 0.32],    # go to T0 (10 cm above hole)
    linear_speed=0.1,
    rotation_speed=0.4,
    max_duration=5
    )

    T_0A_pos_ref = robot.eef_pos
    T_0A_quat_ref = robot.eef_quat

    print_log("\nend-effector T_0A(ref:T_0A)_i_{} translation: \n{}".format(i, T_0A_pos_ref))
    print_log("\nend-effector T_0A(ref:T_0A)_i_{} rot_matrix: \n{}".format(i, T.quat2rotation(T_0A_quat_ref)))

    print_log("Taking picture {}...".format(i))
    image_T_0A_ref = capture.capture_image("A(ref:T_0A)_i_{}".format(i))

    T_0A_ref = []
    T_0A_ref[:3] = T_0A_pos_ref
    T_0A_ref[3:] = T_0A_quat_ref

    print_log("T_0A(ref:T_0A)_i_{}\n".format(i) + str(T_0A_ref))
    file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/A(ref:T_0A)_i_" + str(i) + ".text", "w")

    T_0A = ' '.join([str(elem) for elem in T_0A_ref[0:7]])
    print(T_0A, file=file1)

    success_count = 0
    model_est_error = np.zeros(6)
    controller_error = np.zeros(6)
    final_error = np.zeros(6)

    for i in range(50):

        if i >= 1:
            robot.move_to_pose_request(
                target_pos=T_0A_pos_ref,
                target_quat=T_0A_quat_ref,
                linear_speed=0.1,
                rotation_speed=0.4,
                max_duration=5
            )

        print_log("\nAttempt {}\n".format(i+1))

        # Move to a random pose within the cylinder
        image_T_0B, T_0B_pos, T_0B_quat = move_to_random_pose("B(test:T_0B)_i_{}".format(i))

        T_BA_pos, T_BA_quat = get_relative_pose_T_BA(model, "A(ref:T_0A)_i_0".format(i), "B(test:T_0B)_i_{}".format(i), device, image_T_0A_ref, image_T_0B)

        T_0A_pos_est, T_0A_quat_est = T.multiply_pose(T_0B_pos, T_0B_quat, T_BA_pos, T_BA_quat)
        print_log("Estimated T_0A_pos: \n{}; \nEstimated T_0A_quat: \n{}".format(T_0A_pos_est, T_0A_quat_est))

        # Find errors between T_0A_ref and T_0A_est
        T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error = find_errors_between_2_poses(T_0A_pos_est, T_0A_quat_est, T_0A_pos_ref, T_0A_quat_ref)

        # print_log all the errors above (model error)
        print_log("Model estimation errors T_0A: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error))

        model_est_error[0] += T_0A_x_error 
        model_est_error[1] += T_0A_y_error
        model_est_error[2] += T_0A_z_error
        model_est_error[3] += T_0A_roll_error
        model_est_error[4] += T_0A_pitch_error
        model_est_error[5] += T_0A_yaw_error

        # Move to estimated T_0A
        achieved_T_0A_pos_est, achieved_T_0A_quat_est = move_to_T_0A_est(T_0A_pos_est, T_0A_quat_est)

        mytime.sleep(1)

        # get achieved_T_0A_pos_quat_est and compare it to T_0A_pos_quat_est, the difference is controller error
        controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error = find_errors_between_2_poses(achieved_T_0A_pos_est, achieved_T_0A_quat_est, T_0A_pos_est, T_0A_quat_est)

        # print_log controller errors
        print_log("Controller errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error))

        controller_error[0] += controller_x_error 
        controller_error[1] += controller_y_error
        controller_error[2] += controller_z_error
        controller_error[3] += controller_roll_error
        controller_error[4] += controller_pitch_error
        controller_error[5] += controller_yaw_error

        contact_success, contact_time = robot.move_to_contact_request(
            direction=[0, 0, -1, 0, 0, 0],
            speed=0.01,
            force_thresh=10,
            max_duration=20,
        )

        print_log("Move to contact request success and time: \n")
        print_log(contact_success)
        print_log(contact_time)

        if contact_success is True:
            final_eef_pos = robot.eef_pos
            final_eef_quat = robot.eef_quat
            print_log("\nfinal end-effector translation: \n{}".format(final_eef_pos))
            print_log("\nfinal end-effector rot_matrix: \n{}".format(T.quat2rotation(final_eef_quat)))

            # Find errors between starting position and final position after visual servoing
            final_x_error, final_y_error, final_z_error, final_roll_error, final_pitch_error, final_yaw_error = find_errors_between_2_poses(final_eef_pos, final_eef_quat, starting_eef_pos, starting_eef_quat)
            
            # print_log final(before/after) errors
            print_log("Final errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(final_x_error, final_y_error, final_z_error, final_roll_error, final_pitch_error, final_yaw_error))

            final_error[0] += final_x_error 
            final_error[1] += final_y_error
            final_error[2] += final_z_error
            final_error[3] += final_roll_error
            final_error[4] += final_pitch_error
            final_error[5] += final_yaw_error

            # Estimate hole pose from model output
            hole_pose_est = []
            # x, y from top view; z calculated after contact
            hole_pose_est[:2] = T_0A_pos_est[:2]
            hole_pose_est[2] = T_0A_pos_est[2] - final_eef_pos[2]
            hole_pose_est[0] *= 1000
            hole_pose_est[1] *= 1000
            hole_pose_est[2] *= 1000
            # orientation from top view
            hole_pose_est_euler = euler_from_quaternion(T_0A_quat_est)

            # Convert r, p, y angles to between -pi and pi
            for i in range(len(hole_pose_est_euler)):
                if hole_pose_est_euler[i] > math.pi:
                    hole_pose_est_euler[i] -= (2 * math.pi)
                elif hole_pose_est_euler[i] < -math.pi:
                    hole_pose_est_euler[i] += (2 * math.pi)

            hole_pose_est[3] = np.rad2deg(hole_pose_est_euler[0])
            hole_pose_est[4] = np.rad2deg(hole_pose_est_euler[1])
            hole_pose_est[5] = np.rad2deg(hole_pose_est_euler[2])

            print_log("\n estimated hole pose wrt to base frame: \nx = {}mm, y = {}mm, z = {}mm, roll = {}deg, pitch = {}deg, yaw = {}deg".format(*hole_pose_est))        

            # Check whether the peg has fit into the hole
            # Check x direction
            check_x_success, check_x_time = robot.move_to_contact_request(
            direction=[1, 0, 0, 0, 0, 0],
            speed=0.001,
            force_thresh=10,
            max_duration=2,
            )
            # Check y direction
            check_y_success, check_y_time = robot.move_to_contact_request(
            direction=[0, 1, 0, 0, 0, 0],
            speed=0.001,
            force_thresh=10,
            max_duration=2,
            )

            if (check_x_success and check_y_success is True):
                print_log("Successful insertion!")
                success_count += 1
            #else:
                # Insert reinforcement learning manipulation primitives
                # Try how many times then exit?
                # if success is True:
                    # success_count += 1
                #else:
                    # continue

    print_log(f'Success rate: {success_count}/50')
    average_model_est_error = np.array(model_est_error)/50
    average_controller_error = np.array(controller_error)/50
    average_final_error = np.array(final_error)/50

    print_log(f'Ave model est error: {average_model_est_error}, Ave controller error: {average_controller_error}, Ave final error: {average_final_error}')