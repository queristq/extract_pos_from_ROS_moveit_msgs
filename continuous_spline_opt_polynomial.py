#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# Joint limits (rad/s and rad/s^2)
q_dot_max = np.array([3.0543, 2.7576, 3.0543, 4.0, 4.3633, 5.0])
q_ddot_max = np.array([10.0, 1.5, 1.34, 17.0, 14.0, 17.0])

def optimize_dt_for_window(q_window, dt_init, dt_res=0.004, tol=1e-4, max_iter=50):
    """
    For a given window of 5 waypoints (for all joints),
    find a time interval dt such that when using a 4th–order polynomial 
    (with time stamps [0, dt, 2*dt, 3*dt, 4*dt]) the joint velocity/acceleration 
    profiles satisfy the constraints.
    
    If any joint exceeds its velocity or acceleration limit, dt is increased.
    If all joints are comfortably below 80% of their limits, dt is reduced.
    
    Returns:
      dt_opt, max_vel_all, max_acc_all
    """
    dt = dt_init
    num_joints = q_window.shape[1]
    for iteration in range(max_iter):
        # Define times for the 5 waypoints in the window.
        t_points = np.array([0, dt, 2*dt, 3*dt, 4*dt])
        # Time samples for evaluation (every dt_res seconds)
        t_sample = np.arange(0, 4*dt + dt_res, dt_res)
        all_ok = True      # flag: no joint exceeds its limit
        all_comfortable = True  # flag: all joints are close to 80% of limit
        max_vel_all = np.zeros(num_joints)
        max_acc_all = np.zeros(num_joints)
        
        for j in range(num_joints):
            # Fit a 4th order polynomial exactly through the 5 points for joint j.
            poly_coeff = np.polyfit(t_points, q_window[:, j], 4)
            # Compute derivative (velocity) and second derivative (acceleration) polynomials.
            poly_d = np.polyder(poly_coeff)
            poly_dd = np.polyder(poly_d)
            # Evaluate velocity and acceleration along the trajectory.
            vel = np.polyval(poly_d, t_sample)
            acc = np.polyval(poly_dd, t_sample)
            max_v = np.max(np.abs(vel))
            max_a = np.max(np.abs(acc))
            max_vel_all[j] = max_v
            max_acc_all[j] = max_a
            
            # Check if limits are violated.
            if max_v > q_dot_max[j] or max_a > q_ddot_max[j]:
                all_ok = False
            # Check if the motion is too conservative (i.e. well below 80% of the limits)
            if max_v >= 0.9 * q_dot_max[j] or max_a >= 0.9 * q_ddot_max[j]:
                all_comfortable = False
        
        # Adjust dt according to the criteria:
        if not all_ok:
            dt_new = dt * 1.1  # Increase dt to slow down and reduce derivatives.
        elif all_comfortable:
            dt_new = dt * 0.9  # Decrease dt to speed up the motion.
        else:
            # We are in a “sweet‐spot”: no violation and at least one joint near 80% of its limit.
            break
        
        # If the change is very small, stop iterating.
        if abs(dt_new - dt) < tol:
            dt = dt_new
            break
        dt = dt_new

    return dt, max_vel_all, max_acc_all

def catmull_rom_spline(points, times, dt_res=0.004):
    """
    Given a set of waypoints (points) with nonuniform time stamps (times),
    generate a smooth interpolation using a Catmull–Rom (cubic Hermite) spline.
    
    Slopes at interior points are estimated by finite differences.
    
    Returns:
      t_fine: time samples (from times[0] to times[-1])
      p_fine: interpolated positions at t_fine
    """
    n = len(points)
    m = np.zeros(n)
    # Estimate slopes at endpoints.
    m[0] = (points[1] - points[0]) / (times[1] - times[0])
    m[-1] = (points[-1] - points[-2]) / (times[-1] - times[-2])
    # Interior slopes.
    for i in range(1, n-1):
        m[i] = (points[i+1] - points[i-1]) / (times[i+1] - times[i-1])
    
    t_fine_list = []
    p_fine_list = []
    # For each segment, compute the Hermite interpolation.
    for i in range(n-1):
        t_seg = np.arange(times[i], times[i+1], dt_res)
        if t_seg.size == 0 or t_seg[-1] < times[i+1]:
            t_seg = np.append(t_seg, times[i+1])
        s = (t_seg - times[i]) / (times[i+1] - times[i])
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        p_seg = (h00 * points[i] +
                 h10 * (times[i+1] - times[i]) * m[i] +
                 h01 * points[i+1] +
                 h11 * (times[i+1] - times[i]) * m[i+1])
        t_fine_list.append(t_seg)
        p_fine_list.append(p_seg)
    
    t_fine = np.concatenate(t_fine_list)
    p_fine = np.concatenate(p_fine_list)
    return t_fine, p_fine

def main():
    # ----- Step 1. Load the waypoints -----
    df = pd.read_csv("waypoints.csv")
    # Expecting 6 columns: joint_1, joint_2, ..., joint_6.
    waypoints = df.values  # shape (n_points, 6)
    n_points, num_joints = waypoints.shape

    # ----- Settings -----
    dt_init = 0.5      # initial interval (sec)
    dt_res = 0.004     # time resolution for evaluation (sec)
    window_size = 5    # number of waypoints per sliding window
    log_lines = []     # to collect log information

    # For each trajectory segment (between consecutive waypoints), we will store dt's
    segment_dts = {i: [] for i in range(n_points - 1)}
    
    # Also record the optimal (relative) time stamp for the third waypoint in each window.
    window_third_timestamps = {}

    # ----- Steps 3–8. Process sliding windows and optimize dt -----
    # For each window of 5 consecutive waypoints:
    for i in range(n_points - window_size + 1):
        q_window = waypoints[i:i+window_size, :]  # shape (5, 6)
        dt_opt, max_vel, max_acc = optimize_dt_for_window(q_window, dt_init, dt_res=dt_res)
        # The optimal time stamp for the third waypoint in this window (relative to window start) is:
        t_third = 2 * dt_opt
        log_lines.append(
            f"Window {i:2d}–{i+window_size-1:2d}: dt_opt = {dt_opt:.4f}, "
            f"optimal 3rd waypoint time = {t_third:.4f}, "
            f"max_vel = {np.array2string(max_vel, precision=3)}, "
            f"max_acc = {np.array2string(max_acc, precision=3)}"
        )
        window_third_timestamps[i+2] = t_third  # window's third waypoint index

        # In a window of 5 waypoints, there are 4 segments (i->i+1, …, i+3->i+4).
        for seg in range(i, i + window_size - 1):
            segment_dts[seg].append(dt_opt)

    # ----- Average the dt's for each segment -----
    dt_segments = []
    for seg in range(n_points - 1):
        if segment_dts[seg]:
            dt_seg = np.mean(segment_dts[seg])
        else:
            dt_seg = dt_init  # fallback if no window covers the segment (should not happen)
        dt_segments.append(dt_seg)
    dt_segments = np.array(dt_segments)

    # ----- Compute final (nonuniform) time stamps for each waypoint -----
    time_stamps = np.zeros(n_points)
    for i in range(1, n_points):
        time_stamps[i] = time_stamps[i - 1] + dt_segments[i - 1]
    log_lines.append("Final time stamps for waypoints:")
    log_lines.append(np.array2string(time_stamps, precision=4))
    # Also log the optimal 3rd waypoint timestamps from each window for debugging.
    log_lines.append("Optimal relative timestamps for the 3rd waypoint in each window:")
    for idx, t_third in window_third_timestamps.items():
        log_lines.append(f"Waypoint index {idx}: {t_third:.4f} (relative to window start)")

    # ----- Define a common fine time grid -----
    t_fine = np.arange(time_stamps[0], time_stamps[-1] + dt_res, dt_res)

    # ----- Step 9. Generate final trajectories using Catmull-Rom and B-Spline -----
    traj_catmull = np.zeros((len(t_fine), num_joints))
    traj_bspline = np.zeros((len(t_fine), num_joints))
    for j in range(num_joints):
        # Catmull-Rom spline interpolation.
        t_interp, p_interp = catmull_rom_spline(waypoints[:, j], time_stamps, dt_res)
        traj_catmull[:, j] = np.interp(t_fine, t_interp, p_interp)
        # B-Spline interpolation (using cubic spline, k=3, with s=0 for interpolation).
        tck = splrep(time_stamps, waypoints[:, j], s=0, k=3)
        traj_bspline[:, j] = splev(t_fine, tck)

    # ----- Step 10. Compute joint velocities and accelerations numerically -----
    # For Catmull-Rom trajectory.
    traj_vel_catmull = np.gradient(traj_catmull, dt_res, axis=0)
    traj_acc_catmull = np.gradient(traj_vel_catmull, dt_res, axis=0)
    # For B-Spline trajectory.
    traj_vel_bspline = np.gradient(traj_bspline, dt_res, axis=0)
    traj_acc_bspline = np.gradient(traj_vel_bspline, dt_res, axis=0)

    # ----- Step 11. Visualization -----
    # 1. Positions Comparison
    plt.figure(figsize=(10, 6))
    for j in range(num_joints):
        plt.plot(t_fine, traj_catmull[:, j], label=f'Joint {j+1} Catmull', linestyle='-')
        plt.plot(t_fine, traj_bspline[:, j], label=f'Joint {j+1} B-Spline', linestyle='--')
        plt.plot(time_stamps, waypoints[:, j], 'o', markersize=5, color='k',
                 label='Waypoints' if j == 0 else None)
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Position (rad)")
    plt.title("Trajectory Comparison – Positions")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory_positions_comparison.png")
    
    # 2. Velocities Comparison (subplots per joint)
    fig_vel, axs_vel = plt.subplots(3, 2, figsize=(12, 10))
    axs_vel = axs_vel.flatten()
    for j in range(num_joints):
        axs_vel[j].plot(t_fine, traj_vel_catmull[:, j], label='Catmull', linestyle='-')
        axs_vel[j].plot(t_fine, traj_vel_bspline[:, j], label='B-Spline', linestyle='--')
        axs_vel[j].hlines(q_dot_max[j], t_fine[0], t_fine[-1], colors='k', linestyles='dashed',
                           label='v_max' if j==0 else None)
        axs_vel[j].hlines(-q_dot_max[j], t_fine[0], t_fine[-1], colors='k', linestyles='dashed')
        axs_vel[j].set_xlabel("Time (s)")
        axs_vel[j].set_ylabel("Velocity (rad/s)")
        axs_vel[j].set_title(f"Joint {j+1}")
        axs_vel[j].grid(True)
        if j == 0:
            axs_vel[j].legend(fontsize='small')
    fig_vel.suptitle("Trajectory Comparison – Velocities", fontsize=16)
    fig_vel.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("trajectory_velocities_comparison.png")
    
    # 3. Accelerations Comparison (subplots per joint)
    fig_acc, axs_acc = plt.subplots(3, 2, figsize=(12, 10))
    axs_acc = axs_acc.flatten()
    for j in range(num_joints):
        axs_acc[j].plot(t_fine, traj_acc_catmull[:, j], label='Catmull', linestyle='-')
        axs_acc[j].plot(t_fine, traj_acc_bspline[:, j], label='B-Spline', linestyle='--')
        axs_acc[j].hlines(q_ddot_max[j], t_fine[0], t_fine[-1], colors='r', linestyles='dashed',
                           label='a_max' if j==0 else None)
        axs_acc[j].hlines(-q_ddot_max[j], t_fine[0], t_fine[-1], colors='r', linestyles='dashed')
        axs_acc[j].set_xlabel("Time (s)")
        axs_acc[j].set_ylabel("Acceleration (rad/s²)")
        axs_acc[j].set_title(f"Joint {j+1}")
        axs_acc[j].grid(True)
        if j == 0:
            axs_acc[j].legend(fontsize='small')
    fig_acc.suptitle("Trajectory Comparison – Accelerations", fontsize=16)
    fig_acc.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("trajectory_accelerations_comparison.png")
    
    # Optionally display the figures.
    plt.show()
    
    # ----- Save Final Trajectories to CSV Files -----
    # Define column headers.
    columns = ['time',
               'pos_joint_1','pos_joint_2','pos_joint_3','pos_joint_4','pos_joint_5','pos_joint_6',
               'vel_joint_1','vel_joint_2','vel_joint_3','vel_joint_4','vel_joint_5','vel_joint_6',
               'acc_joint_1','acc_joint_2','acc_joint_3','acc_joint_4','acc_joint_5','acc_joint_6']
    
    # For Catmull-Rom trajectory.
    data_catmull = np.column_stack([t_fine, traj_catmull, traj_vel_catmull, traj_acc_catmull])
    df_catmull = pd.DataFrame(data_catmull, columns=columns)
    df_catmull.to_csv("final_trajectory_catmull.csv", index=False)
    
    # For B-Spline trajectory.
    data_bspline = np.column_stack([t_fine, traj_bspline, traj_vel_bspline, traj_acc_bspline])
    df_bspline = pd.DataFrame(data_bspline, columns=columns)
    df_bspline.to_csv("final_trajectory_bspline.csv", index=False)
    

    # ----- Step 12. Save log file -----
    with open("trajectory_log.txt", "w") as log_file:
        for line in log_lines:
            log_file.write(line + "\n")
    print("Trajectory optimization complete. Comparison figures and log file saved.")



if __name__ == "__main__":
    main()
