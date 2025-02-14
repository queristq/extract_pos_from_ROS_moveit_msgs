import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

# =====================================
# 1. Load Waypoints Data from CSV File
# =====================================
# Ensure "pickup_plan_waypoints.csv" is in your working directory.
df = pd.read_csv("waypoints.csv")
q = df.values  # Shape: (N, 6) where N is the number of waypoints.
N, num_joints = q.shape

# =====================================
# 2. Define Joint Acceleration Limits
# =====================================
# Example acceleration limits for each joint.
# Adjust these to match your robot's actual limits.
acc_limits = np.array([5.0, 2.0, 2.0, 5.0, 5.0, 8.0])  # [rad/s²]

# ======================================================
# 3. Define the Optimization Problem and Helper Functions
# ======================================================
# The decision variables are the time intervals (dt) between waypoints.
# Total trajectory time T = sum(dt).

def total_time(dt):
    """Objective: minimize the total trajectory duration."""
    return np.sum(dt)

def acceleration_violations(dt):
    """
    For each interior waypoint (i = 1, ..., N-2) and for each joint j,
    compute the finite-difference acceleration using:
        v_prev = (q[i] - q[i-1]) / (t[i]-t[i-1])   with t differences = dt[i-1]
        v_next = (q[i+1] - q[i]) / (t[i+1]-t[i])     with t differences = dt[i]
        a = (v_next - v_prev) / (0.5*(dt[i-1] + dt[i]))
    Returns: |a| - acc_limits[j] for each (i, j) [must be <= 0].
    """
    dt = np.asarray(dt)
    violations = []
    for i in range(1, N - 1):
        for j in range(num_joints):
            v_prev = (q[i, j] - q[i - 1, j]) / dt[i - 1]
            v_next = (q[i + 1, j] - q[i, j]) / dt[i]
            avg_dt = 0.5 * (dt[i - 1] + dt[i])
            a = (v_next - v_prev) / avg_dt
            violations.append(np.abs(a) - acc_limits[j])
    return np.array(violations)

# Nonlinear constraint: all acceleration violations must be <= 0.
nlc = NonlinearConstraint(acceleration_violations, -np.inf, 0)

# ================================================
# 4. Initial Guess and Bounds for the Optimization
# ================================================
# Assume an initial total trajectory time (e.g., 5 seconds) with equal dt.
initial_total_time = 5.0
initial_dt = np.ones(N - 1) * (initial_total_time / (N - 1))
bounds = [(0.01, None) for _ in range(N - 1)]

# (Optional) Print initial constraint violations for debugging.
print("Initial constraint violations:", acceleration_violations(initial_dt))

# =========================================================
# 5. Run the Optimization Using SciPy's SLSQP Method
# =========================================================
res = minimize(
    total_time,
    initial_dt,
    method='SLSQP',
    bounds=bounds,
    constraints=[nlc],
    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True}
)

if res.success:
    opt_dt = res.x
    # Construct optimized time stamps:
    t_opt = np.concatenate(([0], np.cumsum(opt_dt)))
    print("Optimized total time:", t_opt[-1])
else:
    raise RuntimeError("Optimization failed: " + res.message)

# ======================================
# 6. Unified Velocity and Acceleration Calculation
# ======================================
def compute_vel_and_acc(t, q):
    """
    Compute velocities and accelerations using the same finite-difference formulas
    as in acceleration_violations.

    First, compute segment velocities:
      v_seg[i] = (q[i+1]-q[i])/(t[i+1]-t[i])  for i = 0,...,N-2.
    Then, assign velocities at waypoints:
      v[0] = v_seg[0], v[N-1] = v_seg[N-2],
      and for interior points, v[i] = 0.5*(v_seg[i-1] + v_seg[i]).
    For acceleration at interior waypoints (i = 1,...,N-2):
      a[i] = (v_seg[i] - v_seg[i-1]) / (0.5*((t[i]-t[i-1])+(t[i+1]-t[i]))).
    For endpoints, set acceleration to zero.
    """
    N = len(t)
    v_seg = np.zeros((N-1, q.shape[1]))
    for i in range(N-1):
        dt = t[i+1]-t[i]
        v_seg[i] = (q[i+1] - q[i]) / dt

    vel = np.zeros_like(q)
    vel[0] = v_seg[0]
    vel[-1] = v_seg[-1]
    for i in range(1, N-1):
        vel[i] = 0.5*(v_seg[i-1] + v_seg[i])
    
    acc = np.zeros_like(q)
    # For interior waypoints, use the unified formula:
    for i in range(1, N-1):
        dt_avg = 0.5*((t[i]-t[i-1])+(t[i+1]-t[i]))
        acc[i] = (v_seg[i] - v_seg[i-1]) / dt_avg
    # Set endpoints accelerations to zero (or use forward/backward differences if desired)
    acc[0] = 0
    acc[-1] = 0
    return vel, acc

# Compute velocities and accelerations for the optimized time stamps.
vel_opt, acc_opt = compute_vel_and_acc(t_opt, q)

# For comparison, compute baseline (equal spacing) timing.
t_equal = np.linspace(0, initial_total_time, N)
vel_equal, acc_equal = compute_vel_and_acc(t_equal, q)

# ======================================
# 7. Save Optimized Trajectory to CSV
# ======================================
# The CSV file will contain columns for time, each joint's position, velocity, and acceleration.
header = (["time"] +
          [f"pos_joint_{i+1}" for i in range(num_joints)] +
          [f"vel_joint_{i+1}" for i in range(num_joints)] +
          [f"acc_joint_{i+1}" for i in range(num_joints)])

# Stack the optimized time stamps, positions, velocities, and accelerations.
data_out = np.column_stack((t_opt, q, vel_opt, acc_opt))
trajectory_df = pd.DataFrame(data_out, columns=header)
output_filename = "optimized_joint_trajectory_with_vel_acc.csv"
trajectory_df.to_csv(output_filename, index=False)
print(f"CSV file '{output_filename}' has been generated.")

# ======================================
# 8. Visualization: Save Comparison Figures
# ======================================

# -- Position Comparison --
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2*num_joints), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, q[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, q[:, j], 's-', label='Optimized')
    axs[j].set_ylabel(f'Joint {j+1} Position')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
fig.suptitle("Joint Position Comparison", y=1.02)
plt.tight_layout()
plt.savefig("position_comparison.png")
print("Position comparison figure saved as 'position_comparison.png'")

# -- Velocity Comparison --
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2*num_joints), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, vel_equal[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, vel_opt[:, j], 's-', label='Optimized')
    axs[j].set_ylabel(f'Joint {j+1} Velocity')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
fig.suptitle("Joint Velocity Comparison", y=1.02)
plt.tight_layout()
plt.savefig("velocity_comparison.png")
print("Velocity comparison figure saved as 'velocity_comparison.png'")

# -- Acceleration Comparison --
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2*num_joints), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, acc_equal[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, acc_opt[:, j], 's-', label='Optimized')
    axs[j].axhline(acc_limits[j], color='r', linestyle='--', label='Acc limit' if j==0 else "")
    axs[j].axhline(-acc_limits[j], color='r', linestyle='--')
    axs[j].set_ylabel(f'Joint {j+1} Acceleration')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
fig.suptitle("Joint Acceleration Comparison", y=1.02)
plt.tight_layout()
plt.savefig("acceleration_comparison.png")
print("Acceleration comparison figure saved as 'acceleration_comparison.png'")
