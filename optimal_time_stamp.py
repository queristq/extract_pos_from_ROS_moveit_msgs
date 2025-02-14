import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

#reference: https://chatgpt.com/share/67ad9d50-c364-8005-9327-e46f778e50d7


# =====================================
# 1. Load Waypoints Data from CSV File
# =====================================
# Ensure "waypoints.csv" is in your working directory.
df = pd.read_csv("pickup_plan_waypoints.csv")
q = df.values  # Shape: (N, 6), where N is the number of waypoints.
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
# We use the time intervals (dt) between waypoints as decision variables.
# The total trajectory time T is sum(dt).

def total_time(dt):
    """Objective: minimize the total trajectory duration."""
    return np.sum(dt)

def acceleration_violations(dt):
    """
    Compute the violation of the acceleration limits using finite differences.
    For each interior waypoint (i = 1, ..., N-2) and each joint j:
      v_prev = (q[i,j] - q[i-1,j]) / dt[i-1]
      v_next = (q[i+1,j] - q[i,j]) / dt[i]
      a_i    = (v_next - v_prev) / (0.5*(dt[i-1] + dt[i]))
    Returns the excess amount over the limits (should be <= 0).
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

# Create a nonlinear constraint to enforce acceleration limits.
nlc = NonlinearConstraint(acceleration_violations, -np.inf, 0)

# ================================================
# 4. Initial Guess and Bounds for the Optimization
# ================================================
# Assume a baseline total trajectory time (e.g., 5 seconds) with equal spacing.
initial_total_time = 5.0
initial_dt = np.ones(N - 1) * (initial_total_time / (N - 1))

# Print initial constraint violations for debugging.
print("Initial constraint violations:", acceleration_violations(initial_dt))

# Lower bounds for dt to ensure they remain positive.
bounds = [(0.01, None) for _ in range(N - 1)]

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
    t_opt = np.concatenate(([0], np.cumsum(opt_dt)))
    print("Optimized total time:", t_opt[-1])
else:
    raise RuntimeError("Optimization failed: " + res.message)

# ================================================
# 6. Baseline: Create Equal Spacing Time Stamps
# ================================================
t_equal = np.linspace(0, initial_total_time, N)

# ========================================================
# 7. Visualization: Compare Joint Position Trajectories
# ========================================================
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 12), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, q[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, q[:, j], 's-', label='Optimized')
    axs[j].set_ylabel(f'Joint {j+1} Position')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
plt.suptitle("Joint Trajectories: Baseline vs. Optimized Timing", y=0.92, fontsize=14)
plt.tight_layout()
plt.savefig("trajectory_comparison.png")
print("Trajectory comparison plot saved as 'trajectory_comparison.png'")

# ========================================================
# 8. Compute Acceleration Profiles for Both Timings
# ========================================================
def compute_accelerations(t, q):
    """
    Compute the acceleration at interior waypoints using finite differences.
    For i = 1,...,N-2:
      v_prev = (q[i] - q[i-1]) / (t[i]-t[i-1])
      v_next = (q[i+1] - q[i]) / (t[i+1]-t[i])
      a = (v_next - v_prev) / (0.5*((t[i]-t[i-1])+(t[i+1]-t[i])))
    Returns:
      acc: Array of shape (N-2, num_joints) of accelerations.
      t_acc: Time stamps corresponding to the computed accelerations (t[1] to t[N-2]).
    """
    dt = np.diff(t)
    acc = np.zeros((len(t) - 2, q.shape[1]))
    for i in range(1, len(t) - 1):
        v_prev = (q[i] - q[i - 1]) / dt[i - 1]
        v_next = (q[i + 1] - q[i]) / dt[i]
        avg_dt = 0.5 * (dt[i - 1] + dt[i])
        acc[i - 1, :] = (v_next - v_prev) / avg_dt
    return acc, t[1:-1]

acc_equal, t_acc_equal = compute_accelerations(t_equal, q)
acc_opt, t_acc_opt = compute_accelerations(t_opt, q)

# ========================================================
# 9. Visualization: Compare Joint Acceleration Profiles
# ========================================================
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 12), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_acc_equal, acc_equal[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_acc_opt, acc_opt[:, j], 's-', label='Optimized')
    # Plot acceleration limits for reference.
    axs[j].axhline(acc_limits[j], color='r', linestyle='--', label='Acc limit' if j == 0 else "")
    axs[j].axhline(-acc_limits[j], color='r', linestyle='--')
    axs[j].set_ylabel(f'Joint {j+1} Acceleration')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
plt.suptitle("Joint Accelerations: Baseline vs. Optimized Timing", y=0.92, fontsize=14)
plt.tight_layout()
plt.savefig("acceleration_comparison.png")
print("Acceleration comparison plot saved as 'acceleration_comparison.png'")

# ========================================================
# 10. Compute and Visualize Velocity Profiles
# ========================================================
def compute_velocities(t, q):
    """
    Compute the velocity for each segment using finite differences:
      v = (q[i+1] - q[i]) / (t[i+1]-t[i])
    Returns:
      velocities: Array of shape (N-1, num_joints).
      t_vel: Midpoint time stamps for each segment.
    """
    dt = np.diff(t)
    velocities = (q[1:] - q[:-1]) / dt[:, None]
    t_vel = 0.5 * (t[:-1] + t[1:])
    return velocities, t_vel

vel_equal, t_vel_equal = compute_velocities(t_equal, q)
vel_opt, t_vel_opt = compute_velocities(t_opt, q)

fig, axs = plt.subplots(num_joints, 1, figsize=(10, 12), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_vel_equal, vel_equal[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_vel_opt, vel_opt[:, j], 's-', label='Optimized')
    axs[j].set_ylabel(f'Joint {j+1} Velocity')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
plt.suptitle("Joint Velocities: Baseline vs. Optimized Timing", y=0.92, fontsize=14)
plt.tight_layout()
plt.savefig("velocity_comparison.png")
print("Velocity comparison plot saved as 'velocity_comparison.png'")
