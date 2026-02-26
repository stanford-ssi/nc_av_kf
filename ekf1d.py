import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass



@dataclass
class DynamicsParams:
    g: float          # gravity [m/s^2], positive magnitude
    thrust_acc: float # upward thrust accel during burn phase [m/s^2]
    burn_time: float  # burn duration [s]
    drag_k: float     # simple quadratic drag coefficient


@dataclass
class SensorParams:
    accel_bias_true: float   # true accelerometer bias [m/s^2]
    accel_noise_std: float   # accel noise std dev [m/s^2]
    baro_noise_std: float    # barometer noise std dev [m]


@dataclass
class EkfParams:
    dt: float         # filter time step [s]
    q_z: float        # process noise variance for altitude
    q_v: float        # process noise variance for velocity
    q_ba: float       # process noise variance for accel bias
    r_baro_std: float # baro measurement noise std dev [m]
    x0: tuple         # initial state (z0, v0, b_a0)
    P0_diag: tuple    # initial covariance diagonal (Pzz, Pvv, Pba_ba)


class Rocket1DEKF:
    """
    1D vertical rocket EKF.

    State vector:
        x = [ z, v, b_a ]^T
        z   : altitude (meters, up)
        v   : vertical velocity (m/s, up)
        b_a : accelerometer bias (m/s^2) in vertical axis

    Inputs (per step):
        a_m : accelerometer measurement (m/s^2, up)
              modeled as: a_m = (z_ddot + g) + b_a + noise

    Dynamics (discrete, dt):
        z_{k+1}  = z_k + v_k * dt
        v_{k+1}  = v_k + (a_m_k - b_a_k - g) * dt
        b_a_{k+1}= b_a_k + w_ba_k   (random walk)

    Measurement:
        barometer: z_meas = z + noise
    """

    def __init__(self,
                 dt: float,
                 g: float,
                 Q_diag: np.ndarray,
                 R_baro_var: float,
                 x0: np.ndarray,
                 P0: np.ndarray):
        """
        dt        : time step [s]
        g         : gravity magnitude [m/s^2] (positive)
        Q_diag    : 3-element array for process noise diagonal [q_z, q_v, q_ba]
        R_baro_var: baro noise variance [m^2]
        x0        : initial state vector [z0, v0, b_a0]
        P0        : initial covariance matrix (3x3)
        """
        self.dt = dt
        self.g = g
        self.n = 3

        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

        self.Q = np.diag(Q_diag)
        self.R_baro = np.array([[R_baro_var]])

        self.H_baro = np.array([[1.0, 0.0, 0.0]])

    def predict(self, a_m: float):
        """
        EKF prediction step using accelerometer measurement a_m.
        """
        dt = self.dt

        z, v, b_a = self.x

        z_pred = z + v * dt
        v_pred = v + (a_m - b_a - self.g) * dt
        b_a_pred = b_a  
        x_pred = np.array([z_pred, v_pred, b_a_pred])

        F = np.array([
            [1.0, dt, 0.0],
            [0.0, 1.0, -dt],
            [0.0, 0.0, 1.0]
        ])

        P_pred = F @ self.P @ F.T + self.Q

        self.x = x_pred
        self.P = P_pred

    def update_baro(self, z_meas: float):
        """
        EKF update step using barometric altitude measurement.
        Returns NIS for validation.
        """
        z = np.array([[z_meas]])  # shape (1,1)
        z_hat = self.H_baro @ self.x.reshape(-1, 1)  # predicted measurement
        y = z - z_hat  # innovation

        S = self.H_baro @ self.P @ self.H_baro.T + self.R_baro  # (1x1)

        K = self.P @ self.H_baro.T @ np.linalg.inv(S)  # (3x1)

        self.x = (self.x.reshape(-1, 1) + K @ y).flatten()

        I = np.eye(self.n)
        self.P = (I - K @ self.H_baro) @ self.P

        nis = float(y.T @ np.linalg.inv(S) @ y)
        return nis



def simulate_rocket_truth_1d(t: np.ndarray, dyn: DynamicsParams):
    """
    Simple 1D rocket model for 'true' trajectory.

    Model:
        - Constant thrust phase (accel up)
        - Then coast under gravity + drag-ish term.

    Returns:
        z_true, v_true, a_spec_true
        where:
          z_true       : altitude
          v_true       : vertical velocity
          a_spec_true  : specific force along +z (what ideal IMU 'sees')
                         = z_ddot + g
    """
    dt = t[1] - t[0]
    N = len(t)

    z = np.zeros(N)
    v = np.zeros(N)
    a_spec = np.zeros(N)

    for k in range(1, N):
        if t[k] <= dyn.burn_time:
            a_inertial = dyn.thrust_acc - dyn.g - dyn.drag_k * v[k-1] * abs(v[k-1])
        else:
            a_inertial = -dyn.g - dyn.drag_k * v[k-1] * abs(v[k-1])

        v[k] = v[k-1] + a_inertial * dt
        z[k] = z[k-1] + v[k-1] * dt + 0.5 * a_inertial * dt**2

        a_spec[k] = a_inertial + dyn.g

    return z, v, a_spec


def generate_measurements_1d(z_true: np.ndarray,
                             a_spec_true: np.ndarray,
                             sensors: SensorParams,
                             rng: np.random.Generator):
    """
    Generate noisy accelerometer and barometer measurements.
    """
    N = len(z_true)

    a_m = a_spec_true + sensors.accel_bias_true + \
          rng.normal(0.0, sensors.accel_noise_std, size=N)

    z_baro = z_true + rng.normal(0.0, sensors.baro_noise_std, size=N)

    return a_m, z_baro


def make_ekf(ekf_params: EkfParams, dyn: DynamicsParams) -> Rocket1DEKF:
    """
    function to construct the EKF from parameter structs.
    """
    Q_diag = np.array([ekf_params.q_z, ekf_params.q_v, ekf_params.q_ba])
    R_baro_var = ekf_params.r_baro_std**2
    x0 = np.array(ekf_params.x0, dtype=float)
    P0 = np.diag(ekf_params.P0_diag)

    return Rocket1DEKF(
        dt=ekf_params.dt,
        g=dyn.g,
        Q_diag=Q_diag,
        R_baro_var=R_baro_var,
        x0=x0,
        P0=P0
    )



def run_single_sim_1d(ekf_params: EkfParams,
                      dyn: DynamicsParams,
                      sensors: SensorParams,
                      t_end: float,
                      seed: int = 42,
                      plot: bool = True):
    """
    Run a single 1D EKF simulation with given parameters.
    """
    dt = ekf_params.dt
    t = np.arange(0.0, t_end, dt)
    N = len(t)

    z_true, v_true, a_spec_true = simulate_rocket_truth_1d(t, dyn)

    rng = np.random.default_rng(seed)
    a_m, z_baro = generate_measurements_1d(z_true, a_spec_true, sensors, rng)

    ekf = make_ekf(ekf_params, dyn)

    z_est = np.zeros(N)
    v_est = np.zeros(N)
    ba_est = np.zeros(N)
    nis_list = []

    for k in range(N):
        ekf.predict(a_m[k])
        nis = ekf.update_baro(z_baro[k])
        nis_list.append(nis)
        z_est[k], v_est[k], ba_est[k] = ekf.x

    nis_arr = np.array(nis_list)

    print("Single-run final state estimate (1D):")
    print(f"  z_est = {z_est[-1]:.2f} m (true {z_true[-1]:.2f} m)")
    print(f"  v_est = {v_est[-1]:.2f} m/s (true {v_true[-1]:.2f} m/s)")
    print(f"  b_a_est = {ba_est[-1]:.3f} m/s^2 (true {sensors.accel_bias_true:.3f} m/s^2)")
    print()
    print("NIS stats (1D baro measurement):")
    print(f"  mean NIS ~ {nis_arr.mean():.3f}, std ~ {nis_arr.std():.3f}")
    print("  Expected 95% NIS range (chi-square(1)) ~ [0.004, 3.84]")
    frac_outside = np.mean((nis_arr < 0.004) | (nis_arr > 3.84))
    print(f"  Fraction of NIS outside [0.004, 3.84]: {frac_outside:.3f}")

    if plot:
        # Altitude
        plt.figure()
        plt.plot(t, z_true, label="z true")
        plt.plot(t, z_est, label="z EKF")
        plt.xlabel("Time [s]")
        plt.ylabel("Altitude [m]")
        plt.title("Altitude: Truth vs EKF")
        plt.legend()
        plt.grid(True)

        # Velocity
        plt.figure()
        plt.plot(t, v_true, label="v true")
        plt.plot(t, v_est, label="v EKF")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title("Velocity: Truth vs EKF")
        plt.legend()
        plt.grid(True)

        # Bias
        plt.figure()
        plt.plot(t, np.ones_like(t)*sensors.accel_bias_true, label="bias true")
        plt.plot(t, ba_est, label="bias EKF")
        plt.xlabel("Time [s]")
        plt.ylabel("Accel bias [m/s^2]")
        plt.title("Accelerometer Bias Estimation")
        plt.legend()
        plt.grid(True)

        # NIS time series
        plt.figure()
        plt.plot(t, nis_arr, label="NIS")
        plt.axhline(3.84, linestyle="--", label="95% upper bound (chi-square(1))")
        plt.xlabel("Time [s]")
        plt.ylabel("NIS")
        plt.title("Normalized Innovation Squared (Baro)")
        plt.legend()
        plt.grid(True)

    



def run_monte_carlo_1d(ekf_params: EkfParams,
                       dyn: DynamicsParams,
                       sensors: SensorParams,
                       t_end: float,
                       n_runs: int = 50):
    """
    Monte Carlo validation for the 1D EKF.
    Prints RMSE statistics and NIS consistency.
    """
    dt = ekf_params.dt
    t = np.arange(0.0, t_end, dt)
    N = len(t)

    z_true, v_true, a_spec_true = simulate_rocket_truth_1d(t, dyn)

    rng = np.random.default_rng(123)

    z_rmse_runs = []
    v_rmse_runs = []
    ba_error_final = []
    nis_all = []

    for run in range(n_runs):
        noise_seed = rng.integers(0, 1_000_000)
        rng_run = np.random.default_rng(noise_seed)

        a_m, z_baro = generate_measurements_1d(z_true, a_spec_true, sensors, rng_run)

      
        ekf = make_ekf(ekf_params, dyn)

        z_est = np.zeros(N)
        v_est = np.zeros(N)
        ba_est = np.zeros(N)
        nis_list = []

        for k in range(N):
            ekf.predict(a_m[k])
            nis = ekf.update_baro(z_baro[k])
            nis_list.append(nis)
            z_est[k], v_est[k], ba_est[k] = ekf.x

        nis_arr = np.array(nis_list)
        nis_all.append(nis_arr)

      
        z_rmse = np.sqrt(np.mean((z_est - z_true)**2))
        v_rmse = np.sqrt(np.mean((v_est - v_true)**2))
        z_rmse_runs.append(z_rmse)
        v_rmse_runs.append(v_rmse)

        ba_error_final.append(ba_est[-1] - sensors.accel_bias_true)

    z_rmse_runs = np.array(z_rmse_runs)
    v_rmse_runs = np.array(v_rmse_runs)
    ba_error_final = np.array(ba_error_final)
    nis_all = np.concatenate(nis_all)

    print("\nMonte Carlo results (1D EKF):")
    print(f"  Runs: {n_runs}")
    print(f"  Altitude RMSE: mean={z_rmse_runs.mean():.2f} m, "
          f"std={z_rmse_runs.std():.2f} m")
    print(f"  Velocity RMSE: mean={v_rmse_runs.mean():.2f} m/s, "
          f"std={v_rmse_runs.std():.2f} m/s")
    print(f"  Final accel bias error: mean={ba_error_final.mean():.3f}, "
          f"std={ba_error_final.std():.3f} m/s^2")

    print("\nNIS statistics across all runs:")
    print(f"  mean NIS ~ {nis_all.mean():.3f}, std ~ {nis_all.std():.3f}")
    frac_outside = np.mean((nis_all < 0.004) | (nis_all > 3.84))
    print(f"  Fraction outside [0.004, 3.84]: {frac_outside:.3f}")

if __name__ == "__main__":
    dyn_params = DynamicsParams(
        g=9.81,
        thrust_acc=20000.0,
        burn_time=0.5,
        drag_k=0.02
    ) #Only for simulation

    sensor_params = SensorParams(
        accel_bias_true=0.3,     # true bias [m/s^2]
        accel_noise_std=0.5,     # accel noise [m/s^2]
        baro_noise_std=5.0       # baro noise [m]
    )

    ekf_params = EkfParams(
        dt=0.01,                 # time step [s]
        q_z=1e-5,                # process noise on altitude
        q_v=1e-3,                # process noise on velocity
        q_ba=1e-8,               # process noise on bias (small -> slow drift)
        r_baro_std=5.0,          # assumed baro noise [m]
        x0=(0.0, 0.0, 0.0),      # initial state [z0, v0, b_a0]
        P0_diag=(100.0, 100.0, 1.0)  # initial covariance diagonal
    )

    T_END = 10.0
    run_monte_carlo_1d(ekf_params, dyn_params, sensor_params,
                       t_end=T_END, n_runs=50)
    run_single_sim_1d(ekf_params, dyn_params, sensor_params,
                      t_end=T_END, seed=42, plot=True)
    plt.show()