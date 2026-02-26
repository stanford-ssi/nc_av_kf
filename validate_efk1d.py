import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ekf1d import EkfParams, DynamicsParams, make_ekf


def load_log028(path: str = "LOG028.TXT") -> pd.DataFrame:
    """
    Load LOG028.TXT, which has a header separated by tabs and
    data rows separated by commas.

    We use a regex separator to split on either:
      - commas (with optional surrounding spaces), or
      - tabs.
    """
    df = pd.read_csv(
        path,
        sep=r"\s*,\s*|\t+",  # split on commas OR tabs
        engine="python"
    )

    df.columns = df.columns.str.strip()

    print("Parsed columns in LOG028.TXT:")
    print(df.columns.tolist())

    required_cols = ["Time", "Yg", "Altitude"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Log file missing required column: {col}\n"
                f"Available columns: {df.columns.tolist()}"
            )

    return df

def build_time_and_measurements(df: pd.DataFrame, g: float = 9.81):
    """
    Build:
      t      : time array [s], starting at 0
      a_m    : vertical specific force [m/s^2]
      z_meas : altitude [m] (from Altitude column)

    Assumptions:
      - Time is in milliseconds or seconds.
      - Yg is accelerometer in 'g' units, with ~1g at rest.
      - We treat Yg as the vertical axis.
      - Altitude is already in meters AGL.
    """
    # ----- Time handling -----
    time_raw = df["Time"].to_numpy().astype(float)
    time_raw = time_raw - time_raw[0]

    dt_raw = np.mean(np.diff(time_raw))
    if dt_raw > 1.0:
        # likely milliseconds
        t = time_raw / 1000.0
    else:
        # already in seconds
        t = time_raw

    # Yg is in g units; to get m/s^2:
    #   a_m = Yg * g
    # This is the "specific force" (what an ideal accel reports).
    y_g = df["Yg"].to_numpy()
    a_m = y_g * g

    z_meas = df["Altitude"].to_numpy()

    return t, a_m, z_meas


def run_ekf_on_log028(
    ekf_params: EkfParams,
    dyn_params: DynamicsParams,
    log_path: str = "LOG028.TXT",
    t_start: float = None,
    t_end: float = None
):
    """
    Use your modular 1D EKF (EkfParams + DynamicsParams + make_ekf)
    on the LOG028.TXT file.

    - Time -> from 'Time' column (auto ms/seconds)
    - a_m  -> from 'Yg' in g units (converted to m/s^2)
    - z_meas -> from 'Altitude' [m]
    """
    df = load_log028(log_path)
    t, a_m, z_meas = build_time_and_measurements(df, g=dyn_params.g)

    dt = float(np.mean(np.diff(t)))
    print(f"Log dt ~ {dt:.6f} s, samples: {len(t)}")
    ekf_params.dt = dt

    ekf = make_ekf(ekf_params, dyn_params)

    N = len(t)
    z_est = np.zeros(N)
    v_est = np.zeros(N)
    ba_est = np.zeros(N)
    nis_list = []

    for k in range(N):
        ekf.predict(a_m[k])          # accelerometer prediction
        nis = ekf.update_baro(z_meas[k])  # altitude measurement update
        nis_list.append(nis)
        z_est[k], v_est[k], ba_est[k] = ekf.x

    nis_arr = np.array(nis_list)

    if t_start is not None or t_end is not None:
        mask = np.ones_like(t, dtype=bool)

    if t_start is not None:
        mask &= (t >= t_start)
    if t_end is not None:
        mask &= (t <= t_end)

        t_plot = t[mask]
        z_plot = z_est[mask]
        v_plot = v_est[mask]
        ba_plot = ba_est[mask]
        z_meas_plot = z_meas[mask]
        nis_plot = nis_arr[mask]
    else:
        t_plot = t
        z_plot = z_est
        v_plot = v_est
        ba_plot = ba_est
        z_meas_plot = z_meas
        nis_plot = nis_arr

    alt_resid = z_est - z_meas
    alt_rmse = np.sqrt(np.mean(alt_resid**2))
    
    print("\n==== EKF vs LOG028.TXT: 1D Vertical ====")
    print(f"Altitude RMSE vs Altitude column: {alt_rmse:.2f} m")
    print(f"Mean NIS (1D baro): {nis_arr.mean():.3f}, std: {nis_arr.std():.3f}")
    frac_outside_95 = np.mean((nis_arr < 0.004) | (nis_arr > 3.84))
    print(f"Fraction of NIS outside [0.004, 3.84] (95% chi-square(1)): "
          f"{frac_outside_95:.3f}")

    # Altitude: EKF vs measured Altitude
    plt.figure()
    plt.plot(t_plot, z_meas_plot, label="Altitude (log) [m]")
    plt.plot(t_plot, z_plot, label="EKF altitude [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("LOG028: Altitude (EKF vs Logged Altitude)")
    plt.legend()
    plt.grid(True)

    # Velocity estimate
    plt.figure()
    plt.plot(t_plot, v_plot, label="EKF vertical velocity [m/s]")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("LOG028: EKF Vertical Velocity")
    plt.legend()
    plt.grid(True)

    # Bias estimate
    plt.figure()
    plt.plot(t_plot, ba_plot, label="EKF accel bias [m/s^2]")
    plt.xlabel("Time [s]")
    plt.ylabel("Bias [m/s^2]")
    plt.title("LOG028: EKF Accelerometer Bias Estimate")
    plt.legend()
    plt.grid(True)

    # NIS time series
    plt.figure()
    plt.plot(t_plot, nis_plot, label="NIS (Altitude)")
    plt.axhline(3.84, linestyle="--", label="95% upper bound (chi-square(1))")
    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title("LOG028: NIS Time Series")
    plt.legend()
    plt.grid(True)

    print("Done. Close plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    # Gravity used by the EKF
    dyn_params = DynamicsParams(
        g=9.81,
        thrust_acc=0.0,   # unused for real-data validation
        burn_time=0.0,    # unused
        drag_k=0.0        # unused
    )

    # EKF design / tuning params.
    # Start with something reasonable, then tune based on plots/NIS.
    ekf_params = EkfParams(
        dt=0.01,          # will be overwritten by log dt
        q_z=1e-4,         # process noise on altitude
        q_v=1e-2,         # process noise on velocity
        q_ba=1e-6,        # process noise on bias
        r_baro_std=3.0,   # assumed noise on Altitude [m] (tune this)
        x0=(0.0, 0.0, 0.0),          # initial [z, v, b_a]
        P0_diag=(10.0, 10.0, 1.0)    # initial covariance diag
    )

    run_ekf_on_log028(
        ekf_params=ekf_params,
        dyn_params=dyn_params,
        log_path="flight_logs/LOG028.TXT", 
        t_start=500,
        t_end=1000  # or full path if needed
    )