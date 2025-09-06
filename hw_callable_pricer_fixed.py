# hw_callable_pricer_fixed.py
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import bisect
from datetime import date
import math

# ---------------------------
# User inputs (replace as needed)
# ---------------------------
valuation_date = date(2024, 9, 26)
issue_date = date(2019, 3, 21)
maturity_date = date(2037, 8, 1)
coupon_rate = 0.03585   # 3.585%
face = 100.0
call_price = 100.0
call_start_date = date(2028, 8, 1)
freq = 2  # semiannual coupons

# Hull-White parameters: DO NOT infer from discount factors.
# Either estimate from history or calibrate to swaptions.
a = 0.03
sigma = 0.01

# Simulation settings
dt = 1.0 / 12.0   # monthly steps -- choose sensible dt (monthly/daily)
T = (maturity_date - valuation_date).days / 365.25
n_steps = int(np.ceil(T / dt))
t_sim = np.linspace(0.0, T, n_steps + 1)
n_paths = 5000   # increase for production; keep moderate for demo
seed = 123456
rng = np.random.default_rng(seed)

# Market curve: if you have QuantLib bootstrapped discount_curve, supply arrays (t_grid, P0T).
# This demo uses a flat curve fallback; replace market_t_grid & market_P0T with your bootstrapped data.
market_t_grid = np.linspace(1e-6, 30.0, 301)
flat_rate = 0.037
market_P0T = np.exp(-flat_rate * market_t_grid)

# ---------------------------
# Build instantaneous forward f(0,t) and df/dt from P0T
# ---------------------------
# Use spline on y(t) = -ln P(0,t) then derivative: f = y'(t)
spline = UnivariateSpline(market_t_grid, -np.log(market_P0T), s=1e-8)
f0_dense = spline.derivative()(market_t_grid)
dfdt_dense = spline.derivative(n=2)(market_t_grid)

f0 = interp1d(market_t_grid, f0_dense, fill_value="extrapolate")
dfdt = interp1d(market_t_grid, dfdt_dense, fill_value="extrapolate")

# initial short rate r0 approximated by instantaneous forward at t->0
r0 = float(f0(1e-6))

# compute theta(t) on sim grid via analytic formula:
# theta(t) = df/dt + a*f(0,t) + (sigma^2/(2a))*(1 - exp(-2 a t))
def theta_on_grid(t_grid, a_val, sigma_val):
    f_vals = f0(t_grid)
    dfdt_vals = dfdt(t_grid)
    term = (sigma_val**2 / (2.0 * a_val)) * (1.0 - np.exp(-2.0 * a_val * t_grid))
    return dfdt_vals + a_val * f_vals + term

theta_sim = theta_on_grid(t_sim, a, sigma)
theta_func = interp1d(t_sim, theta_sim, fill_value="extrapolate")

# ---------------------------
# Simulate Hull-White paths using exact OU-discretization (with left-point theta)
# OU: dr = (theta(t) - a r) dt + sigma dW
# Exact update for constant theta over [t,t+dt] approximated using theta_left
# r_{t+dt} = r_t * exp(-a dt) + theta_left*(1 - exp(-a dt))/a + noise
# noise ~ Normal(0, sigma^2 * (1 - exp(-2 a dt)) / (2 a))
# ---------------------------
def simulate_hw_paths(theta_vals, a_val, sigma_val, r0_val, n_paths, t_grid, rng):
    dt_vec = np.diff(t_grid)
    paths = np.empty((n_paths, len(t_grid)))
    paths[:, 0] = r0_val
    for i, dti in enumerate(dt_vec):
        exp_ad = math.exp(-a_val * dti)
        m = (1.0 - exp_ad) / a_val
        s = math.sqrt((sigma_val**2) * (1.0 - math.exp(-2.0 * a_val * dti)) / (2.0 * a_val))
        theta_left = theta_vals[i]   # use left-point theta
        drift = theta_left * m
        z = rng.normal(size=n_paths)
        paths[:, i+1] = paths[:, i] * exp_ad + drift + s * z
    return paths

# ---------------------------
# Build cashflow dates and indices on simulation grid
# ---------------------------
def generate_cf_dates(valuation_date, maturity_date, freq_per_year=2):
    months = int(12 / freq_per_year)
    dates = []
    d = maturity_date
    while d > valuation_date:
        dates.append(d)
        # subtract months; keep day safe (avoid month-end issues)
        year = d.year
        month = d.month - months
        while month <= 0:
            month += 12
            year -= 1
        day = min(d.day, 28)
        d = date(year, month, day)
    dates = sorted([dd for dd in dates if dd > valuation_date])
    return dates

cf_dates = generate_cf_dates(valuation_date, maturity_date, freq_per_year=freq)
cf_times = np.array([(d - valuation_date).days / 365.25 for d in cf_dates])
cf_idx = np.minimum((cf_times / dt).round().astype(int), len(t_sim) - 1)
call_idx = np.searchsorted(cf_dates, call_start_date, side='left')

imm_cashflows = np.array([coupon_rate * face if i < len(cf_idx)-1 else coupon_rate * face + face
                          for i in range(len(cf_idx))])

# ---------------------------
# Longstaff-Schwartz LSM pricer for callable (issuer call logic)
# We'll price from holder perspective but determine issuer call by comparing
# issuer immediate cost vs continuation (estimated via regression).
# ---------------------------
def lsm_callable_price(paths, t_grid, cf_idx, imm_cash, call_idx, call_price, dt, oas=0.0):
    n_paths = paths.shape[0]
    # discount matrix: pathwise cumulative integral of (r + oas)
    r_plus = paths + oas
    # compute discount to each time step as exp(-integral r dt) using Riemann sum
    disc_to_step = np.exp(-np.cumsum(r_plus[:, 1:] * dt, axis=1))  # shape (n_paths, n_steps)
    # helper: PV to t=0 of immediate cashflow at cf index j
    def pv0_of_cf(j):
        idx = cf_idx[j]
        if idx - 1 >= 0:
            return imm_cash[j] * disc_to_step[:, idx - 1]
        else:
            return imm_cash[j] * np.ones(n_paths)
    # initialize path PVs (PV to time 0) and exercise flags
    pv_future = np.zeros(n_paths)
    exercised = np.zeros(n_paths, dtype=bool)

    # start from last cf and move backward
    for j in range(len(cf_idx)-1, -1, -1):
        idx = cf_idx[j]
        immediate_pv0 = pv0_of_cf(j)
        # set final cashflow for those not exercised yet when j == last
        if j == len(cf_idx)-1:
            pv_future[~exercised] = immediate_pv0[~exercised]
            # mark not exercised as alive for earlier decisions
        # translate continuation (pv_future) to time j (value at time j)
        if idx - 1 >= 0:
            denom = disc_to_step[:, idx - 1]
            cont_at_j = pv_future / denom
        else:
            cont_at_j = pv_future.copy()
        # Only consider alive paths for call decision
        alive = ~exercised
        if not np.any(alive):
            break
        # build regression basis from state at time j: use short rate at index idx
        r_state = paths[:, idx]
        # regress continuation (cont_at_j) on basis using alive paths
        X_alive = np.vstack([np.ones(np.sum(alive)), r_state[alive], r_state[alive]**2]).T
        Y_alive = cont_at_j[alive]
        if X_alive.shape[0] > X_alive.shape[1]:
            coeffs, *_ = np.linalg.lstsq(X_alive, Y_alive, rcond=None)
            # estimate continuation for all paths
            X_all = np.vstack([np.ones(paths.shape[0]), r_state, r_state**2]).T
            cont_est_at_j = X_all.dot(coeffs)
        else:
            cont_est_at_j = cont_at_j  # fallback if insufficient alive paths
        # issuer immediate cost at time j (assume coupon + call principal)
        imm_call = coupon_rate * face + call_price
        # issuer compares imm_call vs continuation (both measured at time j)
        exercise_now = alive & (imm_call < cont_est_at_j)
        # for paths exercised now, set PV to immediate PV (to time 0)
        if idx - 1 >= 0:
            pv0_call = imm_call * disc_to_step[:, idx - 1]
        else:
            pv0_call = imm_call * np.ones(n_paths)
        pv_future[exercise_now] = pv0_call[exercise_now]
        exercised = exercised | exercise_now
        # those not exercised keep their pv_future (already holds PV to 0 of future CFs)
    # final price is mean PV across paths
    return pv_future.mean()

# ---------------------------
# Model price function with OAS (runs simulation -> LSM price)
# ---------------------------
def model_price_with_oas(oas):
    paths = simulate_hw_paths(theta_sim, a, sigma, r0, n_paths, t_sim, rng)
    return lsm_callable_price(paths, t_sim, cf_idx, imm_cashflows, call_idx, call_price, dt, oas=oas)

# ---------------------------
# (Example) create a synthetic market price using some OAS_true, then calibrate OAS by bisection
# ---------------------------
oas_true = 0.00517
print("Simulating synthetic market price (this may take some seconds)...")
market_price = model_price_with_oas(oas_true)
print("Synthetic market price (using OAS_true):", market_price)

def calibrate_oas(target_price, low=-0.02, high=0.02, tol=1e-4, maxiter=40):
    def f(o):
        return model_price_with_oas(o) - target_price
    fl, fh = f(low), f(high)
    if fl * fh > 0:
        raise ValueError("Bracket does not contain root; widen bounds")
    root = bisect(f, low, high, xtol=tol, maxiter=maxiter)
    return root

oas_cal = calibrate_oas(market_price, low=-0.01, high=0.02, tol=5e-4)
print("Calibrated OAS (should be close to oas_true):", oas_cal)
print("Price at calibrated OAS:", model_price_with_oas(oas_cal))
