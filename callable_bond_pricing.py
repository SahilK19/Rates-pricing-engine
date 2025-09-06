
import QuantLib as ql
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# === INPUT SECTION ===

# NSS Parameters
nss_params = {
    'beta0': 4.747453595,
    'beta1': 0.498534975,
    'beta2': -2.457209244,
    'beta3': -3.117106302,
    'tau1': 0.70877096,
    'tau2': 3.211860157
}
valuation_date = ql.Date(26, 9, 2024)
ql.Settings.instance().evaluationDate = valuation_date
day_count = ql.Thirty360(ql.Thirty360.USA)
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
interpolation = ql.Linear()
compounding = ql.Compounded
compoundingFrequency = ql.Semiannual
tenor = ql.Period(ql.Semiannual)
businessday_convention = ql.Unadjusted
date_generation = ql.DateGeneration.Backward
monthend = False
settlement_days = 2
face_value = 100


#Curve bootstrap function
    
def curve_bootstrap(period, par_rates, calc_date, settlement_days, face_value, coupon_freq, businessday_convention, day_count, calendar, compounding):
    ql.Settings.instance().evaluationDate = calc_date
    bond_helpers = []
    
    for r, p in zip(par_rates, period):
        termination_date = calc_date + p
        schedule = ql.Schedule(calc_date, termination_date, coupon_freq, calendar, businessday_convention, businessday_convention, ql.DateGeneration.Backward, False)
        helper = ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(face_value)), settlement_days, face_value, schedule, [r/100.0], day_count, businessday_convention)
        bond_helpers.append(helper)
        
    yieldcurve = ql.PiecewiseLogCubicDiscount(calc_date, bond_helpers, day_count)
    dates, discount_factors = zip(*yieldcurve.nodes())
    zero_rate = [yieldcurve.zeroRate(i, day_count, compounding).rate() for i in dates]
    bootstrapped_zero_curve = ql.ZeroCurve(dates, zero_rate, day_count, calendar)
    return bootstrapped_zero_curve, zero_rate

# Define your dates and zero rates
periods = [ql.Period(1, ql.Months), ql.Period(2, ql.Months), ql.Period(3, ql.Months), ql.Period(4, ql.Months), ql.Period(6, ql.Months),
         ql.Period(1, ql.Years), ql.Period(2, ql.Years), ql.Period(3, ql.Years), ql.Period(4, ql.Years), ql.Period(5, ql.Years),
         ql.Period(6, ql.Years), ql.Period(7, ql.Years), ql.Period(8, ql.Years), ql.Period(9, ql.Years), ql.Period(10, ql.Years),
         ql.Period(15, ql.Years), ql.Period(20, ql.Years), ql.Period(25, ql.Years), ql.Period(30, ql.Years)]

par_rates = [4.711, 4.702, 4.619, 4.567, 4.404, 3.987, 3.63, 3.55, 3.573, 3.568, 3.633, 3.666, 3.724, 3.769, 3.797, 3.962, 4.184, 4.269, 4.132]

# Convert periods to dates using the calendar's advance method
bootstrapping_curve, zero_rates = curve_bootstrap(periods, par_rates, ql.Date(26, 9, 2024), 1, 100, ql.Period(ql.Semiannual), ql.Unadjusted, ql.ActualActual(ql.ActualActual.Bond), ql.UnitedStates(ql.UnitedStates.GovernmentBond), ql.Continuous)
term_structure_handle = ql.YieldTermStructureHandle(bootstrapping_curve)

# Define the maturities for the zero rates
maturities = [valuation_date] + [valuation_date + p for p in periods]

# Calculate the year fractions
yr_fraction = [(m - valuation_date) / 365.0 for m in maturities]

Disc_df = pd.DataFrame({"yr_fraction": yr_fraction, "zero_rates": zero_rates})
Disc_df["ZCB Price"] = np.exp(-1 * Disc_df["zero_rates"] * Disc_df["yr_fraction"])

discount_curve = ql.DiscountCurve(maturities, Disc_df['ZCB Price'], day_count, calendar)
discount_curve.enableExtrapolation()
discount_curve_handle = ql.YieldTermStructureHandle(discount_curve)

# Compute forward rates f(0,t)
def compute_forward_rates(market_curve, market_times):
    forward_rates = []
    for t in market_times:
        f_t = market_curve.forwardRate(t, t, ql.Continuous, ql.NoFrequency).rate()
        forward_rates.append(f_t)
    return np.array(forward_rates)

market_times = yr_fraction  # Use yr_fraction as market_times
forward_rates = compute_forward_rates(discount_curve, market_times)
# Compute theta(t)
def compute_theta(alpha, sigma):
    theta_t = []
    for i, t in enumerate(market_times):
        f_t = forward_rates[i]

        # Compute df/dt using finite difference (approximation)
        if i == len(market_times) - 1:  # Edge case: last point
            df_dt = (f_t - forward_rates[i - 1]) / (market_times[i] - market_times[i - 1])
        else:
            df_dt = (forward_rates[i + 1] - f_t) / (market_times[i + 1] - t)

        # Correct theta formula
        theta_t.append(df_dt + alpha * f_t + (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * t)))
    
    return np.array(theta_t)
# Define optimization function to minimize error in discount factors
def loss_function(params):
    alpha, sigma = params
    if alpha <= 0 or sigma <= 0:
        return np.inf  # Constraint: α, σ > 0

    theta_t = compute_theta(alpha, sigma)
    # hw_model = ql.HullWhite(discount_curve_handle, alpha, sigma)  # Not used
    
    model_dfs = [discount_curve.discount(t) * np.exp(-theta_t[i] * t) for i, t in enumerate(market_times)]

    error = np.sum((np.array(model_dfs) - np.array(Disc_df['ZCB Price']))**2)
    return error

# Initial guess
alpha_init = 0.01
sigma_init = 0.01

# Optimize α, σ
result = minimize(loss_function, [alpha_init, sigma_init], bounds=[(0.01, 2.0), (0.0001, 1)])
alpha_calibrated, sigma_calibrated = result.x 

print(f"Calibrated α: {alpha_calibrated:.6f}, Calibrated σ: {sigma_calibrated:.6f}")

# Bond Info
issue_dt = datetime(2019, 3, 21)
maturity_dt = datetime(2037, 8, 1)
valuation_date = datetime(2024, 9, 26)
coupon = 3.585 / 100
call_start_date = datetime(2028, 8, 1)
face_value = 100
call_price = 100
freq = 2  # semiannual

# Simulation settings
n_paths = 10000
dt = 1/n_paths
steps = int((maturity_dt - valuation_date).days / 360 / dt)
oas = 0.00517


# === FUNCTIONS ===

def nss_yield(t, beta0, beta1, beta2, beta3, tau1, tau2):
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1))
    term3 = beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1))
    term4 = beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))
    return term1 + term2 + term3 + term4


def generate_zero_curve(nss_params, t_grid):
    rates = nss_yield(t_grid, **nss_params)
    return t_grid, rates / 100


def forward_rate(t, z):
    fwd = np.gradient(z, t) * t + z
    return fwd

"""""
def calibrate_hull_white(t, z):
    def model_discount(t, alpha, sigma):
        fwd = forward_rate(t, z)
        B = (1 - np.exp(-alpha * t)) / alpha
        A = np.exp((B * fwd - (sigma ** 2 / (4 * alpha)) * B ** 2) - z * t)
        return A * np.exp(-B * z[0])

    def objective(params):
        alpha, sigma = params
        model_df = model_discount(t, alpha, sigma)
        market_df = np.exp(-z * t)
        return np.mean((model_df - market_df) ** 2)

    res = minimize(objective, [0.03, 0.01], bounds=[(1e-4, 2), (1e-4, 0.5)])
    return res.x
"""

def compute_theta(t, fwd, dfdt, alpha, sigma):
    theta = dfdt + alpha * fwd + (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * t))
    return theta


def simulate_hull_white_paths(theta, alpha, sigma, r0, n_paths, steps, dt):
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = r0
    for i in range(steps):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        t = (i+1) * dt
        theta_t = theta[i]
        paths[:, i+1] = (paths[:, i] + (theta_t - alpha * paths[:, i]) * dt + sigma * dW)
    return paths


def price_callable_bond_with_oas(paths, cashflow_dates, coupon, face_value, call_price,
                                  call_start_idx, dt, valuation_date, oas):
    n_paths, n_steps = paths.shape
    price_paths = np.zeros(n_paths)

    cashflow_times = np.array([(d - valuation_date).days / 365.25 for d in cashflow_dates])
    cf_idx = np.round(cashflow_times / dt).astype(int)

    for p in range(n_paths):
        r = paths[p]
        cf_value = 0
        for i, idx in enumerate(cf_idx):
            if idx >= len(r):
                continue
            
            adjusted_r = r[1:idx+1] + oas
            cf_disc = np.exp(-np.sum(adjusted_r) * dt)

            if i >= call_start_idx:
                call_disc = np.exp(-np.sum(adjusted_r) * dt)
                if call_disc * (coupon * face_value + face_value) > call_disc * call_price:
                    cf_value += call_price * call_disc
                    break
            if i == len(cf_idx) - 1:
                cf_value += (coupon * face_value + face_value) * cf_disc
            else:
                cf_value += (coupon * face_value) * cf_disc
        price_paths[p] = cf_value

    return np.mean(price_paths)


# === MAIN EXECUTION ===

t_grid = np.arange(0.1, 30.1, 0.1)
t, z = generate_zero_curve(nss_params, t_grid)
print(f"Generated zero curve: {z}")
print(f"Time grid: {t}")
fwd = forward_rate(t, z)
print(f"Forward rates: {fwd}")  
dfdt = np.gradient(fwd, t)
print(f"Forward rate derivatives: {dfdt}")
alpha, sigma = result.x
print(f"Calibrated Hull-White parameters: alpha = {alpha:.5f}, sigma = {sigma:.5f}")
theta = compute_theta(t, fwd, dfdt, alpha, sigma)
print(f"Theta values: {theta}")
theta_interp = interp1d(t, theta, fill_value="extrapolate")
print(f"Interpolated theta function: {theta_interp}")
theta_path = [theta_interp(i*dt) for i in range(steps)]
print(f"Theta path: {theta_path}")
paths = simulate_hull_white_paths(theta_path, alpha, sigma, z[0], n_paths, steps, dt)
print(f"Simulated Hull-White paths: {paths}")
cf_dates = pd.date_range(start=maturity_dt, end=issue_dt, freq='-6M')[::-1]
cf_dates = [d for d in cf_dates if d > valuation_date]
call_idx = sum([d < call_start_date for d in cf_dates])

price = price_callable_bond_with_oas(paths, cf_dates, coupon, face_value, call_price,
                                     call_idx, dt, valuation_date, oas)

print(f"Callable Bond Price with OAS = {oas:.5f} is: {price:.2f}")
