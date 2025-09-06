 # Hull-White Callable Bond Pricing with OAS Adjustment
# ----------------------------------------------------
# Simulates Hull-White interest rate paths, prices a callable bond via LSM,
# and fits an OAS (Option-Adjusted Spread) to match a given market price.

import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.polynomial import Polynomial

# ---------------------------------------------
# Step 1: Simulate Hull-White Short Rate Paths
# ---------------------------------------------
def simulate_hull_white_paths(a, sigma, r0, theta, T, dt, n_paths):
	n_steps = int(T / dt)
	rates = np.zeros((n_paths, n_steps + 1))
	rates[:, 0] = r0
	for t in range(1, n_steps + 1):
		dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
		rates[:, t] = rates[:, t - 1] + (theta[t - 1] - a * rates[:, t - 1]) * dt + sigma * dW
	return rates

# ---------------------------------------------
# Step 2: Price Callable Bond using LSM
# ---------------------------------------------
def price_callable_bond_lsm(rates, dt, face, call_price, call_dates_idx, coupon, maturity_idx, oas=0):
	n_paths, n_steps = rates.shape
	cashflows = np.zeros((n_paths, n_steps))

	# Final coupon and principal
	cashflows[:, maturity_idx] = face + coupon

	# Add annual coupons
	for t in range(1, maturity_idx):
		if t % int(1 / dt) == 0:
			cashflows[:, t] = coupon

	called = np.full(n_paths, False)

	for t in reversed(call_dates_idx):
		not_called = ~called
		if not np.any(not_called):
			continue

		df = np.exp(-np.cumsum(rates[:, t:], axis=1) * dt)
		continuation_value = (cashflows[not_called, t:] * df[not_called]).sum(axis=1)

		# Regression to estimate continuation value as a function of r_t
		X = rates[not_called, t]
		Y = continuation_value
		p = Polynomial.fit(X, Y, deg=2).convert()
		estimated_value = p(rates[not_called, t])

		# Decision to call
		call_condition = estimated_value < call_price
		to_call_indices = np.where(not_called)[0][call_condition]

		# Update cashflows
		cashflows[to_call_indices, t] = call_price
		cashflows[to_call_indices, t+1:] = 0
		called[to_call_indices] = True

	# Discount all cashflows with OAS
	discount_factors = np.exp(-np.cumsum(rates + oas, axis=1) * dt)
	total_dcf = np.sum(cashflows * discount_factors[:, :n_steps], axis=1)
	return np.mean(total_dcf)

# ---------------------------------------------
# Step 3: Fit OAS to Match Market Price
# ---------------------------------------------
def find_oas_to_match_price(target_price, *args):
	def objective(oas):
		return (price_callable_bond_lsm(*args, oas=oas) - target_price) ** 2
	result = minimize(objective, x0=0.0, bounds=[(-0.05, 0.05)])
	return result.x[0]

# ---------------------------------------------
# Step 4: Example Execution
# ---------------------------------------------
if __name__ == "__main__":
	# Model Parameters
	a = 0.1 # Mean reversion speed
	sigma = 0.01 # Volatility
	r0 = 0.03 # Initial short rate
	T = 10 # Maturity in years
	dt = 1 / 12 # Monthly time steps
	n_paths = 5000 # Monte Carlo paths

	# Bond Details
	face = 100
	coupon = 5
	call_price = 102
	market_price = 103.25

	n_steps = int(T / dt)
	maturity_idx = n_steps
	call_dates_idx = [int(x / dt) for x in range(3, 10)] # Callable from year 3 onwards

	# Flat theta (for constant yield curve)
	theta = np.full(n_steps, 0.03)

	# Step 1: Simulate interest rate paths
	print("Simulating Hull-White interest rate paths...")
	rates = simulate_hull_white_paths(a, sigma, r0, theta, T, dt, n_paths)

	# Step 2: Fit OAS to match market price
	print("Fitting OAS to market price...")
	oas = find_oas_to_match_price(market_price, rates, dt, face, call_price, call_dates_idx, coupon, maturity_idx)

	# Step 3: Recompute price using fitted OAS
	print("Computing callable bond price using fitted OAS...")
	model_price = price_callable_bond_lsm(rates, dt, face, call_price, call_dates_idx, coupon, maturity_idx, oas)

	# Output results
	print(f"Implied OAS: {oas:.6f}")
	print(f"Model Price: {model_price:.4f}")