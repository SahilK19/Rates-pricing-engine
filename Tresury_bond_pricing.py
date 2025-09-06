# Re-run the demo (state reset earlier)
import datetime as dt, math, numpy as np, pandas as pd, matplotlib.pyplot as plt

def generate_schedule(first_coupon_date, maturity_date, freq=2):
    months_step = 12 // freq
    schedule = []
    d = first_coupon_date
    while d <= maturity_date:
        schedule.append(d)
        year = d.year + (d.month - 1 + months_step) // 12
        month = (d.month - 1 + months_step) % 12 + 1
        day = min(d.day, [31,
                          29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                          31,30,31,30,31,31,30,31,30,31][month-1])
        d = dt.date(year, month, day)
    if schedule[-1] > maturity_date:
        schedule = schedule[:-1]
        schedule.append(maturity_date)
    elif schedule[-1] < maturity_date:
        schedule.append(maturity_date)
    schedule = sorted(list(dict.fromkeys(schedule)))
    return schedule

def year_fraction_act_act(start_date, end_date):
    return (end_date - start_date).days / 365.0

def price_with_zero_curve(face_value,
                          coupon_rate,
                          first_coupon_date,
                          maturity_date,
                          valuation_date,
                          zero_curve,
                          payment_frequency=2,
                          compounding='continuous'):
    schedule = generate_schedule(first_coupon_date, maturity_date, freq=payment_frequency)
    C = face_value * coupon_rate / payment_frequency
    future_dates = [d for d in schedule if d > valuation_date]
    cashflows = []
    for d in future_dates:
        amt = C + (face_value if d==maturity_date else 0)
        cashflows.append((d, amt))
    if isinstance(zero_curve, dict):
        curve_items = sorted(zero_curve.items())
    else:
        curve_items = sorted(zero_curve)
    T_points = np.array([t for t, r in curve_items], dtype=float)
    r_points = np.array([r for t, r in curve_items], dtype=float)
    pv_list = []
    price = 0.0
    for d, amt in cashflows:
        T = year_fraction_act_act(valuation_date, d)
        if T <= T_points[0]:
            r = r_points[0]
        elif T >= T_points[-1]:
            r = r_points[-1]
        else:
            r = np.interp(T, T_points, r_points)
        print(f"Date: {d}, T: {T:.4f}, Rate: {r:.4f}")
        if compounding == 'continuous':
            df = math.exp(-r * T)
        else:
            k = payment_frequency
            df = 1.0 / ((1 + r / k) ** (k * T))
        pv = amt * df
        pv_list.append({'pay_date': d, 'T_years': round(T,6), 'cashflow': amt, 'zero_rate': r, 'discount_factor': df, 'pv': pv})
        price += pv
    df_table = pd.DataFrame(pv_list)
    # accrued interest fraction f
    last_coupon = None; next_coupon = None
    for i in range(len(schedule)-1):
        if schedule[i] <= valuation_date < schedule[i+1]:
            last_coupon = schedule[i]; next_coupon = schedule[i+1]; break
    if last_coupon is None and valuation_date < schedule[0]:
        f = 0.0
    elif last_coupon is None and valuation_date >= schedule[-1]:
        f = 0.0
    else:
        days_elapsed = (valuation_date - last_coupon).days
        period_days = (next_coupon - last_coupon).days
        f = days_elapsed / period_days if period_days>0 else 0.0
    accrued = C * f
    clean_price = price - accrued
    totals = {'dirty_price': price, 'clean_price': clean_price, 'accrued_interest': accrued}
    return totals, df_table, schedule

# Demo values
face_value = 100
coupon_rate = 0.05
first_coupon_date = dt.date(2015,1,15)
maturity_date = dt.date(2030,1,15)
valuation_date = dt.date(2025,9,6)
zero_curve = {0.5:0.020,1.0:0.022,2.0:0.025,3.0:0.028,5.0:0.030,7.0:0.032,10.0:0.035,15.0:0.038}
print(zero_curve)
totals, df_table, schedule = price_with_zero_curve(face_value,coupon_rate,first_coupon_date,maturity_date,valuation_date,zero_curve,payment_frequency=2,compounding='continuous')












