import datetime
from dateutil.relativedelta import relativedelta
import calendar
from matplotlib import pyplot as plt
import numpy as np

# Initial Settings
trade_date = datetime.date(2019, 11, 4)
settle_convention = 0
year_convention = 360
futures_vol = 0.5


# For calculate Futures Expiration date
def third_wednesday(year, month):
    first_day_of_month = datetime.date(year, month, 1)
    # 2 is wednesday of week
    first_wed = first_day_of_month + datetime.timedelta(days=((2 - calendar.monthrange(year, month)[0]) + 7) % 7)
    multiplier = 1 if (first_day_of_month.weekday() > 2 and first_day_of_month.weekday() != 6) else 2

    third_wed = first_wed + datetime.timedelta(days=7*multiplier)
    return third_wed


def discount_mmd(maturity_diff, historic_libor, settle_convention=2, year_convention=360):
    """ Get the LIBOR discount factor on Money Market Deposit

    ts = Trading_date(0) + Settlement_Convention(settle_convention)
    Delta(ts, T) = DayDiff(ts, T) / Money_Market_Deposit_Convention(year_convention)
    Z(0, T) = Z(0, ts)/(1 + delta(ts, T) * L(ts,T))

    Parameters
    ----------
    maturity_diff: maturity time - trade time,
    historic_libor: historical LIBOR [L(ts, T)],
    settle_convention: Settlement convention (default = 2),
    year_convention: Money Market Deposit Convention (default = 360)

    Returns
    -------
    discount_T: Z(0, T) (discount rate)
    """
    # Delta(ts, T)
    delta = (maturity_diff.days + settle_convention) / year_convention

    # Z(0, T) = Z(0, ts)/(1 + delta(ts, T) * L(ts,T))  or  Z(0, T) = 1/(1 + delta(ts, T) * L(ts,T))
    discount = 1 / (1 + delta * historic_libor)

    return discount


def discount_fra(maturity1, maturity2, forward_rate, year_convention=360):
    """ Get the LIBOR discount factor on Forward Rate Agreements

    Forward rate = 1/delta(T1, T2) * (Z(ts, T1)/Z(ts, T2) - 1)

    Parameters
    ----------
    maturity1: Starting time,
    maturity2: End time,
    forward_rate: quoted forward rate
    year_convention: Money Market Deposit Convention (default = 360)

    Returns
    -------
    Z(ts, T2)/Z(ts, T1) (discount rate ratio)
    """
    # Delta(ts, T)
    delta = (maturity2.days - maturity1.days) / year_convention

    discount_ratio = 1 / (1 + delta * forward_rate)
    return discount_ratio


def future2forward(futures_p, volatility, t1, t2):
    volatility = volatility / 100
    forward_rate = 100 - futures_p - t1.days * t2.days * volatility ** 2 / 2
    forward_rate /= 100
    return forward_rate


def discount_on_irs():
    pass


def interpolation(point):
    data_point = [
        (0.5, 0), (1, 1.71), (2, 1.59), (3, 1.53),
        (5, 1.51), (7, 1.54), (10, 1.62), (30, 1.80)
    ]
    x_list = np.array([p[0] for p in data_point])
    y_list = np.array([p[1] for p in data_point])
    if point < x_list[0] or point > x_list[-1]:
        print("Cannot Interpolate")
        exit()

    if point in x_list:
        return y_list[np.argwhere(x_list == point)][0][0]
    else:
        sol = None
        for i, p in enumerate(data_point):
            if point < p[0]:
                x0 = data_point[i-1][0]
                y0 = data_point[i-1][1]
                x1 = p[0]
                y1 = p[1]
                sol = (x1 - point) / (x1 - x0) * y0 + (point - x0) / (x1 - x0) * y1
                break
        return sol


def main():
    mmd_data = [
        {'Term': trade_date + datetime.timedelta(days=7), 'Rate': 1.63163/100},
        {'Term': trade_date + relativedelta(months=1), 'Rate': 1.77425/100},
        {'Term': trade_date + relativedelta(months=2), 'Rate': 1.83888/100},
        {'Term': trade_date + relativedelta(months=3), 'Rate': 1.8905/100},
        {'Term': trade_date + relativedelta(months=6), 'Rate': 1.90238/100},
        {'Term': trade_date + relativedelta(months=12), 'Rate': 1.92525/100}
    ]

    futures_data = [
        {'Term': third_wednesday(2019, 12), 'Rate': 98.09},
        {'Term': third_wednesday(2020, 3), 'Rate': 98.33},
        {'Term': third_wednesday(2020, 6), 'Rate': 98.42},
        {'Term': third_wednesday(2020, 9), 'Rate': 98.48},
        {'Term': third_wednesday(2020, 12), 'Rate': 98.475},
    ]

    swap_data = [
        {'Term': trade_date + relativedelta(years=1), 'Rate': 1.71},
        {'Term': trade_date + relativedelta(years=2), 'Rate': 1.59},
        {'Term': trade_date + relativedelta(years=3), 'Rate': 1.53},
        {'Term': trade_date + relativedelta(years=5), 'Rate': 1.51},
        {'Term': trade_date + relativedelta(years=7), 'Rate': 1.54},
        {'Term': trade_date + relativedelta(years=10), 'Rate': 1.62},
        {'Term': trade_date + relativedelta(years=30), 'Rate': 1.80},
    ]

    mmd_discount_dict = {}
    forward_discount_dict = {}  # key: (T1, T2), Value: Z(0, T2)/Z(0, T1)
    swap_discount_dict = {}

    # Calculate LIBOR discount factor with deposit data
    for r in mmd_data[:5]:
        maturity = r['Term']
        maturity_diff = maturity - trade_date
        historic_libor = r['Rate']

        # Get Discount Factor (Z_T)
        Z_T = discount_mmd(maturity_diff, historic_libor)

        # discount_factor_dict
        mmd_discount_dict[maturity] = Z_T

    # Calculate LIBOR discount factor with futures data
    # Convert from Futures price to Forward rate
    for r in futures_data[1:]:
        futures_price = r['Rate']
        maturity2 = r['Term']
        maturity1 = maturity2 - relativedelta(months=1)
        maturity_diff1 = maturity1 - trade_date
        maturity_diff2 = maturity2 - trade_date

        forward_rate = future2forward(futures_price, futures_vol, maturity_diff1, maturity_diff2)
        discount_ratio = discount_fra(maturity_diff1, maturity_diff2, forward_rate)

        forward_discount_dict[maturity2] = discount_ratio

    # Draw Curve
    # March Futures
    discount_dict = mmd_discount_dict
    mar_key = third_wednesday(2020, 3)
    mar_ratio = forward_discount_dict[mar_key]
    feb_discount = discount_dict[trade_date + relativedelta(months=3)]

    discount_dict[third_wednesday(2020, 3)] = feb_discount * mar_ratio

    # June Futures
    jun_key = third_wednesday(2020, 6)
    jun_ratio = forward_discount_dict[jun_key]
    may_discount = discount_dict[trade_date + relativedelta(months=6)]
    del discount_dict[trade_date + relativedelta(months=6)]

    discount_dict[third_wednesday(2020, 6)] = may_discount * jun_ratio

    # September Futures
    a = third_wednesday(2020, 3)
    b = third_wednesday(2020, 6)
    x = third_wednesday(2020, 8)
    sep_key = third_wednesday(2020, 9)
    # linear - polation
    sep_ratio = forward_discount_dict[sep_key]
    aug_discount = discount_dict[b] + (discount_dict[b] - discount_dict[a]) * (x - b) / (b - a)

    discount_dict[third_wednesday(2020, 9)] = aug_discount * sep_ratio

    # Calculate LIBOR discount factor with futures data
    # Interpolation
    swap_year = np.arange(0.5, 30, 0.5)
    swap_discount = list()
    swap_rate = [interpolation(x) for x in swap_year]

    swap_discount.append(may_discount)
    # d1 = (1 - swap_data[0]['Rate']/100 * 0.5 * (may_discount))/(swap_data[0]['Rate'] * 0.5 + 1)
    # swap_discount.append(d1)
    for i in range(1, len(swap_rate)):
        di = (1 - swap_rate[i]/100 * 0.5 * sum(swap_discount[:i])) / (swap_rate[i]/100 * 0.5 + 1)
        swap_discount.append(di)

    for i, y in enumerate(swap_year[1:]):
        discount_dict[trade_date + relativedelta(months=y*12)] = swap_discount[i]

    libor_dict = {}
    # Calculate Zero rate with discount factor
    for key, discount in discount_dict.items():
        zero_rate = (1 / discount - 1) * year_convention / (key - trade_date).days
        zero_rate *= 100
        libor_dict[key] = zero_rate

    plt.step(list(libor_dict.keys()), list(libor_dict.values()), where='post')
    plt.xlabel('Date')
    plt.ylabel('LIBOR(%)')
    plt.title('LIBOR Curve')
    plt.show()


if __name__ == '__main__':
    main()
