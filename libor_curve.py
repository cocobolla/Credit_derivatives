# -*- coding: utf-8 -*-
"""
Created on Thu Nov 08 06:11 2018
@author: Cocobolla (Chanju Park)
"""
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt


class LIBOR():
    def __init__(self, trade_date, adjust_vol, settle_convention=2, year_convention=360):
        self.trade_date = trade_date
        self.convexity_adjusting_vol = adjust_vol
        self.coupon_freq = 'semi-annual'
        self.settle_convention = settle_convention
        self.year_convention = year_convention
        self.date_dict = {
            'W': '7',
            'M': '30',
            'Y': '365'
        }

        # Should be loaded on data_load function
        self.data = None
        self.data_dict = None

        self.target_factor = {}

    def data_load(self, data_path):
        """ Load excel data

        Unify the data format:
            Term: maturity date.
            Rate: if rate, % -> normalize with 100

        """
        data = pd.read_excel(data_path)
        for index in data.index:
            # Unify date format
            date = (data.loc[index]['Term'])
            if type(date) == str and date not in ['O/N', 'T/N']:
                try:
                    date_unit = self.date_dict[date[-1]]
                    # Change '1W' format to '1*7' string
                    diff_date = date.replace(date[-1], '*' + date_unit)
                    # Change '1*7' string into 1*7 = 7 number
                    diff_date = eval(diff_date)
                    data.loc[index, 'Term'] = self.trade_date + relativedelta(days=diff_date)
                except KeyError:
                    print("Date Format({}) is starange!".format(data.ix[index, 'Term']))
                    print(data.ix[index])
                    raise
            elif type(date) == datetime.datetime:
                data.loc[index, 'Term'] = data.loc[index, 'Term'].date()

            instrument = data.loc[index, 'Instrument']
            # Unify price unit(% -> float point)
            if instrument != 'Futures':
                data.loc[index, 'Rate'] /= 100
        self.data = data
        self.data_dict = {
            'Deposit': self.data.loc[self.data.loc[:, 'Instrument'] == 'MMD'],
            'Futures': self.data.loc[self.data.loc[:, 'Instrument'] == 'Futures'],
            'Swap': self.data.loc[self.data.loc[:, 'Instrument'] == 'Swap']
        }

    @staticmethod
    def discount_on_mmd(maturity_diff, historic_libor, discount_ts, settle_convention=2, year_convention=360):
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
        delta = (maturity_diff.days - settle_convention) / year_convention

        # Z(0, T) = Z(0, ts)/(1 + delta(ts, T) * L(ts,T))
        discount_T = discount_ts / (1 + delta*historic_libor)

        return discount_T

    @staticmethod
    def discount_on_fra(maturity1, maturity2, forward_rate, year_convention=360):
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
        Z(ts, T2)/Z(ts, T1) (discount rate)
        """
        # Delta(ts, T)
        delta = (maturity2.days - maturity1.days) / year_convention

        discount_ratio = 1/(1 + delta * forward_rate)
        return discount_ratio

    @staticmethod
    def future_price_to_forward_rate(futures_p, volatility, t1, t2):
        volatility = volatility/100
        forward_rate = 100 - futures_p - t1.days*t2.days*volatility**2/2
        forward_rate /= 100
        return forward_rate

    @staticmethod
    def discount_on_irs(data):
        pass

    # Calculate LIBOR discount factor with deposit data
    def _calculate_on_deposit(self):
        data = self.data_dict['Deposit']
        discount_factor_dict = {}

        settle_convention = 360

        on_index = data.loc[:, 'Term'] == 'O/N'
        tn_index = data.loc[:, 'Term'] == 'T/N'
        other_index = ((on_index + tn_index) != True )

        # If O/N and T/N data are more than 1, Use first data
        on_libor = data.loc[on_index].iloc[0]['Rate']
        tn_libor = data.loc[tn_index].iloc[0]['Rate']

        # Calculate Discount Factor at ts using overnight and tomorrow next LIBOR
        Z_ts = ((1 + on_libor/settle_convention)*(1 + tn_libor/settle_convention))
        Z_ts = Z_ts**-1

        for index, row in data.loc[other_index].iterrows():
            maturity = row['Term']
            maturity_diff = maturity - self.trade_date
            historic_libor = row['Rate']

            # Get Discount Factor (Z_T)
            Z_T = self.discount_on_mmd(maturity_diff, historic_libor, Z_ts)

            # discount_factor_dict
            discount_factor_dict[maturity] = Z_T
        return discount_factor_dict

    # Calculate LIBOR discount factor with futures data
    def _calculate_on_future(self):
        data = self.data_dict['Futures']
        # key: (T1, T2), Value: Z(0, T2)/Z(0, T1)
        discount_factor_dict = {}

        # Convert from Futures price to Forward rate
        for index, row in data.iterrows():
            futures_price = row['Rate']
            maturity1 = row['Term']
            maturity2 = maturity1 + relativedelta(days=30)
            maturity_diff1 = maturity1 - self.trade_date
            maturity_diff2 = maturity2 - self.trade_date

            forward_rate = self.future_price_to_forward_rate(futures_price, self.convexity_adjusting_vol, maturity_diff1, maturity_diff2)
            discount_ratio = self.discount_on_fra(maturity_diff1, maturity_diff2, forward_rate)

            discount_factor_dict[maturity1, maturity2] = discount_ratio
        return discount_factor_dict

    # Calculate LIBOR discount factor with swaps data
    def _calculate_on_swap(self):
        pass

    def draw_curve(self):
        deposit_discount_dict = self._calculate_on_deposit()
        forward_discount_dict = self._calculate_on_future()
        discount_dict = deposit_discount_dict

        # Get Forward discount
        for maturity_tuple, ratio in forward_discount_dict.items():
            # Use the pre-calculated data if day diff < 20
            T1_diff_list = [maturity for maturity in discount_dict if abs((maturity - maturity_tuple[0]).days) < 20 and discount_dict[maturity] > 0]
            T2_diff_list = [maturity for maturity in discount_dict if abs((maturity - maturity_tuple[1]).days) < 20 and discount_dict[maturity] > 0]

            # If there are data we can use as Z(0,T1), with this we can get Z(0,T2)
            if len(T1_diff_list) > 0:
                discount_factor = discount_dict[T1_diff_list[0]]
                discount_dict[maturity_tuple[1]] = discount_factor*ratio

            # If there are data we can use as Z(0,T2)
            elif len(T2_diff_list) > 0:
                discount_factor = discount_dict[T2_diff_list[0]]
                discount_dict[maturity_tuple[0]] = discount_factor/ratio
            else:
                discount_dict[maturity_tuple[0]] = -1
                discount_dict[maturity_tuple[1]] = -1

        # TODO: Interpolate the value that is -1(undetermined) on discount_dict
        discount_dict = self.interpolation(discount_dict)

        libor_dict = {}

        # Calculate Zero rate with discount factor
        for key, discount in discount_dict.items():
            zero_rate = (1/discount - 1)*self.year_convention/(key - self.trade_date).days
            zero_rate *= 100
            libor_dict[key] = zero_rate

        plt.step(list(libor_dict.keys()), list(libor_dict.values()), where='post')
        plt.xlabel('Date')
        plt.ylabel('LIBOR(%)')
        plt.title('LIBOR Curve')
        plt.show()

    @staticmethod
    def interpolation(target_dict):
        # TODO: implement interpolation such as date weighted method etc..

        return target_dict


if __name__ == '__main__':
    trade_date = datetime.date(2018, 10, 29)
    futures_volatility = 0.3
    data_path = 'data.xlsx'

    libor = LIBOR(trade_date, futures_volatility)
    libor.data_load(data_path)
    libor.draw_curve()

