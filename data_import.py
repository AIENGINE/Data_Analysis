#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:52:31 2018

@author: sabih
"""
import os
import sys
import pandas as pd
import numpy as np
from functools import lru_cache
import copy
from typing import List
import logging
from exceptions import param_check_correct_list
import datetime
logger_this = logging.getLogger(__name__)
logger_this.setLevel(logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@lru_cache()
class DataImport():
    def __init__(self, path=None):
        if path is None:
            self._path = os.getcwd()
        else:
            self._path = path

    def read_data_csv(self):
        self._df_deliveries_idx = pd.read_csv('deliveries.csv')
        self._df_customers = pd.read_csv('customer_details.csv')
        self._df_colocations = pd.read_csv('colocations.csv')
        self._df_levels_idx = pd.read_csv('level_readings.csv')


        self._df_levels = self._df_levels_idx.set_index(['ON_DATE_TIME'])
        self._df_deliveries = self._df_deliveries_idx.set_index(['DELIVERY_DATE'])

        self._df_levels.index = pd.to_datetime(self._df_levels.index)
        self._df_deliveries.index = pd.to_datetime(self._df_deliveries.index)

    def write_sorted(self):
        vessel_ids = self._df_customers['VESSEL_ID']
        self._grp_lv_vid = self._df_levels_idx.groupby('VESSEL_ID').groups
        npr = np.empty(len(self._df_levels), dtype=np.object)

        for idx, vessel_id in enumerate(vessel_ids):
            s_mask = self._grp_lv_vid[vessel_id]
            index = self._df_levels.iloc[s_mask].index
            int_index = self._df_levels_idx.iloc[s_mask].index
            print(index.is_monotonic)
            if index.is_monotonic == False:
                new_index = list(sorted(index))
                npr[int_index] = new_index
                print('new index is sorted')
            else:
                new_index =list(index)
                npr[int_index] = new_index
        self._df_levels_idx['SORTED_DATE_TIME']=pd.to_datetime(npr)
        self._df_levels['SORTED_DATE_TIME']=pd.to_datetime(npr)
        # self._df_levels.reindex(self._df_levels['SORTED_DATE_TIME'])
        self._df_levels.reindex(self._df_levels['SORTED_DATE_TIME'])
        self._df_levels.to_hdf('sorted_levels_idx_fin.h5', 'dataset')
        # self._df_levels.to_hdf('sorted_levels.h5', 'dataset')

    def check_sorted(self):
        vessel_ids = self._df_customers['VESSEL_ID']
        self._grp_lv_vid = self._df_levels_idx.groupby('VESSEL_ID').groups
        self._grp_dl_vid = self._df_deliveries_idx.groupby('VESSEL_ID').groups

        for idx, vessel_id in enumerate(vessel_ids):
            s_mask = self._grp_lv_vid[vessel_id]
            d_mask = self._grp_dl_vid[vessel_id]

            index_lv = self._df_levels.iloc[s_mask]['SORTED_DAY_TIME']
            index_dl = self._df_deliveries.iloc[d_mask].index

            if index_lv.is_monotonic == False:
                print("LEVELS ARE NOT MONOTONIC")
            if index_dl.is_monotonic == False:
                print("DELIVERIES ARE NOT MONOTONIC")

    def read_data_hdf(self):
        self._df_deliveries = pd.read_hdf('deliveries.h5', 'dataset')
        self._df_customers = pd.read_hdf('customers.h5', 'dataset')
        self._df_levels = pd.read_hdf('sorted_levels.h5', 'dataset')
        self._df_colocations = pd.read_hdf('colocations.h5', 'dataset')
        self._df_levels_idx = pd.read_hdf('levels_idx.h5', 'dataset')
        self._df_deliveries_idx = pd.read_hdf('deliveries_idx.h5', 'dataset')
        return (self._df_deliveries, self._df_customers,
                self._df_colocations, self._df_levels)

    def write_to_hdf(self):
        self._df_deliveries.to_hdf('deliveries.h5', 'dataset')
        self._df_deliveries_idx.to_hdf('deliveries_idx.h5', 'dataset')
        self._df_customers.to_hdf('customers.h5', 'dataset')
        self._df_colocations.to_hdf('colocations.h5', 'dataset')
        self._df_levels.to_hdf('levels.h5', 'dataset')
        self._df_levels_idx.to_hdf('levels_idx.h5', 'dataset')


class Utils():
    def __init__(
        self,
        df_deliveries,
        df_customers,
        df_colocations,
        df_levels,
    ):
        self._df_deliveries = df_deliveries
        self._df_customers = df_customers
        self._df_colocations = df_colocations
        self._df_levels = df_levels

    def sample_level_delivery_vessels(
            df_customers,
            df_levels,
            df_deliveries,
            col,
            col_val
    ):

        df_sampled = df_customers.loc[df_customers[col].isin(col_val)]
        df_levels_n = df_levels.loc[df_levels['VESSEL_ID'].isin(
            df_sampled['VESSEL_ID'])]
        df_deliveries_n = df_deliveries.loc[df_deliveries['VESSEL_ID'].isin(
            df_sampled['VESSEL_ID'])]
        return (df_levels_n, df_deliveries_n)

    def get_customer_sets(
        self,
        col,
    ):
        return pd.unique(self._df_customers[col])

    @staticmethod
    def running_customer_sets(
        df: pd.DataFrame,
        running_cols,
        col,
        col_val,
    ):
        col_val = param_check_correct_list(col_val)
        df = df.loc[df[col].isin(col_val)]
        running_cols = list(set(running_cols) - set([col]))
        unique_vals = []
        for col in running_cols:
            uq = pd.unique(df[col])
            unique_vals.append(list(uq))
        return dict(zip(running_cols, unique_vals)), df

class DataPreprocessing():
    def __init__(
        self,
        df_customers,
        df_levels,
        df_deliveries,
        df_levels_idx,
        df_deliveries_idx,
        is_resample,
    ):

        self._df_customers = df_customers
        self._df_levels = df_levels
        self._df_deliveries = df_deliveries

        self._df_levels_idx = df_levels_idx
        self._df_deliveries_idx = df_deliveries_idx
        self._all_vids = self._df_customers['VESSEL_ID']

        self._grp_lv_vid = self._df_levels_idx.groupby('VESSEL_ID').groups
        self._grp_dl_vid = self._df_deliveries_idx.groupby('VESSEL_ID').groups
        if is_resample is True:
            self._df_levels = self.resample_hourly()
        self._df_levels = self.sort_time()


    @classmethod
    def init_frm_file(cls, is_resample):
        di = DataImport()
        di.read_data_hdf()
        return cls(di._df_customers, di._df_levels,
                   di._df_deliveries, di._df_levels_idx,
                   di._df_deliveries_idx, is_resample=is_resample)
    
    def sort_time(self):
        df_list = []
        for idx_vessel, each_vessel_id in enumerate(self._all_vids):
            print('idx vessel ', idx_vessel)
            s_mask = self._grp_lv_vid[each_vessel_id]
            data = self._df_levels.iloc[s_mask]

            oidx = self._df_levels.iloc[s_mask].index
            print(oidx.is_monotonic)
            if oidx.is_monotonic  == False:
                oidx = sorted(self._df_levels.iloc[s_mask].index)
                oidx = pd.to_datetime(oidx)
                new_data = data.reindex(oidx).copy()
            else:
                new_data = data.copy()
            if oidx.is_monotonic  == False:
                print('ERROR')
            df_list.append(new_data)
        return pd.concat(df_list)


    def resample_hourly(self):
        df_list = []
        for idx_vessel, each_vessel_id in enumerate(self._all_vids):
            print('idx vessel ', idx_vessel)
            s_mask = self._grp_lv_vid[each_vessel_id]
            data = self._df_levels.iloc[s_mask]

            oidx = self._df_levels.iloc[s_mask].index
            print(oidx.is_monotonic)
            if oidx.is_monotonic  == False:
                oidx = sorted(self._df_levels.iloc[s_mask].index)
                oidx = pd.to_datetime(oidx)
                new_data = data.reindex(oidx).copy()
            else:
                new_data = data.copy()
            if oidx.is_monotonic  == False:
                print('ERROR')
            nidx = pd.date_range(oidx.min(), oidx.max(), freq='1H')
            res = new_data.reindex(nidx, method='nearest', limit=1).interpolate()
            df_list.append(res)
        return pd.concat(df_list)


    def smooth_data(self,
                    span=3):
        smoothed_data = np.empty(len(self._df_levels), dtype=np.float64)
        for idx_vessel, each_vessel_id in enumerate(self._all_vids):
            s_mask = self._grp_lv_vid[each_vessel_id]
            df_new = self._df_levels.iloc[s_mask]['INST_PRODUCT_AMOUNT']
            indices = self._df_levels_idx.iloc[s_mask]['INST_PRODUCT_AMOUNT'].index
            df_new = df_new.ewm(span=span).mean()
            smoothed_data[indices]=df_new.values
        self._df_levels['INST_PRODUCT_AMOUNT'] = smoothed_data


    def perform_data_checks(self):
        checks = ['error_level_capacity',
                  #'error_delivery_capacity',
                  'error_current_higher',
                  'error_zero_level',
                  'error_fozen_level',
                  ]
        check_funcs = [self.check_level_smaller_capacity,
                       # self.check_dlvolume_smaller_capacity,
                        self.check_prv_level_smaller_crnt_level,
                       self.check_zero_level,
                       self.check_frozen_level,]
        for idx, (check, check_func) in enumerate(zip(checks, check_funcs)):
            print("checking function ", check)
            self._error_list = np.array([False] * len(self._df_levels),
                              dtype=np.bool)

            for idx_vessel, each_vessel_id in enumerate(self._all_vids):
                print('idx vessel ', idx_vessel  )
                check_func(each_vessel_id)
            print("total errors ", np.sum(self._error_list)/len(self._error_list)*100)
            self._df_levels[check] = self._error_list

    def check_level_smaller_capacity(self, each_vid):
        s_mask = self._grp_lv_vid[each_vid]
        capacity = int(self._df_customers.loc[self._df_customers['VESSEL_ID']
                                                  == each_vid]['MAXIMUM_PRODUCT_CAPACITY'])
        levels = self._df_levels.iloc[s_mask]['INST_PRODUCT_AMOUNT'].values
        indices = self._df_levels_idx.iloc[s_mask]['INST_PRODUCT_AMOUNT'].index
        self._error_list[indices[levels > capacity]]=True

    def check_dlvolume_smaller_capacity(self, each_vid):
        '''Check delivered volume, if its larger than capacity then error.
           If tanker is continuously using then dont mark an error.
           Controlled by hyper parameter of SLOPE'''

        SLOPE = 0.9
        capacity = int(self._df_customers.loc[self._df_customers['VESSEL_ID']
                                                  == each_vid]['MAXIMUM_PRODUCT_CAPACITY'])

        s_mask_lv = self._grp_lv_vid[each_vid]
        s_mask_dl = self._grp_dl_vid[each_vid]
        volumes = self._df_deliveries.iloc[s_mask_dl]['DELIVERED_AMOUNT'].values
        levels = self._df_levels.iloc[s_mask_lv]['INST_PRODUCT_AMOUNT'].values

        gradient = np.gradient(levels)
        norms = np.linalg.norm(gradient, axis=0)
        gradient = [np.where(norms==0,0,i/norms) for i in gradient]

        df_gradient = pd.DataFrame(index = self._df_levels.index,
                                   data = gradient,
                                   columns = ['gradient'])

        time_mask = self._df_deliveries.iloc[s_mask_dl].index[volumes>capacity]
        resample_mask = [False] * len(time_mask)

        for idx, timestamp in enumerate(time_mask):
            # get levels gradient just before the error delivery stamp
            slope = df_gradient.iloc[s_mask_lv].truncate(after=\
                timestamp)[-1]['gradient'].values
            if slope > SLOPE:
                self._df_deliveries.iloc[s_mask_dl].loc[tm]['error_delivery\
                    _capacity'] = True

    def check_prv_level_smaller_crnt_level(self, each_vid):
        ''' If level increases and there is no delivery within a span then
            flag an error. The span interval is hyper parameter
        '''
        INTERVAL = datetime.timedelta(hours=1)
        s_mask_dl = self._grp_dl_vid[each_vid]
        s_mask_lv = self._grp_lv_vid[each_vid]
        levels = self._df_levels.iloc[s_mask_lv]['INST_PRODUCT_AMOUNT'].values
        slopes = np.gradient(levels)
        er_mask = slopes>0
        er_ts = self._df_levels.iloc[s_mask_lv].index[er_mask]
        ds = self._df_deliveries.iloc[s_mask_dl].index
        is_error = self.check_delivery_nearby(er_ts, ds, INTERVAL)
        indices = self._df_levels_idx.iloc[s_mask_lv]['INST_PRODUCT_AMOUNT'].index
        self._error_list[indices[er_mask]] = is_error
        print('vessel errors ', each_vid, np.sum(is_error)/len(indices) * 100)

    def check_zero_level(self, each_vid):
        ''' If next positive value to zero level is also an error then all
            the values between them is also bad data
        '''
        s_mask_lv = self._grp_lv_vid[each_vid]
        er_current_higher =  self._df_levels['error_current_higher']
        levels = self._df_levels.iloc[s_mask_lv]['INST_PRODUCT_AMOUNT'].values
        indices = self._df_levels_idx.iloc[s_mask_lv]['INST_PRODUCT_AMOUNT'].index
        zero_mask = levels==0
        non_zero_mask = np.logical_not(zero_mask)

        zero_indices = indices[zero_mask]
        non_zero_indices = indices[non_zero_mask]

        for z_idx in zero_indices:
            try:
                next_non_zero_idx = non_zero_indices[non_zero_indices>z_idx][0]
            except IndexError:
                pass
            else:
                self._
                if er_current_higher.iloc[next_non_zero_idx] is True:
                    self._error_list[indices[z_idx]]=True
                else:
                    pass

    def check_frozen_level(self, each_vid):
        ''' If cummulated deliveries minus slope exceeds capacity then reading
            is in error
        '''
        s_mask_lv = self._grp_lv_vid[each_vid]
        capacity = int(self._df_customers.loc[self._df_customers['VESSEL_ID']
                                              == each_vid]['MAXIMUM_PRODUCT_CAPACITY'])

        levels = self._df_levels.iloc[s_mask_lv]['INST_PRODUCT_AMOUNT'].values
        df_levels = self._df_levels.iloc[s_mask_lv]
        s_mask_dl = self._grp_dl_vid[each_vid]
        indices = self._df_levels_idx.iloc[s_mask_dl]['INST_PRODUCT_AMOUNT'].index
        dl_start_time = self._df_deliveries.iloc[s_mask_dl].index[0]

        df_levels = df_levels.truncate(before=dl_start_time)
        merge_idx = list(df_levels.index) + list(self._df_deliveries.index)
        merge_idx_s = sorted(list(set(merge_idx)))
        df_deliveries = self._df_deliveries.reindex(merge_idx_s,
                                                    fill_value=0)
        df_deliveries = df_deliveries['DELIVERED_AMOUNT']
        df_deliveries = df_deliveries.reindex(df_levels.index)


        levels = df_levels['INST_PRODUCT_AMOUNT'].values
        slopes = levels - np.roll(levels, -1)
        est_levels = df_deliveries['DELIVERED_AMOUNT'].values - slopes
        mask = est_levels>capacity
        truncated = [False] * (len(df_levels) - len(mask))
        mask = truncated + mask
        self._error_list[indices] = mask

    def check_delivery_nearby(self, error_stamps, delivery_stamps, interval):
        error_stamps_plus = error_stamps + interval
        error_stamps_minus = error_stamps - interval
        out_mask = np.array([True] * len(error_stamps))
        for idx, ds in enumerate(delivery_stamps):
            mn = error_stamps_minus < delivery_stamps[idx]
            ps = error_stamps_plus > delivery_stamps[idx]
            cond = np.logical_and(mn, ps)
            out_mask[cond] = False
        return out_mask

#     def categorize_customers(self):
#         categories = ['continuous usage', 'multiple usage rates',
#                       'steady batch user', 'irregular batch user',
#                       'irregular user', 'periodic variable']
        # perform time continuity check
        # if rate is continuous then continous
        # if rate is not then multiple t_isic_code        # if time is irregular
        # if the slope is non-zero-same and mask pattern is the same OR
        # delivery is periodic
        # the slope is nonzero-same but mask pattern is not same
        # if rate is not the same.........


if __name__ == "__main__":
    dp = DataPreprocessing.init_frm_file(is_resample=False)
    dp.perform_data_checks()
    # di = DataImport()
    # di.read_data_hdf() #     ds = DataSampling(df_deliveries, df_customers, df_colocations, df_levels)
    # di.write_sorted()
    # dp = DataPreprocessing(df_customers, df_levels, df_deliveries)
    # dp.infer_frequency()
#     df_lv, df_dl = ds.sample_level_delivery_vessels('VESSEL_ID', ['BR-216732'])
#     ids = pd.unique(df_customers['VESSEL_ID'])
#     for id_ in ids:
#         df_lv, df_dl = ds.sample_level_delivery_vessels('VESSEL_ID', [id_])
#         print("ID ", id_)
#         print(df_lv.index[0], df_lv.index[-1])
#         print("num_samples ", len(df_lv.index))
