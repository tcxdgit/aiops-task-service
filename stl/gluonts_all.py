# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:01:30 2019

@author: x19860
"""
import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.model.prophet import ProphetPredictor, PROPHET_IS_INSTALLED
from gluonts.model.r_forecast import RForecastPredictor
#from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.npts import NPTSPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

def trans_df2gluon(dflist,freq="1H"):
    datadictlist = []
    for data in dflist:
        rawdata  =data.copy()
        rawdata['ds'] = pd.to_datetime(rawdata['ds'])
        rawdata.set_index('ds',inplace=True)
        rawdata = rawdata.resample(freq).mean()
        data_dict = {"start":rawdata.index[0],"target":rawdata.y}
        datadictlist.append(data_dict)
    dataset = ListDataset(datadictlist,freq=freq)
    return dataset

def get_res(fcstList,CI=90):
    outputList = []
    for fcst in fcstList:
        sorted_samples = fcst._sorted_samples
        num_samples = len(sorted_samples)
        q_upper = (100+CI)/200
        sample_idx_upper = int(np.round((num_samples - 1) * q_upper))
        yhat_upper = sorted_samples[sample_idx_upper]
        q_lower = (100-CI)/200
        sample_idx_lower = int(np.round((num_samples - 1) * q_lower))
        yhat_lower = sorted_samples[sample_idx_lower]
        yhat = fcst.mean
        ds = fcst.index
        output = pd.DataFrame({"ds":ds,"yhat":yhat,"y_upper":yhat_upper,"y_lower":yhat_lower})
        outputList.append(output)
    return outputList

def gluonts_prophet(dataset,freq,pred_length,prophet_params={}):
    
    params = dict(freq=freq, prediction_length=pred_length, prophet_params=prophet_params)
    predictor = ProphetPredictor(**params)
    fcst = predictor.predict(dataset)
    fcstlist = []
    for i in fcst:
        fcstlist.append(i)
    return fcstlist

def gluonts_r(dataset,freq, pred_length,period=None,trunc_length=None, method_name = "ets"):
    
    params = dict(freq=freq, prediction_length=pred_length, method_name = method_name)
    predictor = RForecastPredictor(**params)
    fcst = predictor.predict(dataset)
    fcstlist = []
    for i in fcst:
        fcstlist.append(i)
    return fcstlist

def gluonts_npts(dataset,freq, pred_length,context_length=None,
                 kernel_type="exponential",exp_kernel_weights=1.0,
                 use_seasonal_model = True):
    predictor = NPTSPredictor(freq=freq, prediction_length=pred_length,
                              context_length=context_length,
                              kernel_type=kernel_type,
                              exp_kernel_weights=exp_kernel_weights,
                              use_seasonal_model = use_seasonal_model)
    fcst = predictor.predict(dataset)
    fcstlist = []
    for i in fcst:
        fcstlist.append(i)
    return fcstlist

def gluonts_seasonal_naive(dataset,freq, pred_length, season_length=None):
    predictor = SeasonalNaivePredictor(freq=freq,prediction_length = pred_length,
                                       season_length=season_length)
    fcst = predictor.predict(dataset)
    fcstlist = []
    for i in fcst:
        fcstlist.append(i)
    return fcstlist

def gluonts_predict(datalist,freq,pred_length,context_length=None,
                    method_name = "r_ets",CI = 80,params={}):
    if method_name not in ["prophet","r_ets","r_arima","r_tbats","r_croston","r_mlp",
                           "npts","seasonal_naive"]:
        return("method not support")
    dataset = trans_df2gluon(datalist,freq=freq)
    if method_name[0:2] == "r_":
        method = method_name[2:]
        if "period" in params:
            period = params["period"]
        else:
            period = None
        fcstlist = gluonts_r(dataset=dataset,freq=freq, pred_length=pred_length,
                             period=period,
                             trunc_length=context_length, method_name = method)
        outputList = get_res(fcstlist,CI=CI)
        
    elif method_name == "prophet":
        fcstlist = gluonts_prophet(dataset=dataset,freq=freq,pred_length=pred_length,
                                   prophet_params=params)
        outputList = get_res(fcstlist,CI=CI)
        
    elif method_name == "npts":
        kernel = params["kernel"] if "kernel" in params else "exponential"
        fcstlist = gluonts_npts(dataset = dataset,freq=freq, pred_length=pred_length,
                                context_length=context_length,
                                kernel_type=kernel)
        outputList = get_res(fcstlist,CI=CI)
    
    elif method_name == "seasonal_naive":
        season_length = params["season_length"] if "season_length" in params else 24
        fcstlist = gluonts_seasonal_naive(dataset = dataset,freq=freq, pred_length=pred_length,
                                          season_length=season_length)
        outputList = get_res(fcstlist,CI=CI)
    else:
        outputList = []
    return outputList
    


    


    