import pandas as pd
import datetime
from   stl.gluonts_all import gluonts_predict

def do_predict(instance, prediction_length, freq, period_num):
   try:
        start = instance["start"]
        target = instance["target"]
        ds = pd.date_range(start=start, periods=len(target), freq=freq)
        raw_df = pd.DataFrame({"ds": ds, "y": target})
        data_list = [raw_df, ]
        # raw_df.set_index("ds", inplace=True)
        return_data = gluonts_predict(data_list, freq, pred_length=prediction_length, params={"period": period_num,})
        if type(return_data) == str:
            return str(return_data)
        result_df = return_data[0]
        result = {}
        result["cloumns_name"] = ["ds", "yaht", "y_upper", "y_lower"]
        data = []
        for _, values in result_df.iterrows():
            _data = []
            _data.append(int((values["ds"] - datetime.timedelta(hours=8)).timestamp()))
            _data.append(values["yhat"])
            _data.append(values["y_upper"])
            _data.append(values["y_lower"])
            data.append(_data)
        result["data"] = data
        return result
   except Exception as e:
       return str(e)