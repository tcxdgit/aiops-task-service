from flask import Flask, request
try:
    import ujson as json
except Exception:
    import json

from predict import do_predict
app = Flask(__name__)

@app.route("/safe/getforecast", methods=["POST",])
def r_forecast():
    if request.method == "POST":
        try:
            request_data: dict = json.loads(request.data)
            data = request_data.get("data", None)
            start_time = request_data.get("start_time", None)
            if (data and start_time) == None:
                return json.dumps({"message": "the key of data or start_time error", "status": 210})
            if len(data) > 1:
                # for forecast length and granularity
                duration = int(request_data.get("duration", 2))
                freq = int(request_data.get("freq", 0))
                if freq == 0:
                    prediction_length = duration * 288
                    granularity = "5T"
                    period_num = 288
                elif freq == 1:
                    prediction_length = duration * 24
                    granularity = "1H"
                    period_num = 24
                else:
                    return json.dumps({"message": "key freq error", "status": 210})
                # for predict
                instance = {"start": start_time, "target": data}
                result = do_predict(instance, prediction_length, granularity, period_num)
                if type(result) == str:
                    return json.dumps({"status": 210, "message": "algorithm internal error: " + result})
                else:
                    return json.dumps({"status": 200, "message": "success", "data": result})
            else:
                return json.dumps({"message": "no data", "status": 210})
        except Exception as e:
            return json.dumps({"message": "system error:" + str(e), "status": 500})


