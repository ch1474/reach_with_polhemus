import datetime as dt

datetime_obj = dt.datetime.fromisoformat("2021-06-21 06:04:51.474192")

epoch = dt.datetime.utcfromtimestamp(0)

print((datetime_obj - epoch).total_seconds())
