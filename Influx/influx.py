import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from secrets import INFLUX_TOKEN

token = INFLUX_TOKEN # replace with your token
org = "personal"
url = "http://localhost:8086"

write_client = influxdb_client.InfluxDBClient(url=url, token=token)
bucket="personal"
write_api = write_client.write_api(write_options=SYNCHRONOUS)

# write data to Influx
print()
print("Write data to Influx")
for value in range(10):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org="personal", record=point)
  time.sleep(1) # separate points by 1 second

# query data
print()
print("Query all written data")
query_api = write_client.query_api()
query = """from(bucket: "personal")
 |> range(start: -10m)
 |> filter(fn: (r) => r._measurement == "measurement1")"""
tables = query_api.query(query, org="personal")

print(tables)
for table in tables:
  for record in table.records:
    print(record)

# query data
print()
print("Query all the data mean")
# Aggregate functions take the values of all rows in a table and use them to perform an aggregate operation. 
# The result is output as a new value in a single-row table.
query = """from(bucket: "personal")
  |> range(start: -10m)
  |> filter(fn: (r) => r._measurement == "measurement1")
  |> mean()"""
tables = query_api.query(query, org="personal")

for table in tables:
    for record in table.records:
        print(record)  
