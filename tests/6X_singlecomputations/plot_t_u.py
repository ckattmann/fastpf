import json

with open('results1.json') as f:
    data = json.load(f)

for method,bench in data['feeder'].items():
    print(method)

