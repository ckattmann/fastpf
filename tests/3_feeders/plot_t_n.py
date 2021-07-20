import json
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.style.use('~/Dropbox/diss/diss.mplstyle')

with open('results1.json') as f:
    data = json.load(f)

plt.subplots()

methods = list(data[list(data.keys())[0]].keys())
print(methods)
runtimes = {}

for n,bench in data.items():
    print(n)
    n = int(n)
    for method,d in bench.items():
        print(method)
        print(d['runtime'])
        if method in runtimes:
            runtimes[method].append((n,d['runtime']))
        else:
            runtimes[method] = [(n,d['runtime'])]

for m, rt in runtimes.items():
    # nodes, runtime = rt
    nodes = [t[0] for t in rt]
    runtime = [t[1] for t in rt]
    print(nodes,runtime)
    plt.semilogy(nodes,runtime,'o--', label=m)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
