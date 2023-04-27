from nvitop import CudaDevice, ResourceMetricCollector, collect_in_background
import os
import atexit
import pickle

scene = '2'
spp = '256'
adap_samp = 'off'
tiling = '2_128'
run = '2'

filename = 'log_' + scene + '_' + spp + '_' + adap_samp + '_' + tiling + '_' + run

data = []

def on_exit():
    print('Saved data to: ', filename)
    with open(filename, 'wb+') as f:
        pickle.dump(data, f)

def on_collect(metrics):
    new_metrics = {'time' : metrics['metrics-daemon/duration (s)'],
                   'power' : metrics['metrics-daemon/cuda:0 (gpu:0)/power_usage (W)/mean'],
                   'memory' : metrics['metrics-daemon/cuda:0 (gpu:0)/memory_used (MiB)/mean']}
    data.append(new_metrics)
    return True

def on_stop(collector):
    return

atexit.register(on_exit)

collect_in_background(
    on_collect,
    ResourceMetricCollector(CudaDevice.all()),
    interval=5.0,
    on_stop=on_stop,
)

while(True):
    continue
