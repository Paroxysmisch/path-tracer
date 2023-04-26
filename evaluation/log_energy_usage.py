from nvitop import CudaDevice, ResourceMetricCollector, collect_in_background
import os
import atexit
import pickle

scene = '0'
spp = '4096'
adap_samp = 'off'
tiling = '16'
run = '4'

filename = 'log_' + scene + '_' + spp + '_' + adap_samp + '_' + tiling + '_' + run

data = []

def on_exit():
    print('Saved data to: ', filename)
    with open(filename, 'wb+') as f:
        pickle.dump(data, f)

def on_collect(metrics):
    data.append(metrics)
    return True

def on_stop(collector):
    return

atexit.register(on_exit)

collect_in_background(
    on_collect,
    ResourceMetricCollector(CudaDevice.all()),
    interval=1.0,
    on_stop=on_stop,
)

while(True):
    continue
