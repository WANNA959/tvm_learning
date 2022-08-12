# -*- coding: utf-8 -*-
import plot_tools
import os
import numpy as np
import time

import tvm
from tvm import relay
import tvm.relay.testing
from tvm.contrib import graph_runtime
import tvm.contrib.graph_runtime as runtime


_target = "cuda"
_device = tvm.cuda(0)
_dtype = "float32"


def get_network(name, batch_size):
    """获取网络符号定义和随机权重"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if 'resnet' in name:
        n_layer = int(name.split('-')[1])  # 获取层数
        net, params = relay.testing.resnet.get_workload(
            nnm_layers=n_layer, batch_size=batch_size, dtype=_dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=_dtype)
    else:
        raise ValueError("Unsupported network: " + name)
    return net, params, input_shape, output_shape


def cul_model_time() -> dict:
    nums = [11, 13, 16, 19]
    data = {}
    for num in nums:
        net_name = 'vgg-{}'.format(num)
        print("model {}".format(net_name))
        net, params, input_shape, out_shape = get_network(
            net_name, batch_size=5)
        start = time.time()
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(net, target=_target, params={})
        build_time = 1000*(time.time()-start)
        print("build time:", build_time)

        start = time.time()
        module = graph_runtime.create(graph, lib, _device)
        create_time = 1000*(time.time()-start)
        print("create time:", create_time)

        data_tvm = tvm.nd.array(
            (np.random.uniform(size=input_shape)).astype(_dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        print("Evaluate inference time cost...")
        res = []
        for i in range(10):
            start = time.time()
            module.run()
            run_time = 1000*(time.time()-start)
            print("run time:", run_time)
            res.append(run_time)
        data[net_name] = {'build': build_time,
                          'create': create_time, 'run': res}
    return data


if __name__ == '__main__':

    data = cul_model_time()
    plot_tools.plot_model_time(data)


# dev = tvm.device(str(_target), 0)
# ftimer = module.module.time_evaluator("run", dev, number=1, repeat=50)
# prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
# print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
#       (np.mean(prof_res), np.std(prof_res)))
