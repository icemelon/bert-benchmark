import os
import time
import argparse
import numpy as np
import mxnet as mx
import tvm
from tvm import autotvm
from tvm import relay
import tvm.contrib.graph_runtime as runtime

from bert import *

parser = argparse.ArgumentParser(description="Tune/evaluate Bert model")
parser.add_argument("--seq_length", type=int, default=384,
                    help="sequence length (default: 384)")
parser.add_argument("--task", choices=["classification", "regression", "question_answering"],
                    default="question_answering",
                    help="specify the model type (default: question_answering)")
parser.add_argument("--arm", action="store_true", help="Eval on ARM CPU")
args = parser.parse_args()

print("Benchmark BERT for %s with sequence length %s" % (args.task, args.seq_length))

prefix = 'models/%s/%s' % (args.seq_length, args.task)
shape_dict = {
    'data0': (1, args.seq_length),
    'data1': (1, args.seq_length),
    'data2': (1,)
}
inputs = np.random.uniform(size=(1, args.seq_length)).astype('float32')
token_types = np.random.uniform(size=(1, args.seq_length)).astype('float32')
valid_length = np.asarray([args.seq_length]).astype('float32')

mx_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                         ['data0', 'data1', 'data2'],
                                         prefix + '-0000.params')
inputs_nd = mx.nd.array(inputs)
token_types_nd = mx.nd.array(token_types)
valid_length_nd = mx.nd.array(valid_length)
mx_out = mx_net(inputs_nd, token_types_nd, valid_length_nd)
mx_out.wait_to_read()

min_repeat_ms = 1000
number = 20
while True:
    beg = time.time()
    for _ in range(number):
        mx_out = mx_net(inputs_nd, token_types_nd, valid_length_nd)
        mx_out.wait_to_read()
    end = time.time()
    lat = (end - beg) * 1e3
    if lat >= min_repeat_ms:
        break
    number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
print('mxnet mean lat: %.2f ms' % (lat / number))

mod, params = relay.frontend.from_mxnet(mx_net, shape_dict)
ctx = tvm.cpu()
if args.arm:
    target = "llvm -device=arm_cpu -target=aarch64-linux-gnu"
else:
    target = "llvm -mcpu=skylake-avx512"

log_path = "autotvm_logs"
logs = [os.path.join(log_path, f) for f in os.listdir(log_path)]
autotvm_ctx = autotvm.apply_history_best(None)
for log_file in logs:
    autotvm_ctx.load(log_file)

# apply logs
print("Compile...")
with autotvm_ctx:
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod[mod.entry_func], target, params=params)

# benchmark
print("Check correctness...")
ex = runtime.create(graph, lib, ctx)
ex.set_input(data0=inputs, data1=token_types, data2=valid_length, **params)
ex.run()
out = ex.get_output(0)
tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3)

print("Benchmarking...")
ftimer = ex.module.time_evaluator("run", ctx, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
