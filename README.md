# Benchmark BERT performance with TVM

The script compares BERT performance between TVM and mxnet-mkl on c5.9x and a1.4x instances.
AutoTVM logs are tuned on c5.9x and a1.4x for sequence length 128, 256, 384.

## Prerequisite
Install the python packages.
```
pip install --upgrade mxnet-mkl>=1.5.0b20190630
pip install gluonnlp
```

Compile the tvm from `https://github.com/icemelon9/tvm/tree/bert-static`.


## Instruction

Export the BERT model, run

```
./export_model.sh seq_length
```

Run benchmark
```
python benchmark.py --task [task] --seq_length [seq_length] [--arm]
```