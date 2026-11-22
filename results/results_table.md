Hardware for latency: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 logical cores, 1 threads, Linux (CPU only, batch size 1)

| Model | Role | Precision | In-scope acc. | OOS recall | Overall acc. | Size (MB) | Latency (ms, mean of N runs) |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | Classical baseline | FP64 (sparse) | 85.6% | 91.6% | 86.7% | 45.61 | 27.08 (N=200) |
| word2vec + FFN | Neural network (own) | FP32 | 76.5% | 74.3% | 76.1% | 4.46 | 0.13 (N=200) |
| BERT-base fine-tuned | Transformer | FP32 | 91.2% | 91.1% | 91.2% | 438.43 | 104.13 (N=200) |
| BERT-base INT8 | Transformer, quantized (extra) | INT8 dynamic | 88.9% | 82.0% | 87.6% | 181.61 | 33.63 (N=200) |
| Distilled BERT-small FP32 | Distilled Transformer (extra) | FP32 | 88.0% | 90.8% | 88.5% | 115.38 | 14.53 (N=200) |
| Distilled BERT-small INT8 | Quantized Transformer (app) | INT8 dynamic | 88.4% | 90.5% | 88.8% | 76.65 | 8.04 (N=200) |
