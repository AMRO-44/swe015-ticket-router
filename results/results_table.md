Hardware for latency: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, 12 logical cores, 10 threads, Windows (CPU only, batch size 1)

| Model | Role | Precision | In-scope acc. | OOS recall | Overall acc. | Size (MB) | Latency (ms, mean of N runs) |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | Classical baseline | FP64 (sparse) | 85.6% | 91.6% | 86.7% | 48.09 | 1.09 (N=200) |
| word2vec + FFN | Neural network (own) | FP32 | 75.6% | 76.8% | 75.8% | 4.45 | 0.05 (N=200) |
| BERT-base fine-tuned | Transformer | FP32 | 91.2% | 91.1% | 91.2% | 438.43 | 104.13 (N=200) |
| BERT-base INT8 | Transformer, quantized (extra) | INT8 dynamic | 88.9% | 82.0% | 87.6% | 181.61 | 33.63 (N=200) |
| Distilled BERT-small FP32 | Distilled Transformer (extra) | FP32 | 88.0% | 90.8% | 88.5% | 115.38 | 14.53 (N=200) |
| Distilled BERT-small INT8 | Quantized Transformer (app) | INT8 dynamic | 88.4% | 90.5% | 88.8% | 76.65 | 8.04 (N=200) |
