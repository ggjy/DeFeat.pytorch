# VOC Baseline Model Result

## Results and Models

**Notes:**

- Faster RCNN based model
- Batch: sample_per_gpu x gpu_num


| Model | BN | Grad clip | Batch | Lr schd | box AP | Log |
|:-----:|:--:|:---------:|:-----:|:-------:|:------:|:---:|
| R101  | bn | None      |  8x2  | 0.01    | 81.70  |     |
| R101  | bn | None      |  8x2  | 0.02    | 82.17  |     |
| R101  | syncbn | max=35 | 8x2  | 0.01    | 81.59  |     |
| R101  | syncbn | None  |  8x2  | 0.02    | 81.83  |     |
| R50   | bn | max=35    |  8x2  | 0.02    | 80.97  |     |
| R50   | syncbn | None  |  8x2  | 0.02    | 80.76  |     |
| R50   | syncbn | max=35 | 8x2  | 0.01    | 80.66  |     |
| R50   | bn | None      |  8x2  | 0.01    | 80.52  |     |
