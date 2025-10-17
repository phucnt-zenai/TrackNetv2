A copy of repo: https://github.com/wolfyeva/TrackNetV2. Also, I used library ```torch_pruning``` to prune in a structuring way.

### My Model weight
My weights are saved in ```models/```, containing ```full_architecture``` (full-architecture checkpoints) and ```pruning``` (pruned models).
All are trained in batchsize `16` in Kaggle. The pruned models come from pruning and finetuning `model_cur_24epochs_v2.pt`.
### Dataset
Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw. You can use `python3 preprocess.py` to preprocess them.

Alternatively, you can use my work at:  
 - https://www.kaggle.com/datasets/phuc25111/shuttlecock-tracknetv2
 - https://www.kaggle.com/datasets/phuc25111/shuttlecock-tracknetv2-2

### Requirements
```python 
pip install -r requirements.txt
```

### Train TrackNet
```python
python3 train.py --num_frame 3 --epochs 30 --batch_size 10 --learning_rate 0.001 --save_dir exp
```

There are also parameters for resuming or pruning. Please read carefully `train.py` to adjust properly.

### Evaluate TrackNet
I use our self-annotated testset. They stems from public tournaments on Youtube. 

The dataset url: https://www.kaggle.com/datasets/phuc25111/badminton-testset.

```python
python /kaggle/working/TrackNetv2/eval_on_Hau_checked.py --prune --model_file <checkpoint_url> --tolerance 5 --batch_size 8
```

Some results:
| Checkpoint    | Params (M)      | Acc     | Precision     | Recall     | Latency (ms)     |
|---------------|-----------------|---------|---------------|------------|-------------|
| model_cur_8epochs_v2.pt     |<div align="center">10.15</div>   |<div align="center">0.84</div>|<div align="center">0.94</div>|<div align="center">0.87</div>|<div align="center">8.1</div>|
| model_cur_16epochs_v2.pt    |<div align="center">10.15</div>   |<div align="center">0.84</div>|<div align="center">0.94</div>|<div align="center">0.88</div>|<div align="center">8.1</div>|
| model_cur_24epochs_v2.pt    |<div align="center">10.15</div>   |<div align="center">0.84</div>|<div align="center">0.95</div>|<div align="center">0.87</div>|<div align="center">8.1</div>|
| pruning_model_step_1.pt   | <div align="center">8.19</div>     |<div align="center">0.83</div>|<div align="center">0.95</div>|<div align="center">0.85</div>|<div align="center">7.8</div>|
| pruning_model_step_2.pt    | <div align="center">6.47</div>    |<div align="center">0.84</div>|<div align="center">0.95</div>|<div align="center">0.86</div>|<div align="center">7.1</div>|
| pruning_model_step_3.pt    | <div align="center">4.96</div>     |<div align="center">0.81</div>|<div align="center">0.94</div>|<div align="center">0.84</div>|<div align="center">5.9</div>|
| pruning_model_step_4.pt   | <div align="center">3.64</div>      |<div align="center">0.83</div>|<div align="center">0.94</div>|<div align="center">0.86</div>|<div align="center">5.2</div>|



 *Note: I measured in batchsize `1`, if increase batchsize, you will gain a better mean latency. The latency herein is just the average per frame, meanwhile we forward 3 frames per batch for model inputs.*


### Example code
- Training: https://www.kaggle.com/code/nguyenthangphuc/tracknetv2-test-shuttlecock-finetune-fec3c6
- Evaluation: https://www.kaggle.com/code/nguyenthangphuc/test-tracknetv2

<!--*This repo is as a part of VNPT_Media project*-->
