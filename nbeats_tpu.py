'''
Go to Google Colab and enable TPU accelarator from resources.

Install those:
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchtext==0.10.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
!pip install 'u8darts[torch]==0.17.1'
!pip install pyyaml==5.4.1

and then run in a cell:
'''

import numpy as np

from darts import TimeSeries
from darts.models import NBEATSModel

a = np.empty([200], dtype = np.float32)
series = TimeSeries.from_values(a)

trainset_size = 0.6
train, val = series.split_after(trainset_size)

model_nbeats = NBEATSModel(
  input_chunk_length=16,
  output_chunk_length=4,
  n_epochs=11,
  pl_trainer_kwargs={
    "accelerator": "tpu",
    "tpu_cores": [4]
  },
)
model_nbeats.fit(series=train, val_series=val, verbose=True)

'''
which outputs:
[2022-02-23 12:43:12,041] INFO | darts.models.forecasting.torch_forecasting_model | Train dataset contains 101 samples.
[2022-02-23 12:43:12,041] INFO | darts.models.forecasting.torch_forecasting_model | Train dataset contains 101 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 101 samples.
[2022-02-23 12:43:13,013] INFO | darts.models.forecasting.torch_forecasting_model | Time series values are 32-bits; casting model to float32.
[2022-02-23 12:43:13,013] INFO | darts.models.forecasting.torch_forecasting_model | Time series values are 32-bits; casting model to float32.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 32-bits; casting model to float32.
[2022-02-23 12:43:13,021] WARNING | darts.models.forecasting.torch_forecasting_model | DeprecationWarning: kwarg `verbose` is deprecated and will be removed in a future Darts version. Instead, control verbosity with PyTorch Lightning Trainer parameters `enable_progress_bar`, `progress_bar_refresh_rate` and `enable_model_summary` in the `pl_trainer_kwargs` dict at model creation.
[2022-02-23 12:43:13,021] WARNING | darts.models.forecasting.torch_forecasting_model | DeprecationWarning: kwarg `verbose` is deprecated and will be removed in a future Darts version. Instead, control verbosity with PyTorch Lightning Trainer parameters `enable_progress_bar`, `progress_bar_refresh_rate` and `enable_model_summary` in the `pl_trainer_kwargs` dict at model creation.
WARNING:darts.models.forecasting.torch_forecasting_model:DeprecationWarning: kwarg `verbose` is deprecated and will be removed in a future Darts version. Instead, control verbosity with PyTorch Lightning Trainer parameters `enable_progress_bar`, `progress_bar_refresh_rate` and `enable_model_summary` in the `pl_trainer_kwargs` dict at model creation.
GPU available: False, used: False
TPU available: True, using: [4] TPU cores
IPU available: False, using: 0 IPUs

  | Name      | Type       | Params
-----------------------------------------
0 | criterion | MSELoss    | 0     
1 | stacks    | ModuleList | 6.1 M 
-----------------------------------------
6.1 M     Trainable params
1.4 K     Non-trainable params
6.1 M     Total params
24.530    Total estimated model params size (MB)
Epoch 10: 100%
4/4 [05:07<00:00, 76.94s/it, loss=0.136]
<darts.models.forecasting.nbeats.NBEATSModel at 0x7f57a3858e50>


Useful links:
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
https://pytorch-lightning.readthedocs.io/en/stable/advanced/tpu.html#tpu-core-training
https://gsamaras.wordpress.com/code/n-beats-with-tpu-accelerator/
'''
