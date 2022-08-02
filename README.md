# ai

## setup

- dependencies
```
!if cd ai; then git pull; else git clone https://github.com/BambooTuna/ai.git ai; fi

import sys
sys.path.append("./ai/utils")
```

- dataloader

```
from google.colab import auth
auth.authenticate_user()
!gsutil cp gs://???.tar.gz images.tar.gz
!mkdir -p images/category && cd images/category && tar -zxvf /content/images.tar.gz


import data_tools
from torch.utils.data import DataLoader
dataset = data_tools.load_dataset("/content/images")
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
image = next(iter(dataloader))[0]
data_tools.show_image(image)
```