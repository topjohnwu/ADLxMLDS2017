# HW1

My scripts already contains a token to use `curl` to download the file from Gthub, but in case that the model download fails, you can download the file `models.tgz` with URL [https://github.com/topjohnwu/ADLxMLDS2017/releases/download/hw1/models.tgz](https://github.com/topjohnwu/ADLxMLDS2017/releases/download/hw1/models.tgz) (note: this URL only works in browsers that is authenticated).

The models should be extracted into a folder called `models`, the complete directory structure to run the scripts are shown below:

```
├── data
│   ├── 48phone_char.map
│   ├── fbank
│   │   ├── test.ark
│   │   └── train.ark
│   ├── label
│   │   └── train.lab
│   ├── mfcc
│   │   ├── test.ark
│   │   └── train.ark
│   └── phones
│       └── 48_39.map
├── hw1_best.sh
├── hw1_cnn.sh
├── hw1_rnn.sh
├── models
│   ├── cnn.ckpt.data-00000-of-00001
│   ├── cnn.ckpt.index
│   ├── cnn.ckpt.meta
│   ├── rnn.ckpt.data-00000-of-00001
│   ├── rnn.ckpt.index
│   └── rnn.ckpt.meta
└── run.py
```