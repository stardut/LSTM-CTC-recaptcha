# My Deep Learning Practices

## Recaptcha

###  Introduce

基于[mxnet](https://github.com/dmlc/mxnet)、lstm、ctc的6位 `数字+字符` 验证码识别

同时可以扩展到省份证、银行卡、车牌识别等等

代码改于[官方](https://github.com/dmlc/mxnet/blob/master/example/warpctc/lstm_ocr.py)的4位数字验证码识别

1.  将纯4位 `数字` 验证码扩展到6位 `数字+字符` 
2.  修改为训练本地自己的数据，而不是运行中代码生成的数据

![ocr-captchar](https://github.com/Stardust-/DL-practices/blob/master/MarkdownPic/ocr-captchar.png)

### Main Blog

详细介绍博客： [Stardust的博客](http://ranjun.me)

http://ranjun.me

### Effect

![effect](https://github.com/Stardust-/DL-practices/blob/master/MarkdownPic/ocr-effect.png)