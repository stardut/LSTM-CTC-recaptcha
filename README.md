# OCR

## Captcha

基于[mxnet](https://github.com/dmlc/mxnet)、lstm、ctc的6位 `数字+字符` 验证码识别

代码改于[官方](https://github.com/dmlc/mxnet/blob/master/example/warpctc/lstm_ocr.py)的4位数字验证码识别

1.  将纯4位 `数字` 验证码扩展到6位 `数字+字符` 
2.  修改为训练本地自己的数据，而不是运行中代码生成的数据

![ocr-captchar](https://github.com/Stardust-/ocr/raw/master/MarkdownPic/ocr-captchar.png)

## 参考blog

详细介绍博客： [Stardust 的博客](http://ranjun.me)

http://ranjun.me

## 效果

![effect](https://github.com/Stardust-/ocr/raw/master/MarkdownPic/ocr-effect.png)