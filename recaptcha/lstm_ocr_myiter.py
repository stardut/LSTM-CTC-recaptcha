import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll

from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random
import os


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


maps = {}
maps_value = 11

for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    maps[char] = maps_value
    maps_value += 1


def get_label(buf):
    global maps
    ret = np.zeros(len(buf))
    for i in range(len(buf)):
        if is_num(buf[i]):
            ret[i] = int(buf[i]) + 1
        else:
            ret[i] = maps[buf[i]]
    return ret


class OCRIter(mx.io.DataIter):
    def __init__(self, batch_size, num_label, init_states, path, check):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.num_label = num_label
        self.init_states = init_states
        self.path = path
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 100 * 30))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.check = check

    def __iter__(self):
        print 'iter'
        init_state_names = [x[0] for x in self.init_states]
        dir_list = os.listdir(self.path)
        pic_num = len(dir_list)
        num = 0
        for k in range(pic_num / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                if num > pic_num-1:
                    break
                img = cv2.imread(self.path + '/' + dir_list[num], 0)
                lable_value = dir_list[num].split('.')[0]
                lable_value = lable_value.split('_')[1]
                num += 1
                img = cv2.resize(img, (100, 30))
                img = img.transpose(1, 0) 
                img = img.reshape((100 * 30))
                img = np.multiply(img, 1/255.0)

                data.append(img)
                label.append(get_label(lable_value))

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']            
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass



BATCH_SIZE = 100
SEQ_LENGTH = 30

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = label[i]
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total


if __name__ == '__main__':
    num_hidden = 128
    num_lstm_layer = 2
    num_epoch = 10
    learning_rate = 0.01
    momentum = 0.9
    num_label = 6

    prefix = 'mymodel/mymodel'
    iteration = 5
    contexts = [mx.context.gpu(0)]

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                           num_hidden= num_hidden,
                           num_label = num_label)

    init_c = [('l%d_init_c'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    train_path  = '/home/xuyingjie/Captcha-hacker-master/gen_images/img_data6/train_data'
    test_path  = '/home/xuyingjie/Captcha-hacker-master/gen_images/img_data6/test_data'

    data_train = OCRIter(BATCH_SIZE, num_label, init_states, train_path, 'train')
    data_val = OCRIter(BATCH_SIZE, num_label, init_states, test_path, 'test')

    symbol = sym_gen(SEQ_LENGTH)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    # load model
    # model = mx.model.FeedForward.load(prefix,iteration,
    #                                 learning_rate = learning_rate,
    #                                 ctx = contexts, 
    #                                 numpy_batch_size = BATCH_SIZE,
    #                                 num_epoch=num_epoch)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    print 'begin fit'

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 100))
    
    model.save(prefix, iteration)
