import warnings
warnings.filterwarnings("ignore")

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import os
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard

embed_size = 331
source_text = np.load("/home/jack/201804RNN/king_train/train.npz")


source_text = list(source_text['arr_0'])
input_feature = []
output_feature = []
input_features = []
output_features = []
for sentences in source_text:
    for sentence in sentences:
        input_feature.append(sentence[:-1])
        output_feature.append(sentence[-1])
    input_features.append(input_feature)
    output_features.append(output_feature)
    input_feature = []
    output_feature = []

def pad_sequences(feature, max_length, is_target=False):
    pad_input_feature = feature
    if is_target:
        pad_input_feature.append(36)
        pad = 34
        # 不足长度的句子进行"<PAD>"
        if len(pad_input_feature) < (max_length + 1):
            pad_input_feature = pad_input_feature + [pad] * (max_length + 1 - len(pad_input_feature))
            return pad_input_feature
        else:
            return pad_input_feature[:max_length + 1]
    else:
        pad = [np.float32(0)] * 331
        # 不足长度的句子进行"<PAD>"
        if len(pad_input_feature) < max_length:
            pad_input_feature = pad_input_feature + [pad] * (max_length - len(pad_input_feature))
            return pad_input_feature
        else:
            return pad_input_feature[:max_length]

def pad_sequences_target_input(feature, max_length):
    pad_input_feature = [35]
    pad_input_feature.extend(feature)
    pad = 34
    # 不足长度的句子进行"<PAD>"
    if len(pad_input_feature) < (max_length + 1):
        pad_input_feature = pad_input_feature + [pad] * (max_length + 1 - len(pad_input_feature))
        return pad_input_feature
    else:
        return pad_input_feature[:max_length + 1]

input_length = 20
Tx = input_length
input_ = []
for feature in input_features:
    input_.append(pad_sequences(feature, Tx, is_target=False))

Ty = input_length
output_ = []
output_target_input = []

for feature in output_features:
    output_.append(pad_sequences(feature, Ty, is_target=True))
    output_target_input.append(pad_sequences_target_input(feature, Ty))
X = np.array(input_, dtype=np.float32)
Y = np.array(output_, dtype=np.float32)
Yc = np.array(output_target_input, dtype=np.float32)

# 对Y做One Hot Encoding
Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=37), Y)), dtype=np.float32)

# 自定义softmax函数
def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

# 定义全局网络层对象
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation="tanh")
densor_relu = Dense(1, activation="relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)

def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """

    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])

    return context

# 获取Embedding layer
embedding_layer = Embedding(36 + 1, 100, trainable=True)
embedding_layer.build((None,))

n_a = 32  # The hidden size of Bi-LSTM
n_s = 128  # The hidden size of LSTM in Decoder
decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(37, activation=softmax)

# 定义网络层对象（用在model函数中）
reshapor = Reshape((1, 100))
concator = Concatenate(axis=-1)

def slice(x, index):
    return x[:, index, :]

def model(Tx, Ty, n_a, n_s):
    """
    构造模型

    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    """

    # 定义输入层
    X = Input(shape=(Tx, embed_size))
    # Decoder端LSTM的初始状态
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    # Decoder端LSTM的初始输入
    Y = Input(shape=(Ty + 1,))
    embed = embedding_layer(Y)

    s = s0
    c = c0

    # 模型输出列表，用来存储翻译的结果
    outputs = []

    # 定义LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    # a = LSTM(n_a, return_sequences=True)(X)
    # Decoder端，迭代Ty+1轮，每轮生成一个翻译结果
    for t in range(Ty + 1):
        # 获取Context Vector
        context = one_step_attention(a, s)

        # 将Context Vector与decode_input进行concat
        decoder_input = reshapor(Lambda(slice, output_shape=(None, 1, 100), arguments={'index': t})(embed))
        context = concator([context, decoder_input])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])

        # 将LSTM的输出结果与全连接层链接
        out = output_layer(s)

        # 存储输出结果
        outputs.append(out)

    model = Model([X, s0, c0, Y], outputs)

    return model

model = model(Tx, Ty, n_a, n_s)

model.summary()

plot_model(model, to_file='birnn_attention_keras_mahjong_len20.png', show_shapes=True)

out = model.compile(optimizer=Adam(lr=0.0001),
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

m = X.shape[0]
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))
# model.load_weights("mahjong_seq2seq_model_len20_bi_w.h5")
# 训练模型
model.fit([X, s0, c0, Yc], outputs, epochs=1, batch_size=32,
          callbacks=[TensorBoard(log_dir='mahjong_log_len20_bi/')])

# 保存参数
# model.save_weights("mahjong_seq2seq_model_len8_w.h5")
# model.save("mahjong_seq2seq_model_len8.h5")
# print('Saving Log -------------')
# with open('./mahjong_log/mahjong_seq2seq.log', 'w') as f:
#     f.write(str(out.history))
# print('\nTesting ------------')
# # outputs = list(Yoh[-200:].swapaxes(0,1))
# inputs = [X, s0, c0, Yc]
# results = model.evaluate(x=inputs, y=outputs)
# print(np.shape(results))
# print(results)
# print('\ntest loss: ', loss)
# print('\ntest accuracy: ', accuracy)
# 绘制acc-loss曲线
# history.loss_plot('epoch')

# model.load_weights("mahjong_seq2seq_model.h5")
# def make_prediction(your_sentences):
#     sentences = your_sentences.split('\n')
#     input_feature = []
#     output_feature = []
#     input_features = []
#     output_features = []
#     for sentence in sentences:
#         input_list = [float(x.strip()) for x in sentence.strip().split() if x.strip()][:-1]
#         output_list = [float(x.strip()) for x in sentence.strip().split() if x.strip()][-1]
#         input_feature.append(input_list)
#         output_feature.append(output_list)
#
#     input_features.append(input_feature)
#     output_features.append(output_feature)
#
#     input_ = []
#
#     for feature in input_features:
#         input_.append(pad_sequences(feature, Tx, is_target=False))
#
#     output_ = []
#     output_target_input = []
#
#     for feature in output_features:
#         output_.append(pad_sequences(feature, Ty, is_target=True))
#         output_target_input.append(pad_sequences_target_input(feature, Ty))
#
#     X = np.array(input_)
#     Y = np.array(output_)
#     Yc = np.array(output_target_input)
#
#     # 对Y做One Hot Encoding
#     # Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=37), Y)))
#
#     m = X.shape[0]
#     s0 = np.zeros((m, n_s))
#     c0 = np.zeros((m, n_s))
#
#     # 预测结果
#     preds = model.predict([X, s0, c0, Yc])
#     predictions = np.argmax(preds, axis=-1)
#
#     idx = [idx[0] for idx in predictions]
#
#     # 返回最后一个结果
#     return idx[-1]
#
# your_sentences = input("Please input your sentences: ")
# print(make_prediction(your_sentences))

def runepoch(source_text, i):
    global model
    source_text = list(source_text['arr_0'])
    input_feature = []
    output_feature = []
    input_features = []
    output_features = []
    for sentences in source_text:
        for sentence in sentences:
            input_feature.append(sentence[:-1])
            output_feature.append(sentence[-1])
        input_features.append(input_feature)
        output_features.append(output_feature)
        input_feature = []
        output_feature = []

    def pad_sequences(feature, max_length, is_target=False):
        pad_input_feature = feature
        if is_target:
            pad_input_feature.append(36)
            pad = 34
            # 不足长度的句子进行"<PAD>"
            if len(pad_input_feature) < (max_length + 1):
                pad_input_feature = pad_input_feature + [pad] * (max_length + 1 - len(pad_input_feature))
                return pad_input_feature
            else:
                return pad_input_feature[:max_length + 1]
        else:
            pad = [np.float32(0)] * 331
            # 不足长度的句子进行"<PAD>"
            if len(pad_input_feature) < max_length:
                pad_input_feature = pad_input_feature + [pad] * (max_length - len(pad_input_feature))
                return pad_input_feature
            else:
                return pad_input_feature[:max_length]

    def pad_sequences_target_input(feature, max_length):
        pad_input_feature = [35]
        pad_input_feature.extend(feature)
        pad = 34
        # 不足长度的句子进行"<PAD>"
        if len(pad_input_feature) < (max_length + 1):
            pad_input_feature = pad_input_feature + [pad] * (max_length + 1 - len(pad_input_feature))
            return pad_input_feature
        else:
            return pad_input_feature[:max_length + 1]

    input_length = 20
    Tx = input_length
    input_ = []
    for feature in input_features:
        input_.append(pad_sequences(feature, Tx, is_target=False))

    Ty = input_length
    output_ = []
    output_target_input = []

    for feature in output_features:
        output_.append(pad_sequences(feature, Ty, is_target=True))
        output_target_input.append(pad_sequences_target_input(feature, Ty))
    X = np.array(input_, dtype=np.float32)
    Y = np.array(output_, dtype=np.float32)
    Yc = np.array(output_target_input, dtype=np.float32)

    # 对Y做One Hot Encoding
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=37), Y)), dtype=np.float32)
    n_a = 32  # The hidden size of Bi-LSTM
    n_s = 128  # The hidden size of LSTM in Decoder
    m = X.shape[0]
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))
    # model.load_weights("mahjong_seq2seq_model_len20_bi.h5")
    # 训练模型
    model.fit([X, s0, c0, Yc], outputs, epochs=1, batch_size=32,
              callbacks=[TensorBoard(log_dir='mahjong_log_len20_bi/')])

    # 保存参数
    if i%60 == 5:
        model.save_weights("mahjong_seq2seq_model_len20_bi_w.h5")
        model.save("mahjong_seq2seq_model_len20_bi.h5")

    # print('Saving Log -------------')
    # with open('./mahjong_log/mahjong_seq2seq.log', 'w') as f:
    #     f.write(str(out.history))
    # print('\nTesting ------------')
    # # outputs = list(Yoh[-200:].swapaxes(0,1))
    # inputs = [X, s0, c0, Yc]
    # results = model.evaluate(x=inputs, y=outputs)
    # print(np.shape(results))
    # print(results)

def runeval(source_text):
    global model
    source_text = list(source_text['arr_0'])
    input_feature = []
    output_feature = []
    input_features = []
    output_features = []
    for sentences in source_text:
        for sentence in sentences:
            input_feature.append(sentence[:-1])
            output_feature.append(sentence[-1])
        input_features.append(input_feature)
        output_features.append(output_feature)
        input_feature = []
        output_feature = []

    def pad_sequences(feature, max_length, is_target=False):
        pad_input_feature = feature
        if is_target:
            pad_input_feature.append(36)
            pad = 34
            # 不足长度的句子进行"<PAD>"
            if len(pad_input_feature) < (max_length + 1):
                pad_input_feature = pad_input_feature + [pad] * (max_length + 1 - len(pad_input_feature))
                return pad_input_feature
            else:
                return pad_input_feature[:max_length + 1]
        else:
            pad = [np.float32(0)] * 331
            # 不足长度的句子进行"<PAD>"
            if len(pad_input_feature) < max_length:
                pad_input_feature = pad_input_feature + [pad] * (max_length - len(pad_input_feature))
                return pad_input_feature
            else:
                return pad_input_feature[:max_length]

    def pad_sequences_target_input(feature, max_length):
        pad_input_feature = [35]
        pad_input_feature.extend(feature)
        pad = 34
        # 不足长度的句子进行"<PAD>"
        if len(pad_input_feature) < (max_length + 1):
            pad_input_feature = pad_input_feature + [pad] * (max_length + 1 - len(pad_input_feature))
            return pad_input_feature
        else:
            return pad_input_feature[:max_length + 1]

    input_length = 20
    Tx = input_length
    input_ = []
    for feature in input_features:
        input_.append(pad_sequences(feature, Tx, is_target=False))

    Ty = input_length
    output_ = []
    output_target_input = []

    for feature in output_features:
        output_.append(pad_sequences(feature, Ty, is_target=True))
        output_target_input.append(pad_sequences_target_input(feature, Ty))
    X = np.array(input_, dtype=np.float32)
    Y = np.array(output_, dtype=np.float32)
    Yc = np.array(output_target_input, dtype=np.float32)

    # 对Y做One Hot Encoding
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=37), Y)), dtype=np.float32)
    n_a = 32  # The hidden size of Bi-LSTM
    n_s = 128  # The hidden size of LSTM in Decoder
    m = X.shape[0]
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))
    # model.load_weights("mahjong_seq2seq_model_len20_bi.h5")
    # 训练模型
    inputs = [X, s0, c0, Yc]
    results = model.evaluate(x=inputs, y=outputs)
    print(np.shape(results))
    print(results)

for i in range(360):
    if i%6 == 0:
        source_text = np.load("/home/jack/201804RNN/king_train/train.npz")
    elif i%6 == 1:
        source_text = np.load("/home/jack/201804RNN/king_train/train2.npz")
    elif i%6 == 2:
        source_text = np.load("/home/jack/201804RNN/king_train/train3.npz")
    elif i%6 == 3:
        source_text = np.load("/home/jack/201804RNN/king_train/train4.npz")
    elif i%6 == 4:
        source_text = np.load("/home/jack/201804RNN/king_train/train5.npz")
    else:
        source_text = np.load("/home/jack/201804RNN/king_train/train6.npz")
    runepoch(source_text, i)
source_text = np.load("/home/jack/201804RNN/king_test/test.npz")
runeval(source_text)
source_text = np.load("/home/jack/201804RNN/king_test/test2.npz")
runeval(source_text)
source_text = np.load("/home/jack/201804RNN/king_test/test3.npz")
runeval(source_text)
source_text = np.load("/home/jack/201804RNN/king_test/test4.npz")
runeval(source_text)
source_text = np.load("/home/jack/201804RNN/king_test/test5.npz")
runeval(source_text)
source_text = np.load("/home/jack/201804RNN/king_test/test6.npz")
runeval(source_text)
# source_text = np.load("/home/jack/201804RNN/king_train/train.npz")
# run(source_text)
# source_text = np.load("/home/jack/201804RNN/king_train/train.npz")
# source_text = np.load("/home/jack/201804RNN/king_train/train2.npz")
# source_text = np.load("/home/jack/201804RNN/king_train/train3.npz")
# source_text = np.load("/home/jack/201804RNN/king_train/train4.npz")
# source_text = np.load("/home/jack/201804RNN/king_train/train5.npz")
# source_text = np.load("/home/jack/201804RNN/king_train/train6.npz")
# source_text = source_text['arr_0'] + source_text2['arr_0'] + source_text3['arr_0'] + source_text4['arr_0'] + source_text5['arr_0'] + source_text6['arr_0']
# source_text = list(source_text['arr_0'])+list(source_text2['arr_0'])+list(source_text3['arr_0'])+list(source_text4['arr_0'])+list(source_text5['arr_0'])+list(source_text6['arr_0'])
