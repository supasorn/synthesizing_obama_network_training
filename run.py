import sys

from util import *

import json
import copy
import random
import platform
import bisect
import numpy as np


class Speech(TFBase):
  def __init__(self):
    super(Speech, self).__init__()
    self.parser.add_argument('--timedelay', type=int, default=20,
                       help='time delay between output and input')
    self.parser.add_argument('--rnn_size', type=int, default=60,
                     help='size of RNN hidden state')
    self.parser.add_argument('--num_layers', type=int, default=1,
                     help='number of layers in the RNN')
    self.parser.add_argument('--batch_size', type=int, default=100,
                     help='minibatch size')
    self.parser.add_argument('--seq_length', type=int, default=100,
                     help='RNN sequence length')
    self.parser.add_argument('--input', type=str, default='',
                       help='input for generation')
    self.parser.add_argument('--input2', type=str, default='',
                       help='input for any mfcc wav file')
    self.parser.add_argument('--guy', type=str, default='Obama2',
                       help='dataset')
    self.parser.add_argument('--normalizeoutput', action='store_true')

    self.args = self.parser.parse_args()
    if self.args.save_dir == "":
      raise ValueError('Missing save_dir')

    # self.training_dir = base + "/face-singleview/data/" + self.args.guy + "/"
    self.training_dir = "obama_data/"

    self.fps = 29.97
    self.loadData()
    self.model = self.standardL2Model

    self.audioinput = len(self.args.input2)
    if (self.audioinput):
      self.args.input = self.args.input2

    if len(self.args.input):
      self.test()
    else:
      self.train()

  def createInputFeature(self, audio, audiodiff, timestamps, startframe, nframe):
    startAudio = bisect.bisect_left(timestamps, (startframe - 1) / self.fps)
    endAudio = bisect.bisect_right(timestamps, (startframe + nframe - 2) / self.fps)

    inp = np.concatenate((audio[startAudio:endAudio, :-1], audiodiff[startAudio:endAudio, :]), axis=1)
    return startAudio, endAudio, inp 


  def preprocess(self, save_dir):
    files = [x.split("\t")[0].strip() for x in open(self.training_dir + "processed_fps.txt", "r").readlines()]

    inps = {"training": [], "validation": []}
    outps = {"training": [], "validation": []}

    # validation = 0.2
    validation = 0
    for i in range(len(files)):
      tp = "training" if random.random() > validation else "validation"

      dnums = sorted([os.path.basename(x) for x in glob.glob(self.training_dir + files[i] + "}}*")])

      audio = np.load(self.training_dir + "/audio/normalized-cep13/" + files[i] + ".wav.npy") 
      audiodiff = audio[1:,:-1] - audio[:-1, :-1]

      print files[i], audio.shape, tp
      timestamps = audio[:, -1]

      for dnum in dnums:
        print dnum 
        fids = readCVFloatMat(self.training_dir + dnum + "/frontalfidsCoeff_unrefined.bin")
        if not os.path.exists(self.training_dir + dnum + "/startframe.txt"):
          startframe = 1
        else:
          startframe = readSingleInt(self.training_dir + dnum + "/startframe.txt")
        nframe = readSingleInt(self.training_dir + dnum + "/nframe.txt")

        startAudio, endAudio, inp = self.createInputFeature(audio, audiodiff, timestamps, startframe, nframe)

        outp = np.zeros((endAudio - startAudio, fids.shape[1]), dtype=np.float32)
        leftmark = 0
        for aud in range(startAudio, endAudio):
          audiotime = audio[aud, -1]
          while audiotime >= (startframe - 1 + leftmark + 1) / self.fps:
            leftmark += 1
          t = (audiotime - (startframe - 1 + leftmark) / self.fps) * self.fps;
          outp[aud - startAudio, :] = fids[leftmark, :] * (1 - t) + fids[min(len(fids) - 1, leftmark + 1), :] * t;
            
        inps[tp].append(inp)
        outps[tp].append(outp)

    return (inps, outps)

  def standardL2Model(self, args, infer=False):
    if infer:
      args.batch_size = 1
      args.seq_length = 1

    cell_fn = tf.nn.rnn_cell.LSTMCell
    cell = cell_fn(args.rnn_size, state_is_tuple=True)

    if infer == False and args.keep_prob < 1: # training mode
      cell0 = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = args.keep_prob)
      cell1 = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = args.keep_prob, output_keep_prob = args.keep_prob)
      self.network = tf.nn.rnn_cell.MultiRNNCell([cell0] * (args.num_layers -1) + [cell1], state_is_tuple=True)
    else:
      self.network = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)


    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dimin])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dimout])
    self.initial_state = self.network.zero_state(batch_size=args.batch_size, dtype=tf.float32)

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [args.rnn_size, self.dimout])
      output_b = tf.get_variable("output_b", [self.dimout])

    inputs = tf.split(1, args.seq_length, self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, states = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, self.network, loop_function=None, scope='rnnlm')

    output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = states
    self.output = output

    flat_target_data = tf.reshape(self.target_data,[-1, self.dimout])
        
    lossfunc = tf.reduce_sum(tf.squared_difference(flat_target_data, output))
    #lossfunc = tf.reduce_sum(tf.abs(flat_target_data - output))
    self.cost = lossfunc / (args.batch_size * args.seq_length * self.dimout)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def load_preprocessed(self, inps, outps):
    newinps = {"training": [], "validation": []}
    newoutps = {"training": [], "validation": []}
    for key in newinps:
      for i in range(len(inps[key])):
        if len(inps[key][i]) - self.args.timedelay >= (self.args.seq_length+2):
          if self.args.timedelay > 0:
            newinps[key].append(inps[key][i][self.args.timedelay:])
            newoutps[key].append(outps[key][i][:-self.args.timedelay])
          else:
            newinps[key].append(inps[key][i])
            newoutps[key].append(outps[key][i])
    print "load preprocessed", len(newinps), len(newoutps)
    return newinps, newoutps


  def sample(self, sess, args, data, pt):
    if self.audioinput:
      self.sample_audioinput(sess, args, data, pt)
    else:
      self.sample_videoinput(sess, args, data, pt)

  def sample_audioinput(self, sess, args, data, pt):
    meani, stdi, meano, stdo = data["inputmean"], data["inputstd"], data["outputmean"], data["outputstd"]
    audio = np.load(self.training_dir + "/audio/normalized-cep13/" + self.args.input2 + ".wav.npy") 

    audiodiff = audio[1:,:-1] - audio[:-1, :-1]
    timestamps = audio[:, -1]

    times = audio[:, -1]
    inp = np.concatenate((audio[:-1, :-1], audiodiff[:, :]), axis=1)

    state = []
    for c, m in self.initial_state: # initial_state: ((c1, m1), (c2, m2))
      state.append((c.eval(), m.eval()))

    if not os.path.exists("results/"):
      os.mkdir("results/")

    f = open("results/" + self.args.input2 + "_" + args.save_dir + ".txt", "w")
    print "output to results/" + self.args.input2 + "_" + args.save_dir + ".txt"
    f.write("%d %d\n" % (len(inp), self.dimout + 1))
    fetches = []
    fetches.append(self.output)
    for c, m in self.final_state: # final_state: ((c1, m1), (c2, m2))
      fetches.append(c)
      fetches.append(m)

    feed_dict = {}
    for i in range(len(inp)):
      for j, (c, m) in enumerate(self.initial_state):
        feed_dict[c], feed_dict[m] = state[j]

      input = (inp[i] - meani) / stdi
      feed_dict[self.input_data] = [[input]]
      res = sess.run(fetches, feed_dict)
      output = res[0] * stdo + meano

      if i >= args.timedelay:
        shifttime = times[i - args.timedelay]
      else:
        shifttime = times[0]
      f.write(("%f " % shifttime) + " ".join(["%f" % x for x in output[0]]) + "\n")

      state_flat = res[1:]
      state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)] 
    f.close()


def main():
  s = Speech()

if __name__ == '__main__':
  main()


