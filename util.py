import numpy as np
import sys
import tensorflow as tf

import math
import struct
import argparse
import time
import os
import cPickle
import random
import platform
import glob

plat = platform.dist()[0]
if plat == "Ubuntu":
  base = "/home/supasorn/"
else:
  base = "/projects/grail/supasorn2nb/"

def readSingleInt(path):
  with open(path) as f:
    return int(f.readline())

def readCVFloatMat(fl):
  f = open(fl)
  t = struct.unpack('B', f.read(1))[0]
  if t != 5:
    return 0
  h = struct.unpack('i', f.read(4))[0]
  w = struct.unpack('i', f.read(4))[0]
  return np.reshape(np.array(struct.unpack('%df' % (h * w), f.read(4 * h * w)), float), (h, w))

def _str_to_bool(s):
  if s.lower() not in ['true', 'false']:
      raise ValueError('Need bool; got %r' % s)
  return s.lower() == 'true'

def add_boolean_argument(parser, name, default=False):
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
  group.add_argument('--no' + name, dest=name, action='store_false')

def normalizeData(lst, savedir, name, varnames, normalize=True):
  allstrokes = np.concatenate(lst)
  mean = np.mean(allstrokes, 0)
  std = np.std(allstrokes, 0) 

  f = open(savedir + "/" + name + ".txt", "w")
  minv = np.min(allstrokes, 0)
  maxv = np.max(allstrokes, 0)

  if not isinstance(normalize, list):
    normalize = [normalize] * len(mean)

  for i, n in enumerate(varnames):
    if normalize[i]:
      f.write(n + "\n  mean: %f\n  std :%f\n  min :%f\n  max :%f\n\n" % (mean[i], std[i], minv[i], maxv[i]))
    else:
      f.write(n + "\n  mean: %f (-> 0)\n  std :%f (-> 1)\n  min :%f\n  max :%f\n\n" % (mean[i], std[i], minv[i], maxv[i]))
      mean[i] = 0
      std[i] = 1

  np.save(savedir + '/' + name + '.npy', {'min': minv, 'max': maxv, 'mean': mean, 'std': std})
  for i in range(len(lst)):
    lst[i] = (lst[i] - mean) / std

  f.close()
  return mean, std


class TFBase(object):
  def __init__(self):
    np.random.seed(42)
    random.seed(42)
    self.parser = argparse.ArgumentParser()
    self.addDefaultParameters()

  def addDefaultParameters(self):
    self.parser.add_argument('--num_epochs', type=int, default=300,
                       help='number of epochs')
    self.parser.add_argument('--save_every', type=int, default=10,
                       help='save frequency')
    self.parser.add_argument('--grad_clip', type=float, default=10.,
                       help='clip gradients at this value')
    self.parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='learning rate')
    self.parser.add_argument('--decay_rate', type=float, default=1,
                       help='decay rate for rmsprop')
    self.parser.add_argument('--keep_prob', type=float, default=1,
                       help='dropout keep probability')

    self.parser.add_argument('--save_dir', type=str, default='',
                       help='save directory')
    self.parser.add_argument('--usetrainingof', type=str, default='',
                       help='trainingset')

    add_boolean_argument(self.parser, "reprocess")
    add_boolean_argument(self.parser, "normalizeinput", default=True)

  def normalize(self, inps, outps):
    meani, stdi = normalizeData(inps["training"], "save/" + self.args.save_dir, "statinput", ["fea%02d" % x for x in range(inps["training"][0].shape[1])], normalize=self.args.normalizeinput)
    meano, stdo = normalizeData(outps["training"], "save/" + self.args.save_dir, "statoutput", ["fea%02d" % x for x in range(outps["training"][0].shape[1])], normalize=self.args.normalizeoutput)

    for i in range(len(inps["validation"])):
      inps["validation"][i] = (inps["validation"][i] - meani) / stdi;

    for i in range(len(outps["validation"])):
      outps["validation"][i] = (outps["validation"][i] - meano) / stdo;

    return meani, stdi, meano, stdo

  def loadData(self):
    if not os.path.exists("save/"):
      os.mkdir("save/")
    if not os.path.exists("save/" + self.args.save_dir):
      os.mkdir("save/" + self.args.save_dir)

    if len(self.args.usetrainingof):
      data_file = "data/training_" + self.args.usetrainingof + ".cpkl"
    else:
      data_file = "data/training_" + self.args.save_dir + ".cpkl"

    if not (os.path.exists(data_file)) or self.args.reprocess:
      print "creating training data cpkl file from raw source"
      inps, outps = self.preprocess(data_file)

      meani, stdi, meano, stdo = self.normalize(inps, outps)

      if not os.path.exists(os.path.dirname(data_file)):
        os.mkdir(os.path.dirname(data_file))
      f = open(data_file, "wb")
      cPickle.dump({"input": inps["training"], "inputmean": meani, "inputstd": stdi, "output": outps["training"], "outputmean":meano, "outputstd": stdo, "vinput": inps["validation"], "voutput": outps["validation"]}, f, protocol=2) 
      f.close() 


    f = open(data_file,"rb")
    data = cPickle.load(f)
    inps = {"training": data["input"], "validation": data["vinput"]} 
    outps = {"training": data["output"], "validation": data["voutput"]} 
    f.close()

    self.dimin = inps["training"][0].shape[1]
    self.dimout = outps["training"][0].shape[1]

    self.inps, self.outps = self.load_preprocessed(inps, outps)
    self.num_batches = {}
    self.pointer = {}
    for key in self.inps:
      self.num_batches[key] = 0
      for inp in self.inps[key]:
        self.num_batches[key] += int(math.ceil((len(inp) - 2) / self.args.seq_length))
      self.num_batches[key] = int(self.num_batches[key] / self.args.batch_size)
      self.reset_batch_pointer(key)

  def preprocess(self):
    raise NotImplementedError()

  def next_batch(self, key="training"):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in xrange(self.args.batch_size):
      inp = self.inps[key][self.pointer[key]]
      outp = self.outps[key][self.pointer[key]]

      n_batch = int(math.ceil((len(inp) - 2) / self.args.seq_length)) 

      idx = random.randint(1, len(inp) - self.args.seq_length - 1)
      x_batch.append(np.copy(inp[idx:idx+self.args.seq_length]))
      y_batch.append(np.copy(outp[idx:idx+self.args.seq_length]))

      if random.random() < 1.0 / float(n_batch): 
        self.tick_batch_pointer(key)
    return x_batch, y_batch

  def tick_batch_pointer(self, key):
    self.pointer[key] += 1
    if self.pointer[key] >= len(self.inps[key]):
      self.pointer[key] = 0

  def reset_batch_pointer(self, key):
    self.pointer[key] = 0


  def test(self):
    # only use save_dir from args
    save_dir = self.args.save_dir

    with open(os.path.join("save/" + save_dir, 'config.pkl')) as f:
      saved_args = cPickle.load(f)

    if len(saved_args.usetrainingof):
      pt = saved_args.usetrainingof
    else:
      pt = save_dir

    with open("./data/training_" + pt + ".cpkl", "rb") as f:
      raw = cPickle.load(f)

    model = self.model(saved_args, True)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state("save/" + save_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print "loading model: ", ckpt.model_checkpoint_path

    saved_args.input = self.args.input

    self.sample(sess, saved_args, raw, pt)

  def train(self):
    with open(os.path.join("save/" + self.args.save_dir, 'config.pkl'), 'w') as f:
      cPickle.dump(self.args, f)

    with tf.Session() as sess:
      model = self.model(self.args)

      tf.initialize_all_variables().run()
      ts = TrainingStatus(sess, self.args.num_epochs, self.num_batches["training"], save_interval = self.args.save_every, graph = sess.graph, save_dir = "save/" + self.args.save_dir)

      print "training batches: ", self.num_batches["training"]
      for e in xrange(ts.startEpoch, self.args.num_epochs):
        sess.run(tf.assign(self.lr, self.args.learning_rate * (self.args.decay_rate ** e)))
        self.reset_batch_pointer("training")
        self.reset_batch_pointer("validation")


        state = []
        for c, m in self.initial_state: 
          state.append((c.eval(), m.eval()))

        fetches = []
        fetches.append(self.cost)
        fetches.append(self.train_op)

        feed_dict = {}
        for i, (c, m) in enumerate(self.initial_state):
          feed_dict[c], feed_dict[m] = state[i]

        for b in xrange(self.num_batches["training"]):
          ts.tic()
          x, y = self.next_batch()

          feed_dict[self.input_data] = x
          feed_dict[self.target_data] = y

          res = sess.run(fetches, feed_dict)
          train_loss = res[0]

          print ts.tocBatch(e, b, train_loss)

        validLoss = 0
        if self.num_batches["validation"] > 0:
          fetches = []
          fetches.append(self.cost)
          for b in xrange(self.num_batches["validation"]):
            x, y = self.next_batch("validation")

            feed_dict[self.input_data] = x
            feed_dict[self.target_data] = y

            loss = sess.run(fetches, feed_dict)
            validLoss += loss[0]
          validLoss /= self.num_batches["validation"]
          
        ts.tocEpoch(sess, e, validLoss)


class TrainingStatus:
  def __init__(self, sess, num_epochs, num_batches, logwrite_interval = 25, eta_interval = 25, save_interval = 100, save_dir = "save", graph = None):
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    #if graph is not None:
      #self.writer = tf.train.SummaryWriter(save_dir, graph)
    #else:
      #self.writer = tf.train.SummaryWriter(save_dir)

    self.save_dir = save_dir
    self.model_dir = os.path.join(save_dir, 'model.ckpt')
    #self.saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)
    self.saver = tf.train.Saver(tf.all_variables())

    lastCheckpoint = tf.train.latest_checkpoint(save_dir) 
    if lastCheckpoint is None:
      self.startEpoch = 0
    else:
      print "Last checkpoint :", lastCheckpoint
      self.startEpoch = int(lastCheckpoint.split("-")[-1])
      self.saver.restore(sess, lastCheckpoint)

    print "startEpoch = ", self.startEpoch

    self.logwrite_interval = logwrite_interval
    self.eta_interval = eta_interval
    self.totalTask = num_epochs * num_batches
    self.num_epochs = num_epochs
    self.num_batches = num_batches
    self.save_interval = save_interval

    self.etaCount = 0
    self.etaStart = time.time()
    self.duration = 0

    self.avgloss = 0
    self.avgcount = 0

  def tic(self):
      self.start = time.time()

  def tocBatch(self, e, b, loss):
      self.end = time.time()
      taskNum = (e * self.num_batches + b)

      self.etaCount += 1
      if self.etaCount % self.eta_interval == 0:
        self.duration = time.time() - self.etaStart
        self.etaStart = time.time()

      etaTime = float(self.totalTask - (taskNum + 1)) / self.eta_interval * self.duration
      m, s = divmod(etaTime, 60)
      h, m = divmod(m, 60)
      etaString = "%d:%02d:%02d" % (h, m, s)
      self.avgloss += loss
      self.avgcount += 1

      if taskNum == 0:
        with open(self.save_dir + "/avgloss.txt", "w") as f:
          f.write("0 %f %f\n" % (loss, loss))

      return "%.2f%% (%d/%d): %.3f  t %.3f  @ %s (%s)" % (taskNum * 100.0 / self.totalTask, e, self.num_epochs, loss, self.end - self.start, time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + etaTime)), etaString)

  def tocEpoch(self, sess, e, validLoss=0):
    if (e + 1) % self.save_interval == 0 or e == self.num_epochs - 1:
      self.saver.save(sess, self.model_dir, global_step = e + 1)
      print "model saved to {}".format(self.model_dir)

    
    lines = open(self.save_dir + "/avgloss.txt", "r").readlines()
    with open(self.save_dir + "/avgloss.txt", "w") as f:
      for line in lines:
        if int(line.split(" ")[0]) >= e + 1:
          break
        f.write(line)
      f.write("%d %f %f\n" % (e+1, self.avgloss / self.avgcount, validLoss))

    self.avgcount = 0
    self.avgloss = 0;


