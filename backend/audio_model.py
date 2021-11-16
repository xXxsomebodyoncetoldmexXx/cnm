import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import optimize

import librosa
import librosa.display

import srt
import datetime


class TDNN(nn.Module):
  def __init__(
      self,
      input_dim=23,
      output_dim=512,
      context_size=5,
      stride=1,
      dilation=1,
      batch_norm=False,
      dropout_p=0.2
  ):
    '''
    TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
    Affine transformation not applied globally to all frames but smaller windows with local context
    batch_norm: True to include batch normalisation after the non linearity

    Context size and dilation determine the frames selected
    (although context size is not really defined in the traditional sense)
    For example:
      context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
      context size 3 and dilation 2 is equivalent to [-2, 0, 2]
      context size 1 and dilation 1 is equivalent to [0]
    '''
    super(TDNN, self).__init__()
    self.context_size = context_size
    self.stride = stride
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dilation = dilation
    self.dropout_p = dropout_p
    self.batch_norm = batch_norm

    self.kernel = nn.Linear(input_dim*context_size, output_dim)
    self.nonlinearity = nn.ReLU()
    if self.batch_norm:
      self.bn = nn.BatchNorm1d(output_dim)
    if self.dropout_p:
      self.drop = nn.Dropout(p=self.dropout_p)

  def forward(self, x):
    '''
    input: size (batch, seq_len, input_features)
    outpu: size (batch, new_seq_len, output_features)
    '''

    _, _, d = x.shape
    assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
        self.input_dim, d)
    x = x.unsqueeze(1)

    # Unfold input into smaller temporal contexts
    x = F.unfold(
        x,
        (self.context_size, self.input_dim),
        stride=(1, self.input_dim),
        dilation=(self.dilation, 1)
    )

    # N, input_dim*context_size, new_t = x.shape
    x = x.transpose(1, 2)
    x = self.kernel(x.float())
    x = self.nonlinearity(x)

    if self.dropout_p:
      x = self.drop(x)

    if self.batch_norm:
      x = x.transpose(1, 2)
      x = self.bn(x)
      x = x.transpose(1, 2)

    return x


class X_vector(nn.Module):
  def __init__(self, input_dim=40, num_classes=46):
    super(X_vector, self).__init__()
    self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512,
                      context_size=5, dilation=1, dropout_p=0.5)
    self.tdnn2 = TDNN(input_dim=512, output_dim=512,
                      context_size=3, dilation=2, dropout_p=0.5)
    self.tdnn3 = TDNN(input_dim=512, output_dim=512,
                      context_size=3, dilation=3, dropout_p=0.5)
    self.tdnn4 = TDNN(input_dim=512, output_dim=512,
                      context_size=1, dilation=1, dropout_p=0.5)
    self.tdnn5 = TDNN(input_dim=512, output_dim=512,
                      context_size=1, dilation=1, dropout_p=0.5)
    self.segment6 = nn.Linear(1024, 512)
    self.segment7 = nn.Linear(512, 512)
    self.output = nn.Linear(512, num_classes)

  def forward(self, inputs):
    tdnn1_out = self.tdnn1(inputs)
    tdnn2_out = self.tdnn2(tdnn1_out)
    tdnn3_out = self.tdnn3(tdnn2_out)
    tdnn4_out = self.tdnn4(tdnn3_out)
    tdnn5_out = self.tdnn5(tdnn4_out)

    mean = torch.mean(tdnn5_out, 1)
    std = torch.var(tdnn5_out, 1)
    stat_pooling = torch.cat((mean, std), 1)

    segment6_out = self.segment6(stat_pooling)
    x_vec = self.segment7(segment6_out)

    predictions = self.output(x_vec)

    return predictions, x_vec


def get_list_inverse_index(unique_ids):
  result = dict()
  for i, unique_id in enumerate(unique_ids):
    result[unique_id] = i
  return result


def compute_sequence_match_accuracy(sequence1, sequence2):
  unique_ids1 = sorted(set(sequence1))
  unique_ids2 = sorted(set(sequence2))
  inverse_index1 = get_list_inverse_index(unique_ids1)
  inverse_index2 = get_list_inverse_index(unique_ids2)
  cnt_silent = 0

  count_matrix = np.zeros((len(unique_ids1), len(unique_ids2)))
  for item1, item2 in zip(sequence1, sequence2):
    index1 = inverse_index1[item1]
    index2 = inverse_index2[item2]
    if item1 == '':
      for i in range(len(unique_ids1)):
        count_matrix[i, index2] += 1
    else:
      count_matrix[index1, index2] += 1
#   print(count_matrix.astype(int))
  row_index, col_index = optimize.linear_sum_assignment(-count_matrix)
  optimal_match_count = count_matrix[row_index, col_index].sum()
  accuracy = (optimal_match_count+cnt_silent) / len(sequence1)
  return accuracy


def segment(y, lst_label, predict):
  lst_label.append(len(y))

  lst_pre_spk = [-1]*len(y)
  sid = 0
  eid = 0
  c = False
  maxval = 0
  for i in range(len(lst_pre_spk)):
    while eid < len(lst_label) and lst_label[eid] < min(i+(5*2)*160, len(lst_pre_spk)):
      eid += 1
      c = True

    while sid < len(lst_label) and lst_label[sid] <= max(i-(5*3)*160, 0):
      sid += 1
      c = True

    if sid < eid:
      if c:
        lst = list(predict[sid:eid])
        maxval = max(lst, key=lst.count)
        c = False
      lst_pre_spk[i] = maxval

  left = []
  last = -1
  cnt = 0
  for v in lst_pre_spk:
    if v == last or v == -1:
      cnt += 1
    else:
      last = v
      cnt = 1
    left.append([last, cnt])

  lst_pre_spk.reverse()
  right = []
  last = -1
  cnt = 0
  for v in lst_pre_spk:
    if v == last or v == -1:
      cnt += 1
    else:
      last = v
      cnt = 1
    right.append([last, cnt])
  lst_pre_spk.reverse()
  right.reverse()

  for i, v in enumerate(lst_pre_spk):
    if v == -1:
      if (right[i][0] == -1 or (left[i][1] < right[i][1] and left[i][0] != -1)):
        lst_pre_spk[i] = left[i][0]
      else:
        lst_pre_spk[i] = right[i][0]

  return lst_pre_spk


def file2srt(filename, model):
  with torch.no_grad():
    y, sr = librosa.load(filename, sr=16000)
    name = filename.split('/')[-1].split('.')[0]

    lst_raw_emb = list()
    lst_start = list()

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=512, hop_length=160, n_mfcc=40).T
    start_point = 0
    for idx in range(0, len(mfcc)-30, 5):
      lst_raw_emb.append(mfcc[idx:idx+30])
      lst_start.append(start_point)
      start_point += 160*5

    lst_emb_x = list()
    for mfcc in lst_raw_emb:
      feat = torch.from_numpy(np.expand_dims(mfcc, axis=0))
      _, xvector = model(feat)
      lst_emb_x.append(xvector)
    lst_emb_x = np.vstack(lst_emb_x)

    lst_Kmeans = [KMeans(n_clusters=i).fit(
        lst_emb_x).labels_ for i in range(2, 8)]
    predict_K1 = np.argmax([silhouette_score(lst_emb_x, labels)
                           for labels in lst_Kmeans])
    predict = lst_Kmeans[predict_K1]
    lst_pre_spk = segment(y, lst_start, predict)

    start = 0
    end = -1
    last = -1
    sr = 16000
    lst_sub_predict = []
    for idx, lbl in enumerate(lst_pre_spk):
      if lbl != last:
        if last != -1:
          lst_sub_predict.append(
              {'start': start/sr, 'end': end/sr, 'predict': last+1})
        last = lbl
        start = idx
      else:
        end = idx
    lst_sub_predict.append(
        {'start': start/sr, 'end': end/sr, 'predict': last+1})

    subs = list()
    idx = 0
    for lines in lst_sub_predict:
      idx += 1
      subs.append(srt.Subtitle(index=idx,
                               start=datetime.timedelta(
                                   seconds=lines['start']),
                               end=datetime.timedelta(seconds=lines['end']),
                               content='SPEAKER {}'.format(lines['predict'])))

  with open(f"{name}.srt", "w") as f:
    f.write(srt.compose(subs))


def load_model(path):
  model = X_vector(40, 46)
  if torch.cuda.is_available():
    model.load_state_dict(torch.load(path)['model'])
  else:
    model.load_state_dict(torch.load(
        path, map_location=torch.device('cpu'))['model'])
  model.eval()
  return model


def main():
  model_path = "pretrain.pt"
  audio_path = "merged_2_0.wav"

  model = load_model(model_path)

  file2srt(audio_path, model)


if __name__ == "__main__":
  main()
