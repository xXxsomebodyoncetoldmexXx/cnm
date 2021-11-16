import pysubs2
import os
from pydub import AudioSegment
from pprint import pprint


def subs_to_dict(path, noise_dur=1000):
  subs = pysubs2.load(path)
  result = dict()
  prev_key = None
  for i, e in enumerate(subs):
    key = e.text
    seg_len = e.end - e.start

    # if seg_len <= noise_dur and prev_key and prev_key == subs[i+1].text:
    #   # Too many overhead, will do in the future
    #   # key = prev_key
    #   pass

    try:
      result[key].append((e.start, e.end))
    except KeyError:
      result[key] = [(e.start, e.end)]
    prev_key = key
  return result


def split_actor(tb: dict, audio):
  idx = 0
  audio_list = list()
  for actor in tb.values():
    new_audio = AudioSegment.empty()
    for seg in actor:
      new_audio += audio[seg[0]:seg[1]]
    idx += 1
    # new_audio.export(os.path.join("lambda", f"{idx}.wav"), format="wav")
    audio_list.append(new_audio)
  return audio_list


def main():
  subs_path = os.path.join("lambda", "merged_2_1.srt")
  Audio = AudioSegment.from_wav(os.path.join("lambda", "merged_2_1.wav"))

  tb = subs_to_dict(subs_path)
  pprint(tb)
  split_actor(tb, Audio)


if __name__ == "__main__":
  main()
