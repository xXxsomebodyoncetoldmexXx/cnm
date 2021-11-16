from pydub import AudioSegment
from flask import Flask, request
from flask_cors import CORS
from db_connect import DB
from pprint import pprint
from audio_process import subs_to_dict, split_actor
from audio_model import load_model, file2srt
import base64
import io
import os
import uuid

app = Flask(__name__, static_folder="..\\frontend\\build",
            static_url_path="/")
# db = DB()
PATH_TO_MODEL = "pretrain.pt"
model = load_model(PATH_TO_MODEL)

CORS(app, origins=["*"])
app.config['SECRET_KEY'] = '47d3eec7d3d30885d432c2b95c31f12fc9cf4742'

ALLOWED_EXTENSIONS = set(['.wav'])


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def b64_to_audio(b64_string):
  binary = base64.b64decode(b64_string)
  return AudioSegment.from_file(io.BytesIO(binary))


def audio_to_b64(audio_seg):
  b64_string = None
  audio_seg.export("out.wav")
  with open("out.wav", "rb") as f:
    b64_string = base64.b64encode(f.read()).decode()
  return 'data:audio/wav;base64,' + b64_string


def gen_tmp_name():
  return uuid.uuid4()


@app.route("/")
def index():
  # return "<h1>Hello World!</h1>"
  return app.send_static_file("index.html")


@app.route('/api/upload', methods=['POST'])
def upload_file():
  audio_content = request.json['audio_data']
  # print(audio_content[:100])
  audio = b64_to_audio(audio_content.split(",", 1)[1])
  name = gen_tmp_name()
  audio.export(f"{name}.wav")
  file2srt(f"{name}.wav", model)
  trans_script = subs_to_dict(f"{name}.srt")
  aud_list = split_actor(trans_script, audio)
  os.remove(f"{name}.wav")
  os.remove(f"{name}.srt")
  result = {f"Speaker{i+1}": audio_to_b64(aud)
            for i, aud in enumerate(aud_list)}
  # result = {
  #     "Speaker 1": audio_to_b64(audio[:half]),
  #     "Speaker 2": audio_to_b64(audio[half:]),
  # }
  return {"speakers": result}


# @app.route("/api/download/", methods=["GET"])
# @app.route("/api/download/<int:id>", methods=["GET"])
# def download(id=None):
#   content = None
#   if id:
#     content = db.get_audio(id)
#   else:
#     content = db.get_audios()

#   if not content:
#     return "File not found", 404
#   return {"status": 200, "content": content}


def main():
  app.run("localhost", 3001, debug=True)


if __name__ == "__main__":
  main()
