import rds_config
import pymysql
import base64
from hashlib import sha1


class DB():
  def __init__(self) -> None:
    self.host = rds_config.db_host
    self.user = rds_config.db_username
    self.password = rds_config.db_password
    self.db_name = rds_config.db_name

    self.conn = pymysql.connect(host=self.host, user=self.user,
                                passwd=self.password, db=self.db_name, connect_timeout=5)

    with self.conn.cursor() as cursor:
      sql = "CREATE TABLE IF NOT EXISTS audio_file (AudioID INT NOT NULL AUTO_INCREMENT,\
                                                  AudioName NVARCHAR(255) NOT NULL,\
                                                  AudioHash CHAR(40) NOT NULL,\
                                                  AudioData MEDIUMTEXT NOT NULL,\
                                                  CONSTRAINT PK_Audio PRIMARY KEY (AudioID))"
      cursor.execute(sql)
      self.conn.commit()

  def get_audio(self, audio_id):
    sql = "SELECT AudioName, AudioData FROM audio_file WHERE AudioID=%s"

    with self.conn.cursor() as cursor:
      cursor.execute(sql, audio_id)
      result = cursor.fetchone()
      self.conn.commit()

    return result

  def get_audios(self):
    sql = "SELECT AudioID, AudioName FROM audio_file"

    with self.conn.cursor() as cursor:
      cursor.execute(sql)
      result = cursor.fetchall()
      self.conn.commit()

    return result

  def save_audio(self, audio_name, audio_binary):
    sql = "INSERT INTO audio_file(AudioName, AudioHash, AudioData) value (%s, %s, %s)"
    sql2 = "SELECT AudioID from audio_file where AudioName=%s"

    data = base64.b64encode(audio_binary).decode()
    h = sha1(audio_binary).hexdigest()

    with self.conn.cursor() as cursor:
      cursor.execute(sql, (audio_name, h, data))
      self.conn.commit()

      cursor.execute(sql2, audio_name)
      result = cursor.fetchone()
      self.conn.commit()

    return result


def main():
  DB()


if __name__ == "__main__":
  main()
