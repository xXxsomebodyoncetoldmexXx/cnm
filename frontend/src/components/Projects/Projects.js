// import uniqid from 'uniqid'
import axios from 'axios'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faDownload, faPlay } from '@fortawesome/free-solid-svg-icons'
import { useState } from 'react'
import { projects } from '../../portfolio'
// import ProjectContainer from '../ProjectContainer/ProjectContainer'
import './Projects.css'

const Projects = () => {
  // if (!projects.length) return null
  const [speaker, setSpeaker] = useState([])
  const apiUrl =
    'https://uu9amy5di7.execute-api.ap-southeast-1.amazonaws.com/dev/audioapi'
  const speakerAudio = []
  const uploadHandle = async () => {
    const file = document.getElementById('upload_input').files[0]
    document.getElementById('upload_input').value = ''
    if (file !== undefined) {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => {
        // Save audio remotely
        // axios.post(apiUrl, { audio_name: file.name, audio_data: reader.result })
        //
        // Send for server to process
        axios
          .post('/api/upload', {
            audio_data: reader.result,
          })
          .then((res) => {
            Object.values(res.data.speakers).forEach((value, idx) => {
              speakerAudio.push({
                id: idx,
                data: value,
              })
            })
            setSpeaker(speakerAudio)
          })
          .catch(console.log)
      }
    }
  }

  const downloadHandle = (id) => {
    const downloadPlaceholder = document.getElementById('download')
    downloadPlaceholder.href = speaker[id].data
    downloadPlaceholder.download = `Speaker${id + 1}.wav`
    downloadPlaceholder.click()
  }

  let audio

  const listenHandle = (id) => {
    if (audio) {
      audio.pause()
    }
    audio = new Audio(speaker[id].data)
    audio.currentTime = 0
    audio.play()
  }

  return (
    <section
      id='projects'
      className='section projects'
      style={{ 'text-align': 'center' }}
    >
      <a href='/' id='download' style={{ display: 'none' }}>
        download placeholder
      </a>
      <div className='formbg-outer'>
        <div className='formbg'>
          <div className='formbg-inner padding-horizontal--48'>
            <span className='padding-bottom--15'>Upload wav file</span>
            <div className='padding-bottom--15'>
              <input type='file' id='upload_input' />
              <button
                type='button'
                className='btn btn--plain more'
                onClick={uploadHandle}
              >
                Upload file
              </button>
            </div>
            <div>
              <ul>
                {speaker.map((value) => (
                  <li key={value.id} className='speaker__list-item'>
                    Speaker {value.id + 1}:
                    <FontAwesomeIcon
                      className='spacing'
                      icon={faPlay}
                      onClick={() => listenHandle(value.id)}
                    />
                    <FontAwesomeIcon
                      className='spacing'
                      icon={faDownload}
                      onClick={() => downloadHandle(value.id)}
                    />
                  </li>
                ))}
              </ul>
              <button
                type='button'
                className='center-grid btn btn--plain'
                style={
                  speaker.length ? { display: 'block' } : { display: 'none' }
                }
                onClick={() => setSpeaker([])}
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Projects
