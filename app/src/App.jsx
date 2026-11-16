import { useEffect, useRef, useState } from 'react'
import './App.css'

// Must match the API's preprocessing limits (JPEG/PNG, smaller than 20 MB).
const ACCEPTED_TYPES = ['image/jpeg', 'image/png']
const MAX_BYTES = 20 * 1024 * 1024
// Detection runs at 1600 px, so photos are shrunk to 2048 px before upload:
// a 12 MP phone photo goes from ~2.5 MB to ~0.5 MB with no loss in detection.
const UPLOAD_MAX_SIDE = 2048
const UPLOAD_QUALITY = 0.9
const REQUEST_TIMEOUT_MS = 120000

// "preprocess;dur=812, detect;dur=4210, total;dur=5100" -> 5100
function serverTotalMs(header) {
  const match = /total;dur=(\d+)/.exec(header || '')
  return match ? Number(match[1]) : null
}

function errorMessage(body, status) {
  return body?.error?.message || `The server returned an error (${status}).`
}

async function callApi(path, options) {
  const response = await fetch(`/api${path}`, options)
  if (response.status === 204) return null
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(errorMessage(body, response.status))
  return { ...body, serverMs: serverTotalMs(response.headers.get('Server-Timing')) }
}

// XMLHttpRequest instead of fetch, because fetch cannot report upload progress.
function uploadImage(file, onProgress) {
  return new Promise((resolve, reject) => {
    const body = new FormData()
    body.append('file', file, file.name || 'photo.jpg')
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/images')
    xhr.responseType = 'json'
    xhr.timeout = REQUEST_TIMEOUT_MS
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total)
    }
    xhr.upload.onload = () => onProgress(1)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve({ ...xhr.response, serverMs: serverTotalMs(xhr.getResponseHeader('Server-Timing')) })
      } else {
        reject(new Error(errorMessage(xhr.response, xhr.status)))
      }
    }
    xhr.onerror = () => reject(new Error('Network error. Check your connection and try again.'))
    xhr.ontimeout = () => reject(new Error('The request took too long. Try again.'))
    xhr.send(body)
  })
}

// Shrink large photos in the browser before upload. Drawing an <img> applies the
// EXIF rotation, so the uploaded JPEG is already upright. Falls back to the original file.
async function shrinkForUpload(file) {
  const url = URL.createObjectURL(file)
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight))
    if (scale === 1 && file.size < 1.5 * 1024 * 1024) return file
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(img.naturalWidth * scale)
    canvas.height = Math.round(img.naturalHeight * scale)
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', UPLOAD_QUALITY))
    if (!blob || blob.size >= file.size) return file
    const name = (file.name || 'photo').replace(/\.[^.]+$/, '') + '.jpg'
    return new File([blob], name, { type: 'image/jpeg' })
  } catch {
    return file
  } finally {
    URL.revokeObjectURL(url)
  }
}

const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`

function App() {
  const uploadInputRef = useRef(null)
  const captureInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [message, setMessage] = useState('')
  const [isError, setIsError] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  // idle -> working -> review (warnings: retake or continue) | done
  const [phase, setPhase] = useState('idle')
  const [imageId, setImageId] = useState('')
  const [warnings, setWarnings] = useState([])
  const [result, setResult] = useState(null)
  // Which borders the result image shows: 'words' (blue, per word) or 'lines' (green, per line).
  const [view, setView] = useState('words')
  // TrOCR text per line, filled by "Read text" (POST /read).
  const [recognition, setRecognition] = useState(null)
  const [isReading, setIsReading] = useState(false)
  const [totalMs, setTotalMs] = useState(null)
  const startedAtRef = useRef(0)

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    },
    [previewUrl],
  )

  const showMessage = (text, error = false) => {
    setMessage(text)
    setIsError(error)
  }

  const resetResult = () => {
    setPhase('idle')
    setImageId('')
    setWarnings([])
    setResult(null)
    setTotalMs(null)
    setView('words')
    setRecognition(null)
  }

  const selectFile = (nextFile) => {
    showMessage('')
    if (!nextFile) return

    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      showMessage('Choose a JPG or PNG image.', true)
      return
    }
    if (nextFile.size >= MAX_BYTES) {
      showMessage('The image must be smaller than 20 MB.', true)
      return
    }

    resetResult()
    setFile(nextFile)
    setPreviewUrl(URL.createObjectURL(nextFile))
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  const applyResponse = (body) => {
    setImageId(body.image_id)
    setWarnings(body.warnings || [])
    if (body.status === 'needs_review') {
      setPhase('review')
      showMessage('This image may be hard to read. Retake it or continue anyway.', true)
      return
    }
    setResult(body)
    setRecognition(body.recognition || null)
    setPhase('done')
    // The success message is shown once the annotated image has loaded (see onLoad below).
    showMessage('Loading the result image…')
  }

  const resultLoaded = () => {
    if (totalMs != null) return // switching Lines/Words reloads the image; keep the first timing
    setTotalMs(performance.now() - startedAtRef.current)
    const lines = `${result.count} handwritten ${result.count === 1 ? 'line' : 'lines'}`
    const words = result.word_count ? ` and ${result.word_count} ${result.word_count === 1 ? 'word' : 'words'}` : ''
    showMessage(result.count === 0 ? 'No handwriting was found in this image.' : `Found ${lines}${words}.`)
  }

  const resultImageUrl = result && view === 'words' && result.words_annotated_url
    ? result.words_annotated_url
    : result?.annotated_url

  const runRequest = async (request, phaseOnError, startMessage) => {
    setPhase('working')
    startedAtRef.current = performance.now()
    showMessage(startMessage)
    try {
      applyResponse(await request())
    } catch (error) {
      setPhase(phaseOnError)
      showMessage(error.message, true)
    }
  }

  const detect = () => {
    resetResult()
    runRequest(
      async () => {
        const upload = await shrinkForUpload(file)
        return uploadImage(upload, (done) =>
          showMessage(done < 1 ? `Uploading… ${Math.round(done * 100)}%` : 'Detecting handwriting…'),
        )
      },
      'idle',
      'Preparing the photo…',
    )
  }

  const continueAnyway = () =>
    runRequest(() => callApi(`/images/${imageId}/confirm`, { method: 'POST' }), 'review', 'Detecting handwriting…')

  const retake = async () => {
    try {
      await callApi(`/images/${imageId}`, { method: 'DELETE' })
    } catch {
      // Already gone or expired: nothing to clean up.
    }
    resetResult()
    setFile(null)
    setPreviewUrl('')
    showMessage('Upload or capture a new image.')
  }

  const readText = async () => {
    setIsReading(true)
    showMessage(`Reading ${result.count} ${result.count === 1 ? 'line' : 'lines'}… about 1 second per line.`)
    try {
      const body = await callApi(`/images/${imageId}/read`, { method: 'POST' })
      setRecognition(body)
      const took = body.serverMs != null ? ` in ${seconds(body.serverMs)}` : ''
      showMessage(`Read ${body.lines.length} ${body.lines.length === 1 ? 'line' : 'lines'}${took}.`)
    } catch (error) {
      showMessage(error.message, true)
    } finally {
      setIsReading(false)
    }
  }

  const isWorking = phase === 'working' || isReading

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="Handwriting home">
          Pro Detection
        </a>
        <p className="topbar-note">Text detection workspace</p>
      </header>

      <section className="workspace" aria-labelledby="workspace-title">
        <div className="intro">
          <h1 id="workspace-title">Detect handwritten text.</h1>
          <p>Upload a clear page to prepare it for line and word detection.</p>
        </div>

        <div className="work-area">
          <div
            className={`document-stage${isDragging ? ' is-dragging' : ''}`}
            onDragEnter={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            aria-busy={isWorking}
          >
            {result ? (
              <img
                key={resultImageUrl}
                src={resultImageUrl}
                alt={`Handwritten document with each detected ${view === 'words' ? 'word' : 'line'} outlined`}
                onLoad={resultLoaded}
                onError={() => showMessage('The result image could not be loaded. Try again.', true)}
              />
            ) : previewUrl ? (
              <img src={previewUrl} alt="Selected handwritten document" />
            ) : (
              <div className="empty-document">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M7 3.75h7l3 3v13.5H7z" />
                  <path d="M14 3.75v3h3M9.5 11h5M9.5 14h5M9.5 17h3" />
                </svg>
                <p>Upload or capture a handwritten page</p>
                <span>JPG or PNG, up to 20 MB</span>
              </div>
            )}
          </div>

          <aside className="control-panel" aria-label="Detection controls">
            {phase === 'review' && warnings.length > 0 && (
              <div>
                <h2>Check this image</h2>
                <ul className="warning-list">
                  {warnings.map((warning) => (
                    <li key={warning.code}>{warning.message}</li>
                  ))}
                </ul>
              </div>
            )}

            {result?.words_annotated_url && (
              <div className="view-toggle" role="group" aria-label="Show borders around">
                {['words', 'lines'].map((option) => (
                  <button
                    key={option}
                    type="button"
                    aria-pressed={view === option}
                    onClick={() => setView(option)}
                  >
                    {option === 'words' ? 'Words' : 'Lines'}
                  </button>
                ))}
              </div>
            )}

            {result && (
              <dl className="file-details">
                <div>
                  <dt>Handwritten lines</dt>
                  <dd>{result.count}</dd>
                </div>
                {result.word_count != null && (
                  <div>
                    <dt>Words</dt>
                    <dd>{result.word_count}</dd>
                  </div>
                )}
                <div>
                  <dt>Detector</dt>
                  <dd>{result.detector === 'easyocr' ? 'EasyOCR' : result.detector === 'paddle' ? 'PaddleOCR' : 'None'}</dd>
                </div>
                {result.serverMs != null && (
                  <div>
                    <dt>Processing time</dt>
                    <dd>{seconds(result.serverMs)}</dd>
                  </div>
                )}
                {totalMs != null && (
                  <div>
                    <dt>Total time</dt>
                    <dd>{seconds(totalMs)}</dd>
                  </div>
                )}
                <div>
                  <dt>Image size</dt>
                  <dd>
                    {result.image.width} × {result.image.height}
                  </dd>
                </div>
              </dl>
            )}

            {recognition && (
              <section className="recognized-text" aria-labelledby="recognized-title">
                <h2 id="recognized-title">Text</h2>
                <ol>
                  {recognition.lines.map((line) => (
                    <li key={line.line} className={line.low_confidence ? 'is-unclear' : undefined}>
                      {line.text || '—'}
                      {line.low_confidence && <span className="unclear-note"> (unclear)</span>}
                    </li>
                  ))}
                </ol>
              </section>
            )}

            <div className="actions">
              <input
                ref={uploadInputRef}
                className="visually-hidden"
                type="file"
                accept="image/jpeg,image/png"
                onChange={(event) => selectFile(event.target.files[0])}
              />
              <input
                ref={captureInputRef}
                className="visually-hidden"
                type="file"
                accept="image/jpeg,image/png"
                capture="environment"
                onChange={(event) => selectFile(event.target.files[0])}
              />

              {phase === 'review' ? (
                <>
                  <button className="secondary-button" type="button" onClick={retake}>
                    Retake
                  </button>
                  <button className="primary-button" type="button" onClick={continueAnyway}>
                    Continue anyway
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={isWorking}
                    onClick={() => uploadInputRef.current?.click()}
                  >
                    Upload image
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={isWorking}
                    onClick={() => captureInputRef.current?.click()}
                  >
                    Capture image
                  </button>
                  {phase === 'done' && result?.count > 0 && !recognition ? (
                    <button className="primary-button" type="button" disabled={isReading} onClick={readText}>
                      {isReading ? 'Reading…' : 'Read text'}
                    </button>
                  ) : (
                    <button
                      className="primary-button"
                      type="button"
                      disabled={!file || isWorking || phase === 'done'}
                      onClick={detect}
                    >
                      {phase === 'working' ? 'Detecting…' : 'Detect handwriting'}
                    </button>
                  )}
                </>
              )}
            </div>

            <p className={`status-message${isError ? '' : ' is-info'}`} role="status" aria-live="polite">
              {message}
            </p>
          </aside>
        </div>
      </section>
    </main>
  )
}

export default App
