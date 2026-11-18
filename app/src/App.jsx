import { useEffect, useRef, useState } from 'react'
import './App.css'

// Must match the API's preprocessing limits (JPEG/PNG, smaller than 20 MB).
const ACCEPTED_TYPES = ['image/jpeg', 'image/png']
const MAX_BYTES = 20 * 1024 * 1024
// Detection runs at 1600 px, so photos are shrunk to 2048 px before upload:
// a 12 MP phone photo goes from ~2.5 MB to ~0.5 MB with no loss in detection.
const UPLOAD_MAX_SIDE = 2048
const UPLOAD_QUALITY = 0.9
// OCR plus two parallel OpenAI calls (each can take up to 120 s) in one request.
const REQUEST_TIMEOUT_MS = 240000

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

// The current image id lives in the address (?image=…), so a reload can fetch the saved result.
const savedImageId = () => new URLSearchParams(window.location.search).get('image')

function rememberImage(imageId) {
  const url = new URL(window.location.href)
  if (imageId) url.searchParams.set('image', imageId)
  else url.searchParams.delete('image')
  window.history.replaceState(null, '', url)
}

const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`
const count = (value) => (value == null ? '—' : value.toLocaleString())

const stageLabels = { wait: 'Queue', preprocess: 'Preprocess', detect: 'PaddleOCR', words: 'Word boxes', llm: 'OpenAI (parallel)', total: 'Server total' }

// One OpenAI call: tokens (input, cached, output) and time.
function LlmCallMetrics({ title, call }) {
  if (!call) return null
  return (
    <div className="llm-call">
      <h3>{title}</h3>
      {call.status === 'failed' ? (
        <p className="status-message">{call.error?.message || 'This call failed.'}</p>
      ) : (
        <dl className="recognition-metrics">
          <div><dt>Model</dt><dd>{call.model}</dd></div>
          <div><dt>Requests</dt><dd>{count(call.request_count)}</dd></div>
          <div><dt>Input tokens</dt><dd>{count(call.usage?.input_tokens)}</dd></div>
          <div><dt>Cached tokens</dt><dd>{count(call.usage?.cached_tokens)}</dd></div>
          <div><dt>Output tokens</dt><dd>{count(call.usage?.output_tokens)}</dd></div>
          <div><dt>OpenAI latency</dt><dd>{call.usage?.latency_ms == null ? '—' : seconds(call.usage.latency_ms)}</dd></div>
          <div><dt>Feature time</dt><dd>{call.processing_ms == null ? '—' : seconds(call.processing_ms)}</dd></div>
        </dl>
      )}
    </div>
  )
}

const previewLayers = [
  { key: 'numbers', label: 'Box numbers', path: 'M10 4 8 20m8-16-2 16M5 9h15M4 15h15' },
  { key: 'word_boxes', label: 'Blue word boxes', path: 'M3 6h7v4H3zM13 6h8v4h-8zM3 14h10v4H3zM16 14h5v4h-5z' },
  { key: 'line_boxes', label: 'Green line boxes', path: 'M3 6h18v4H3zM3 14h18v4H3z' },
]
const defaultLayers = { numbers: true, word_boxes: true, line_boxes: true }

function App() {
  const uploadInputRef = useRef(null)
  const captureInputRef = useRef(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [message, setMessage] = useState('')
  const [isError, setIsError] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  // idle -> working -> review (warnings: retake or continue) | done
  const [phase, setPhase] = useState('idle')
  const [imageId, setImageId] = useState('')
  const [warnings, setWarnings] = useState([])
  const [result, setResult] = useState(null)
  const [layers, setLayers] = useState(defaultLayers)
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
    setLayers(defaultLayers)
    rememberImage('')
  }

  const selectFile = (nextFile) => {
    if (isWorking || !nextFile) return
    showMessage('')

    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      showMessage('Choose a JPG or PNG image.', true)
      return
    }
    if (nextFile.size >= MAX_BYTES) {
      showMessage('The image must be smaller than 20 MB.', true)
      return
    }

    resetResult()
    setPreviewUrl(URL.createObjectURL(nextFile))
    runRequest(
      async () => {
        const upload = await shrinkForUpload(nextFile)
        return uploadImage(upload, (done) =>
          showMessage(done < 1 ? `Uploading… ${Math.round(done * 100)}%` : 'Detecting handwriting and reading it with OpenAI…'),
        )
      },
      'idle',
      'Preparing the photo…',
    )
  }

  const handleFileChange = (event) => {
    selectFile(event.target.files[0])
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  const applyResponse = (body) => {
    setImageId(body.image_id)
    rememberImage(body.image_id)
    setWarnings(body.warnings || [])
    if (body.status === 'needs_review') {
      setPhase('review')
      showMessage('This image may be hard to read. Retake it or continue anyway.', true)
      return
    }
    setResult(body)
    setPhase('done')
    showMessage('Loading the result image…')
  }

  const resultLoaded = () => {
    if (totalMs != null) return // Changing preview overlays reloads the image.
    setTotalMs(performance.now() - startedAtRef.current)
    showMessage('')
  }

  let detectionSummary = ''
  if (result && totalMs != null) {
    const lines = `${result.count} handwritten ${result.count === 1 ? 'line' : 'lines'}`
    const words = result.word_count ? ` and ${result.word_count} ${result.word_count === 1 ? 'word' : 'words'}` : ''
    detectionSummary = result.count === 0 ? 'No handwriting was found in this image.' : `Found ${lines}${words}.`
  }

  const transcription = result?.llm?.transcription
  const spelling = result?.llm?.spelling
  const spellingIssues = spelling?.mistakes ?? []
  const resultImageUrl = result?.words_annotated_url
    ? `${result.words_annotated_url}?${new URLSearchParams({ ...layers, spelling: spellingIssues.length > 0, crop_to_text: true })}`
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

  // After a reload, show the saved result (OCR, text and mistakes) for the image in the address.
  useEffect(() => {
    const saved = savedImageId()
    if (!saved) return
    runRequest(
      async () => {
        try {
          return await callApi(`/images/${encodeURIComponent(saved)}`)
        } catch (error) {
          rememberImage('') // expired (24 h) or deleted: start fresh
          throw error
        }
      },
      'idle',
      'Loading the saved result…',
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const continueAnyway = () =>
    runRequest(
      () => callApi(`/images/${imageId}/confirm`, { method: 'POST' }),
      'review',
      'Detecting handwriting and reading it with OpenAI…',
    )

  const retake = async () => {
    try {
      await callApi(`/images/${imageId}`, { method: 'DELETE' })
    } catch {
      // Already gone or expired: nothing to clean up.
    }
    resetResult()
    setPreviewUrl('')
    showMessage('Upload or capture a new image.')
  }

  const isWorking = phase === 'working'

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-brand">
          {(previewUrl || result) && (
            <button
              className="back-button"
              type="button"
              disabled={isWorking}
              onClick={retake}
              aria-label="Back to upload"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M15 18l-6-6 6-6" />
              </svg>
              Back
            </button>
          )}
          <a className="wordmark" href="/" aria-label="Handwriting home">
          Pro Detection
        </a>
        </div>
        <p className="topbar-note">Text detection workspace</p>
      </header>

      <section className="workspace" aria-label="Handwriting detection">
        <div className="work-area">
          <div
            className={`document-stage${result ? ' has-result' : ''}${isDragging ? ' is-dragging' : ''}`}
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
              <div className="document-result">
                <div className="document-result-header">
                  <h2>Preview</h2>
                  {result.words_annotated_url && (
                    <div className="preview-toggles" role="group" aria-label="Preview overlays">
                      {previewLayers.map(({ key, label, path }) => (
                        <button
                          key={key}
                          className="view-toggle"
                          data-layer={key}
                          type="button"
                          aria-label={label}
                          aria-pressed={layers[key]}
                          onClick={() => setLayers((current) => ({ ...current, [key]: !current[key] }))}
                        >
                          <svg viewBox="0 0 24 24" aria-hidden="true">
                            <path d={path} />
                          </svg>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <img
                  key={resultImageUrl}
                  src={resultImageUrl}
                  alt="Handwritten document preview"
                  onLoad={resultLoaded}
                  onError={() => showMessage('The result image could not be loaded. Try again.', true)}
                />
              </div>
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
                <div className="empty-document-actions">
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
                </div>
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

            {result && (
              <div className="detection-summary">
                <p className="status-message is-info" role="status" hidden={!detectionSummary}>{detectionSummary}</p>
                <details className="result-details" key={imageId}>
                  <summary>Detection details</summary>
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
                    {Object.entries(result.timings_ms || {}).map(([stage, ms]) => (
                      <div key={stage}>
                        <dt>{stageLabels[stage] || stage}</dt>
                        <dd>{seconds(ms)}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              </div>
            )}

            {transcription && (
              <details className="result-details" open key={`text-${imageId}`}>
                <summary>Text</summary>
                {transcription.status === 'failed' ? (
                  <p className="status-message">{transcription.error?.message}</p>
                ) : (
                  <p className="transcript">{transcription.text || 'No text was read.'}</p>
                )}
              </details>
            )}

            {spelling && (
              <details className="result-details" open key={`spelling-${imageId}`}>
                <summary>Spelling mistakes</summary>
                {spelling.status === 'failed' ? (
                  <p className="status-message">{spelling.error?.message}</p>
                ) : spellingIssues.length ? (
                  <ul className="spelling-issues">
                    {spellingIssues.map((mistake) => (
                      <li key={mistake.box_number}>
                        Box {mistake.box_number}: <span>{mistake.wrong_spelling}</span> → {mistake.correct_spelling}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="spelling-issues">No spelling mistakes flagged.</p>
                )}
              </details>
            )}

            {result?.llm && (
              <details className="result-details" key={`llm-${imageId}`}>
                <summary>OpenAI calls</summary>
                <LlmCallMetrics title="Transcription" call={transcription} />
                <LlmCallMetrics title="Spelling check" call={spelling} />
              </details>
            )}

            <div className="actions">
              <input
                ref={uploadInputRef}
                className="visually-hidden"
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleFileChange}
              />
              <input
                ref={captureInputRef}
                className="visually-hidden"
                type="file"
                accept="image/jpeg,image/png"
                capture="environment"
                onChange={handleFileChange}
              />

              {phase === 'review' && (
                <>
                  <button className="secondary-button" type="button" onClick={retake}>
                    Retake
                  </button>
                  <button className="primary-button" type="button" onClick={continueAnyway}>
                    Continue anyway
                  </button>
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
