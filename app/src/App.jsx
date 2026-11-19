import { useCallback, useEffect, useRef, useState } from 'react'
import { Button } from '@astryxdesign/core/Button'
import './App.css'

// true: after processing, the Data panel opens Detection details and LLM call metrics,
//       and both Run buttons are shown (Run = OCR + LLM, Run LLM only).
// false: the Data panel stays on the gallery, and only the RUN_MODE button is shown, labelled "Run".
const DEV = true
// Used when DEV is false. Set VITE_RUN_MODE in app/.env.local: "hybrid" (OCR + LLM, default) or "llm_only".
const RUN_MODE = import.meta.env.VITE_RUN_MODE === 'llm_only' ? 'llm_only' : 'hybrid'

// Must match the API's preprocessing limits (JPEG/PNG, smaller than 20 MB).
const ACCEPTED_TYPES = ['image/jpeg', 'image/png']
const MAX_BYTES = 20 * 1024 * 1024
// Detection runs at 1600 px, so photos are shrunk to 2048 px before upload:
// a 12 MP phone photo goes from ~2.5 MB to ~0.5 MB with no loss in detection.
const UPLOAD_MAX_SIDE = 2048
const UPLOAD_QUALITY = 0.9
// OCR plus two parallel LLM calls (each can take up to 120 s) in one request.
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
  if (!response.ok) {
    throw Object.assign(new Error(errorMessage(body, response.status)), { status: response.status })
  }
  if (body === null) return null
  return { ...body, serverMs: serverTotalMs(response.headers.get('Server-Timing')) }
}

// XMLHttpRequest instead of fetch, because fetch cannot report upload progress.
// path: '/api/images' (OCR + LLM) or '/api/llm-images' (LLM only, no OCR models).
function uploadImage(file, onProgress, path = '/api/images') {
  return new Promise((resolve, reject) => {
    const body = new FormData()
    body.append('file', file, file.name || 'photo.jpg')
    const xhr = new XMLHttpRequest()
    xhr.open('POST', path)
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

const stageLabels = { wait: 'Queue', preprocess: 'Preprocess', detect: 'PaddleOCR', words: 'Word boxes', llm: 'LLM calls (parallel)', total: 'Server total' }

// One LLM call (Gemini or OpenAI): tokens (input, cached, output) and time.
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
          {call.usage?.thinking_tokens != null && (
            <div><dt>Thinking tokens</dt><dd>{count(call.usage.thinking_tokens)}</dd></div>
          )}
          <div><dt>API latency</dt><dd>{call.usage?.latency_ms == null ? '—' : seconds(call.usage.latency_ms)}</dd></div>
          <div><dt>Feature time</dt><dd>{call.processing_ms == null ? '—' : seconds(call.processing_ms)}</dd></div>
        </dl>
      )}
    </div>
  )
}

const previewLayers = [
  { key: 'numbers', label: 'Box numbers', path: 'M10 4 8 20m8-16-2 16M5 9h15M4 15h15' },
  { key: 'word_boxes', label: 'Word boxes', path: 'M3 6h7v4H3zM13 6h8v4h-8zM3 14h10v4H3zM16 14h5v4h-5z' },
  { key: 'line_boxes', label: 'Line boxes', path: 'M3 6h18v4H3zM3 14h18v4H3z' },
]
const defaultLayers = { numbers: true, word_boxes: true, line_boxes: true }

const galleryKey = 'pro-detection.recent-images'
const validImageId = (id) => typeof id === 'string' && /^[a-f0-9]{32}$/.test(id)
const testImageIds = (import.meta.env.VITE_TEST_IMAGE_IDS || '').split(',').map((id) => id.trim()).filter(validImageId)
const resourceImages = Object.entries(import.meta.glob([
  '../resources/*.[jJ][pP][gG]',
  '../resources/*.[jJ][pP][eE][gG]',
  '../resources/*.[pP][nN][gG]',
], {
  eager: true,
  query: '?url',
  import: 'default',
}))
const panels = [{ title: 'Data', weight: 1 }, { title: 'Result', weight: 2 }, { title: 'Suggestion', weight: 1 }]

async function imageHash(file) {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

async function resourceUploadHashes() {
  const hashes = await Promise.all(resourceImages.map(async ([path, url]) => {
    const response = await fetch(url)
    if (!response.ok) throw new Error('Sample image unavailable')
    const file = new File([await response.blob()], path.split('/').pop(), {
      type: /\.png$/i.test(path) ? 'image/png' : 'image/jpeg',
    })
    const upload = await shrinkForUpload(file)
    return imageHash(upload)
  }))
  return new Set(hashes)
}

function readRecentImages(key = galleryKey) {
  try {
    const ids = JSON.parse(localStorage.getItem(key))
    return Array.isArray(ids) ? [...new Set(ids.filter(validImageId))] : []
  } catch {
    return []
  }
}

function spellingPattern(phrases) {
  const alternatives = phrases.map((phrase) => phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
  return new RegExp(`(?<![\\p{L}\\p{N}_])(${alternatives})(?![\\p{L}\\p{N}_])`, 'giu')
}

function HighlightedTranscript({ text, mistakes, selectedIssue, onSelect }) {
  const phrases = [...new Set(mistakes.map((mistake) => mistake.wrong_spelling?.trim()).filter(Boolean))]
    .sort((left, right) => right.length - left.length)
  if (!phrases.length) return text
  const pattern = spellingPattern(phrases)
  return text.split(pattern).map((part, index) => {
    if (index % 2 === 0) return part
    const mistake = mistakes.find((issue) => issue.wrong_spelling?.trim().toLowerCase() === part.toLowerCase())
    if (!mistake) return part
    return <button type="button" className="spelling-underline" key={index}
      aria-label={`Review spelling of ${part}`} aria-pressed={selectedIssue === mistake.box_number}
      onClick={() => onSelect(mistake.box_number)}>{part}</button>
  })
}

function WorkspacePanel({ title, collapsed, onToggle, actions, imageId, showCollapse = false, busy = false, children }) {
  const contentId = `workspace-${title.toLowerCase()}`
  return (
    <section className="workspace-panel min-w-0" hidden={title === 'Data' && collapsed} aria-labelledby={`${contentId}-heading`}>
      <div className="panel-header">
      <h2 id={`${contentId}-heading`} className={title === 'Suggestion' && collapsed ? 'visually-hidden' : 'panel-heading'}>
        {title}{imageId && <span className="image-id panel-image-id">Image ID <span>{imageId}</span></span>}
      </h2>
      {(actions || showCollapse) && (
        <div className="panel-actions">
          {actions}
          {showCollapse && <button
            className="panel-collapse-button"
            type="button"
            aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${title}`}
            aria-expanded={!collapsed}
            aria-controls={contentId}
            onClick={onToggle}
          >
            <svg className="sidebar-icon" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="3" y="4" width="18" height="16" rx="2" />
              <path d="M15 4v16" />
            </svg>
          </button>}
        </div>
      )}
      </div>
      <div id={contentId} hidden={collapsed} className="panel-content" aria-busy={busy}>
        {busy ? (
          <div className="panel-skeleton" role="status" aria-live="polite">
            <span className="visually-hidden">{title === 'Result' ? 'Processing image…' : 'Loading analysis…'}</span>
            <div aria-hidden="true" className="skeleton-content">
              {title === 'Result' && <div className="skeleton-block skeleton-image" />}
              {[0, 1, 2].map((index) => (
                <div className="skeleton-card" key={index}>
                  <div className="skeleton-block skeleton-label" />
                  <div className="skeleton-block skeleton-line" />
                  <div className="skeleton-block skeleton-line skeleton-line-short" />
                </div>
              ))}
            </div>
          </div>
        ) : children}
      </div>
    </section>
  )
}

function GalleryTile({ id, label, selected, disabled, onSelect, onMissing, thumbnail }) {
  const [loadedThumbnail, setLoadedThumbnail] = useState('')

  useEffect(() => {
    if (thumbnail) return
    const controller = new AbortController()
    callApi(`/images/${id}`, { signal: controller.signal })
      .then((body) => setLoadedThumbnail(body.annotated_url || ''))
      .catch((error) => {
        if (error.status === 404) onMissing(id)
      })
    return () => controller.abort()
  }, [id, onMissing, thumbnail])

  const src = thumbnail || loadedThumbnail
  return (
    <button
      type="button"
      className="gallery-tile relative aspect-square overflow-hidden"
      aria-label={label}
      aria-pressed={selected}
      disabled={disabled}
      onClick={() => onSelect(id)}
    >
      {src ? <img src={src} alt="" loading="lazy" className="absolute inset-0 object-cover" /> : <span>{label}</span>}
    </button>
  )
}

function App() {
  const uploadInputRef = useRef(null)
  const captureInputRef = useRef(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [pendingFile, setPendingFile] = useState(null)
  // The picked photo, kept after a saved result is shown (pendingFile is cleared then), for "Run LLM only".
  const [selectedFile, setSelectedFile] = useState(null)
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
  const requestVersion = useRef(0)
  const recordUploadRef = useRef(false)
  const [collapsed, setCollapsed] = useState({ Data: true })
  const [showDetectionDetails, setShowDetectionDetails] = useState(false)
  const [selectedIssue, setSelectedIssue] = useState(null)
  const [copyMessage, setCopyMessage] = useState('')
  const [recentIds, setRecentIds] = useState(() => readRecentImages())
  const [uploadedIds, setUploadedIds] = useState(() => readRecentImages('pro-detection.user-uploads'))
  const legacyUploadIds = useRef(readRecentImages())
  const [historyFiltered, setHistoryFiltered] = useState(false)
  const [missingIds, setMissingIds] = useState([])

  useEffect(() => {
    let cancelled = false
    const filterHistory = async () => {
      try {
        const hashes = await resourceUploadHashes()
        const entries = await Promise.allSettled(legacyUploadIds.current.map(async (id) => {
          if (uploadedIds.includes(id)) return null
          const body = await callApi(`/images/${id}`)
          return hashes.has(body.source_sha256) ? id : null
        }))
        if (!cancelled) {
          const samples = new Set(entries.filter((entry) => entry.status === 'fulfilled').map((entry) => entry.value).filter(Boolean))
          setRecentIds((ids) => ids.filter((id) => !samples.has(id)))
        }
      } catch {
        // Preserve saved entries when their source cannot be checked.
      } finally {
        if (!cancelled) setHistoryFiltered(true)
      }
    }
    filterHistory()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(galleryKey, JSON.stringify(recentIds))
    } catch {
      // The gallery remains usable for this page when browser storage is unavailable.
    }
  }, [recentIds])

  useEffect(() => {
    try {
      localStorage.setItem('pro-detection.user-uploads', JSON.stringify(uploadedIds))
    } catch {
      // Keep upload provenance available in this session.
    }
  }, [uploadedIds])

  const removeMissingImage = useCallback((id) => {
    setMissingIds((ids) => ids.includes(id) ? ids : [...ids, id])
    setRecentIds((ids) => ids.includes(id) ? ids.filter((entry) => entry !== id) : ids)
  }, [])

  const togglePanel = (title) => setCollapsed((current) => ({ ...current, [title]: !current[title] }))
  const panelColumns = (row) => {
    const visiblePanels = row.filter(({ title }) => title !== 'Data' || !collapsed.Data)
    const allCollapsed = visiblePanels.every(({ title }) => collapsed[title])
    return visiblePanels.map(({ title, weight }) => {
      if (title === 'Data') return 'clamp(16rem, 25vw, 24rem)'
      if (collapsed[title] && !allCollapsed) return '9rem'
      if (title === 'Suggestion') return 'clamp(16rem, 25vw, 24rem)'
      return `minmax(0, ${allCollapsed ? 1 : weight}fr)`
    }).join(' ')
  }
  const galleries = [
    { title: 'Discover', prefix: 'Sample', ids: [...new Set(testImageIds)].filter((id) => !missingIds.includes(id)), images: resourceImages, empty: 'No samples available.' },
    { title: 'Upload history', prefix: 'Upload', ids: historyFiltered ? recentIds.filter((id) => !testImageIds.includes(id)) : [], empty: historyFiltered ? 'Your uploads will appear here.' : 'Loading upload history…' },
  ]

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
    setCopyMessage('')
    setSelectedIssue(null)
    setShowDetectionDetails(false)
    setPendingFile(null)
    setSelectedFile(null)
    setPhase('idle')
    setImageId('')
    setWarnings([])
    setResult(null)
    setTotalMs(null)
    setLayers(defaultLayers)
    rememberImage('')
  }

  const selectFile = async (nextFile, record = true) => {
    if (phase === 'working' || !nextFile) return
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
    recordUploadRef.current = record
    setPreviewUrl(URL.createObjectURL(nextFile))
    setPendingFile(nextFile)
    setSelectedFile(nextFile)
    const version = ++requestVersion.current
    setPhase('loading')
    try {
      const upload = await shrinkForUpload(nextFile)
      const sha256 = await imageHash(upload)
      const saved = await callApi(`/image-analysis?${new URLSearchParams({ sha256 })}`)
      if (version !== requestVersion.current) return
      if (saved) {
        startedAtRef.current = performance.now()
        applyResponse(saved, record)
      } else {
        setPhase('idle')
      }
    } catch {
      if (version !== requestVersion.current) return
      setPhase('idle')
      showMessage('Saved analysis could not be checked. Click Run to run this image.')
    }
  }

  const handleFileChange = (event) => {
    selectFile(event.target.files[0])
    event.target.value = ''
  }

  const selectResourceImage = async (name, url) => {
    if (isWorking) return
    const version = ++requestVersion.current
    setPhase('working')
    showMessage('Loading sample image…')
    let file
    try {
      const response = await fetch(url)
      if (!response.ok) throw new Error()
      file = new File([await response.blob()], name, {
        type: /\.png$/i.test(name) ? 'image/png' : 'image/jpeg',
      })
    } catch {
      if (version !== requestVersion.current) return
      setPhase(phase)
      showMessage('Could not load this sample image.', true)
      return
    }
    if (version !== requestVersion.current) return
    setPhase(phase)
    selectFile(file, false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  const applyResponse = (body, record) => {
    if (!validImageId(body?.image_id) || (body.status !== 'needs_review' && (body.status !== 'done' || !body.image))) {
      throw new Error('The server returned an incomplete analysis. Please try again.')
    }
    setPendingFile(null)
    setImageId(body.image_id)
    rememberImage(body.image_id)
    setWarnings(body.warnings || [])
    if (body.status === 'needs_review') {
      setPhase('review')
      showMessage('This image may be hard to read. Retake it or continue anyway.', true)
      return
    }
    if (record && validImageId(body.image_id)) {
      setRecentIds((ids) => [body.image_id, ...ids.filter((id) => id !== body.image_id)])
      setUploadedIds((ids) => ids.includes(body.image_id) ? ids : [...ids, body.image_id])
    }
    setResult(body)
    setShowDetectionDetails(DEV)
    setPhase('done')
    showMessage('')
  }

  const resultLoaded = () => {
    if (totalMs != null) return // Changing preview overlays reloads the image.
    setTotalMs(performance.now() - startedAtRef.current)
  }

  let detectionSummary = ''
  if (result) {
    const lines = `${result.count} handwritten ${result.count === 1 ? 'line' : 'lines'}`
    const words = result.word_count ? ` and ${result.word_count} ${result.word_count === 1 ? 'word' : 'words'}` : ''
    detectionSummary = result.count === 0 ? 'No handwriting was found in this image.' : `Found ${lines}${words}.`
  }

  const transcription = result?.llm?.transcription
  const spelling = result?.llm?.spelling
  const spellingIssues = spelling?.mistakes ?? []
  const correctSpelling = (mistake) => {
    const wrong = mistake.wrong_spelling?.trim()
    if (!wrong || !transcription?.text) return
    const corrected = transcription.text.replace(spellingPattern([wrong]), () => mistake.correct_spelling)
    if (corrected === transcription.text) {
      showMessage('This spelling could not be found in the extracted text.', true)
      return
    }
    setResult((current) => ({ ...current, llm: {
      ...current.llm,
      transcription: { ...current.llm.transcription, text: corrected },
      spelling: { ...current.llm.spelling, mistakes: current.llm.spelling.mistakes.filter((issue) => issue.wrong_spelling?.trim().toLowerCase() !== wrong.toLowerCase()) },
    } }))
    setSelectedIssue(null)
    setCopyMessage('')
  }

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(transcription.text)
      setCopyMessage('Copied')
    } catch {
      setCopyMessage('Copy failed. Select the text to copy it.')
    }
  }
  const resultImageUrl = result?.words_annotated_url
    ? `${result.words_annotated_url}?${new URLSearchParams({ ...layers, spelling: spellingIssues.length > 0, crop_to_text: true })}`
    : result?.annotated_url

  const runRequest = async (request, phaseOnError, startMessage, { record = false, saved = '' } = {}) => {
    const version = ++requestVersion.current
    setPhase(saved ? 'loading' : 'working')
    startedAtRef.current = performance.now()
    showMessage(startMessage)
    try {
      const body = await request()
      if (version === requestVersion.current) applyResponse(body, record)
    } catch (error) {
      if (version !== requestVersion.current) return
      setPhase(phaseOnError)
      if (saved && error.status === 404) {
        removeMissingImage(saved)
        setImageId('')
        rememberImage('')
        showMessage('This saved image is no longer available.', true)
        return
      }
      showMessage(error.message, true)
    }
  }

  const selectSavedImage = (id, record = false) => {
    if (!validImageId(id)) return
    resetResult()
    recordUploadRef.current = uploadedIds.includes(id)
    rememberImage(id)
    setImageId(id)
    setPreviewUrl('')
    runRequest(() => callApi(`/images/${id}`), 'idle', 'Loading the saved result…', { record, saved: id })
  }

  // Reopen the active image after a page reload.
  useEffect(() => {
    const saved = savedImageId()
    if (saved) selectSavedImage(saved)
    return () => { requestVersion.current += 1 }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const continueAnyway = () =>
    runRequest(
      () => callApi(`/images/${imageId}/confirm`, { method: 'POST' }),
      'review',
      'Detecting handwriting and reading it with the LLM…',
      { record: recordUploadRef.current },
    )

  const analyseImage = () => {
    if (isWorking) return
    setTotalMs(null)
    if (pendingFile) {
      runRequest(async () => {
        const upload = await shrinkForUpload(pendingFile)
        return uploadImage(upload, (done) =>
          showMessage(done < 1 ? `Uploading… ${Math.round(done * 100)}%` : 'Detecting handwriting and reading it with the LLM…'),
        )
      }, 'idle', 'Preparing the photo…', { record: recordUploadRef.current })
      return
    }
    if (result) {
      runRequest(() => callApi(`/images/${imageId}/analyse`, { method: 'POST' }), 'done', 'Detecting handwriting and reading it with the LLM…', { record: recordUploadRef.current })
    }
  }

  // Send the photo straight to the LLM: no preprocessing, PaddleOCR or word boxes.
  const analyseLlmOnly = () => {
    if (isWorking || (!selectedFile && !imageId)) return
    setTotalMs(null)
    if (!selectedFile) {
      // Already uploaded (gallery or page reload): the server reuses the photo it stored.
      runRequest(() => callApi(`/images/${imageId}/llm-only`, { method: 'POST' }), 'done',
        'Reading it with the LLM (no OCR)…', { record: recordUploadRef.current })
      return
    }
    runRequest(async () => {
      const upload = await shrinkForUpload(selectedFile)
      return uploadImage(upload, (done) =>
        showMessage(done < 1 ? `Uploading… ${Math.round(done * 100)}%` : 'Reading it with the LLM (no OCR)…'),
      '/api/llm-images')
    }, 'idle', 'Preparing the photo…', { record: recordUploadRef.current })
  }

  const clearSelection = () => {
    resetResult()
    setPreviewUrl('')
    showMessage('')
  }

  const retake = async () => {
    setPhase('working')
    try {
      await callApi(`/images/${imageId}`, { method: 'DELETE' })
    } catch (error) {
      if (error.status !== 404) {
        setPhase('review')
        showMessage(error.message, true)
        return
      }
    }
    removeMissingImage(imageId)
    clearSelection()
    showMessage('Upload or capture a new image.')
  }

  const isWorking = phase === 'working' || phase === 'loading'

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-brand">
          <button className="panel-collapse-button" type="button"
            aria-label={`${collapsed.Data ? 'Expand' : 'Collapse'} Data`}
            title={`${collapsed.Data ? 'Expand' : 'Collapse'} Data`}
            aria-expanded={!collapsed.Data} aria-controls="workspace-data"
            onClick={() => togglePanel('Data')}>
            <svg className="sidebar-icon" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="3" y="4" width="18" height="16" rx="2" />
              <path d="M9 4v16" />
            </svg>
          </button>
          <a className="wordmark" href="/" aria-label="Handwriting home">
            Pro Detection
          </a>
        </div>
      </header>

      <section
        className="workspace grid w-full content-start"
        style={{
          '--tablet-columns': panelColumns(panels.slice(0, 2)),
          '--desktop-columns': panelColumns(panels),
        }}
        aria-label="Handwriting detection"
      >
        <WorkspacePanel title="Data" collapsed={collapsed.Data} onToggle={() => togglePanel('Data')}>
          <div
            className={`upload-area flex flex-wrap ${showDetectionDetails ? 'justify-start' : 'justify-center'} gap-2${isDragging ? ' is-dragging' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false)
            }}
            onDrop={handleDrop}
          >
            {showDetectionDetails ? (
              <Button label="Back" variant="primary" onClick={() => setShowDetectionDetails(false)} />
            ) : (
              <>
                <Button label="Upload image" size="lg" width="min(100%, 10rem)" variant="primary" isDisabled={isWorking} onClick={() => uploadInputRef.current?.click()} />
                <Button label="Capture image" size="lg" width="min(100%, 10rem)" isDisabled={isWorking} onClick={() => captureInputRef.current?.click()} />
              </>
            )}
          </div>
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

          <input
            ref={uploadInputRef}
            hidden
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleFileChange}
          />
          <input
            ref={captureInputRef}
            hidden
            type="file"
            accept="image/jpeg,image/png"
            capture="environment"
            onChange={handleFileChange}
          />

          {phase === 'review' && (
            <div className="review-actions">
              <Button label="Retake" onClick={retake} />
              <Button label="Continue anyway" variant="primary" onClick={continueAnyway} />
            </div>
          )}

          {showDetectionDetails && result && (
            <div className="detection-summary">
              <section className="result-details" aria-labelledby="detection-details-heading" key={imageId}>
                <h3 id="detection-details-heading">Detection details</h3>
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
                    <dd>{result.detector === 'easyocr' ? 'EasyOCR' : result.detector === 'paddle' ? 'PaddleOCR' : result.detector === 'llm' ? 'LLM only (no OCR)' : 'None'}</dd>
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
              </section>
            </div>
          )}
          {showDetectionDetails && result?.llm && (
            <section className="result-details llm-details" aria-labelledby="llm-details-heading" key={`llm-${imageId}`}>
              <h3 id="llm-details-heading">LLM calls</h3>
              <LlmCallMetrics title="Transcription" call={transcription} />
              <LlmCallMetrics title="Spelling check" call={spelling} />
            </section>
          )}
          <div hidden={showDetectionDetails} className="data-gallery-content">
          {galleries.map(({ title, prefix, ids, images = [], empty }) => (
            <section key={prefix} className="image-gallery" aria-labelledby={`gallery-${prefix.toLowerCase()}`}>
              <h3 id={`gallery-${prefix.toLowerCase()}`}>{title}</h3>
              {ids.length || images.length ? (
                <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,5rem),1fr))] gap-2">
                  {images.map(([path, url]) => {
                    const name = path.slice(path.lastIndexOf('/') + 1)
                    return (
                      <GalleryTile
                        key={path}
                        id={path}
                        label={name}
                        disabled={isWorking}
                        onSelect={() => selectResourceImage(name, url)}
                        thumbnail={url}
                      />
                    )
                  })}
                  {ids.map((id, index) => (
                    <GalleryTile
                      key={id}
                      id={id}
                      label={`${prefix} ${index + 1}`}
                      selected={imageId === id}
                      disabled={phase === 'working'}
                      onSelect={selectSavedImage}
                      onMissing={removeMissingImage}
                      thumbnail={imageId === id ? result?.annotated_url : undefined}
                    />
                  ))}
                </div>
              ) : <p className="panel-copy">{empty}</p>}
            </section>
          ))}
          </div>
        </WorkspacePanel>

        <WorkspacePanel title="Result" imageId={imageId} busy={isWorking} collapsed={false} actions={
          <>
          {(DEV || RUN_MODE === 'hybrid') && (
            <Button label="Run" variant="primary" isDisabled={(!pendingFile && !result) || isWorking} onClick={analyseImage} />
          )}
          {(DEV || RUN_MODE === 'llm_only') && (
            <Button label={DEV ? 'Run LLM only' : 'Run'} variant={DEV ? undefined : 'primary'}
              isDisabled={(!selectedFile && !imageId) || isWorking} onClick={analyseLlmOnly} />
          )}
                {result?.words_annotated_url && (
                  <details className="preview-filter">
                    <summary className="view-toggle" aria-label="Overlay filters" title="Overlay filters">
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M4 5h16l-6 7v6l-4 2v-8z" />
                      </svg>
                    </summary>
                  <div className="preview-toggles" role="group" aria-label="Overlay filters">
                    {previewLayers.map(({ key, label, path }) => (
                      <button
                        key={key}
                        className="view-toggle"
                        data-layer={key}
                        type="button"
                        aria-pressed={layers[key]}
                        onClick={() => setLayers((current) => ({ ...current, [key]: !current[key] }))}
                      >
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d={path} />
                        </svg>
                        {label}
                      </button>
                    ))}
                  </div>
                  </details>
                )}
          </>

        }>
          <div className={result ? 'result-layout' : undefined}>
          {result && <p className="image-detection-summary" role="status">{detectionSummary}</p>}
          <div className={result ? 'result-card' : undefined}>
          <div
            className={`document-stage${result ? ' has-result' : ''}${isDragging ? ' is-dragging' : ''}`}
            onDragEnter={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false)
            }}
            onDrop={handleDrop}
            aria-busy={isWorking}
          >
            {result ? (
              <div className="document-result">
                <img
                  key={resultImageUrl}
                  src={resultImageUrl}
                  alt="Handwritten document preview"
                  onLoad={resultLoaded}
                  onError={() => showMessage('The preview could not be loaded. Select the image again to retry.', true)}
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
                <p>{phase === 'loading' ? 'Loading saved image…' : 'Select or upload an image to preview it.'}</p>
              </div>
            )}
          </div>

          <p className={`status-message${isError ? '' : ' is-info'}`} role="status" aria-live="polite" hidden={!message}>
            {message}
          </p>


          </div>

          {transcription && (
            <section className="result-details extracted-text" aria-label="Extracted text" key={`text-${imageId}`}>
              {transcription.status === 'failed' ? (
                <p className="status-message">{transcription.error?.message || 'Transcription is unavailable.'}</p>
              ) : (
                <p className="transcript"><HighlightedTranscript text={transcription.text || 'No text was read.'} mistakes={spellingIssues} selectedIssue={selectedIssue} onSelect={(number) => {
                  setSelectedIssue(number)
                  setCollapsed((current) => ({ ...current, Suggestion: false }))
                }} /></p>
              )}
              {transcription.status !== 'failed' && transcription.text && (
                <div className="text-copy-actions">
                  <button className="copy-text-button" type="button" onClick={copyText} aria-label="Copy text" title="Copy text">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2" /><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" /></svg>
                  </button>
                  <span role="status">{copyMessage}</span>
                </div>
              )}
            </section>
          )}
          </div>
        </WorkspacePanel>

        <WorkspacePanel title="Suggestion" showCollapse busy={isWorking} collapsed={collapsed.Suggestion} onToggle={() => togglePanel('Suggestion')}>
          {!result && (
            <>
              <p className="panel-copy">
                {pendingFile ? 'Click Run to analyse the selected image.' : 'Select an image to see its saved analysis.'}
              </p>
            </>
          )}
          {result && !transcription && !spelling && (
            <p className="panel-copy">No saved analysis is available for this image.</p>
          )}

          {spelling && (
            <section className="writing-suggestions" aria-label="Spelling suggestions" key={`spelling-${imageId}`}>
              <h3 className="suggestions-heading">{spellingIssues.length ? `${spellingIssues.length} spelling suggestions` : 'Spelling'}</h3>
              {spelling.status === 'failed' ? (
                <p className="status-message">{spelling.error?.message || 'Spelling analysis is unavailable.'}</p>
              ) : spellingIssues.length ? (
                <ul className="suggestion-cards">
                  {spellingIssues.map((mistake) => (
                    <li key={mistake.box_number}>
                      <div className="suggestion-card" data-selected={selectedIssue === mistake.box_number}>
                        <button className="suggestion-select" type="button" aria-pressed={selectedIssue === mistake.box_number} onClick={() => setSelectedIssue(mistake.box_number)}>
                        <span className="suggestion-category">Spelling</span>
                        <span className="suggestion-change"><del>{mistake.wrong_spelling}</del><span aria-hidden="true">→</span><strong>{mistake.correct_spelling}</strong></span>
                        </button>
                        <div className="suggestion-card-actions"><Button label="Correct" variant="primary" onClick={() => correctSpelling(mistake)} /></div>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="spelling-issues">No spelling mistakes flagged.</p>
              )}
            </section>
          )}


        </WorkspacePanel>
      </section>
    </main>
  )
}

export default App
