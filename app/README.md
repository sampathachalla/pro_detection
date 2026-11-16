# app

Web app (React 19, Vite and Tailwind) for uploading or capturing a handwritten page and seeing the detected handwriting outlined.

| Path | What it is | What it does |
|---|---|---|
| `index.html` | Vite entry page | Loads `src/main.jsx`. |
| `vite.config.js` | Vite config | Plugins, plus a **dev proxy**: `/api/*` → `API_PROXY_TARGET` (default `http://localhost:8000`; the `dev_tunnel` container sets `http://host.docker.internal:8000`). |
| `src/main.jsx` | React entry | Mounts `App`. |
| `src/App.jsx` | The workspace screen | Upload or capture → **Detect handwriting** → shows the annotated image and a summary, or the warnings with **Retake** or **Continue anyway**. |
| `src/App.css`, `src/index.css`, `src/base.css`, `src/tokens.css` | Styles | Layout, components and design tokens. |
| `public/` | Static files | Favicon and icons. |
| `dist/` | Build output | Created by `npm run build`. |

## How `App.jsx` talks to the API

| User action | Request | What the page shows |
|---|---|---|
| Picks or captures a file | none. The page checks it's JPEG or PNG and smaller than 20 MB. | A preview |
| **Detect handwriting** | The photo is shrunk to 2048 px JPEG in the browser, then `POST /api/images` with upload progress | **done:** the word-border image (`words_annotated_url`), with a **Words / Lines** switch to `annotated_url`; line and word counts, detector, processing and total time, and image size. **needs_review:** the warnings plus Retake / Continue anyway. |
| **Read text** (after a result) | `POST /api/images/{id}/read` | The text of each line in a numbered list, with unclear lines greyed out |
| **Continue anyway** | `POST /api/images/{id}/confirm` | The annotated image and summary |
| **Retake** | `DELETE /api/images/{id}` | Clears the page for a new upload |

Errors show the API's `error.message` in the status line. The full API is described in [`../api/src/server/README.md`](../api/src/server/README.md).

## Run

```bash
npm install
npm run dev        # http://localhost:5173, /api proxied to localhost:8000
```

Or through `../dev_tunnel` (Docker and the mobile tunnel). The API container must be running: `docker compose up -d` in the project root.
