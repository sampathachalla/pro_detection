# app

Web app (React 19, Vite and Tailwind) for uploading or capturing a handwritten page and seeing the detected handwriting outlined.

| Path | What it is | What it does |
|---|---|---|
| `index.html` | Vite entry page | Loads `src/main.jsx`. |
| `vite.config.js` | Vite config | Plugins, plus a **dev proxy**: `/api/*` → `API_PROXY_TARGET` (default `http://localhost:8000`). |
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

The API container must be running: `docker compose up -d` in the project root.

To seed the Data panel with images already uploaded to the same OCR backend, add their comma-separated IDs to `VITE_TEST_IMAGE_IDS` in `app/.env.local` and restart Vite. An image ID is the value after `?image=` when viewing a saved result. The gallery also remembers completed uploads in the current browser. Saved images are available while the backend retains them.

**Dev view and run mode.** `DEV` at the top of `src/App.jsx` switches the developer view:

| `DEV` | Data panel after processing | Run buttons |
|---|---|---|
| `true` | Detection details and LLM call metrics | both: **Run** (OCR + LLM) and **Run LLM only** |
| `false` | the gallery | one **Run** button, chosen by `VITE_RUN_MODE` |

Set `VITE_RUN_MODE` in `app/.env.local` to `hybrid` (OCR + LLM, the default) or `llm_only` (photo straight to the LLM, no OCR models), then restart Vite.

Add JPG, JPEG, or PNG files directly to `app/resources/` to show them in Discover. Selecting a sample runs the normal OCR upload flow. Rebuild the frontend to include new files in a production deployment.

## Evaluate sample spelling

With the OCR API running on port 8000, run `python evaluate_spelling.py 1 10` from the project root. The range is inclusive. The script uploads those original JPEGs through the API, continues past preprocessing warnings, and compares the spelling predictions with `resources/test_results.json`, ignoring case and surrounding whitespace. When a prediction covers several words, the evaluator aligns the original and corrected text and scores the changed words.

The report shows OCR line and word box counts, correct suggestions, wrong suggestions, missed words, and extra flags relative to the key. Detection recall and precision measure whether the expected words were flagged. The correction rate measures how many expected words received the right suggestion. Failed images appear separately and are excluded from the scores.
