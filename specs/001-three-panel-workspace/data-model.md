# Data model

| Entity | Fields and behavior |
|---|---|
| Gallery entry | Image ID, source (configured test image or browser upload), optional display label and saved timestamp. Deduplicate by ID. Resolve thumbnail URL from existing result metadata. |
| Active image | Image ID when assigned, local preview URL for a new upload, phase, detection result, warnings and available recognition output. Clear prior analysis on selection and apply asynchronous responses only to the active selection. |
| Panel state | An expanded flag for Result, Data and Suggestion. Independent of selection, upload and analysis state. |

Persist only gallery metadata. Record completed uploads, including uploads that finish after quality confirmation. Keep entries on transient failures. Remove confirmed expired entries from browser history and skip missing configured IDs until the fixtures are restored. Revoke local preview URLs when replaced.
