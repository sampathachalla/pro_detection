# Validation quickstart

After implementation, start the existing OCR backend and run npm run dev from app. Configure at least one known, unexpired test image ID.

1. On a wide viewport, confirm the three panels fill the width. Collapse each panel, check that the remaining open panels expand and reopen all three. Verify the all-collapsed state exposes every reopen control.
2. Upload a JPG or PNG and repeat with camera capture on a supported device. Verify the local preview, quality review if triggered, annotated result and gallery entry.
3. Select the configured test image and another gallery item. Confirm previews and suggestions match the active image, including after rapid selection and page reload. Check that reopening a saved image does not send an upload request.
4. Verify missing or expired images and temporary request failures. Confirm the empty, loading, no-mistakes and analysis-unavailable states.
5. Confirm keyboard access and full-width stacked panels on a narrow viewport.
