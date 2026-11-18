# Workspace UI contract

| Panel | Expanded content |
|---|---|
| Result | Original preview for a new upload and annotated preview when available. Existing overlay controls and detection details. |
| Data | Upload and capture controls, quality review actions, and a compact thumbnail grid below. Use consistent tile heights, cover crops and keyboard-selectable tiles with a visible selected state. |
| Suggestion | Recognized text and spelling corrections with box numbers where available. Distinguish no selected image, analysis in progress, unavailable analysis and completed analysis with no mistakes. |

Below the header, the desktop grid spans the entire available viewport width. Expanded tracks fill the space left by collapsed headers. When all panels collapse, distribute the three reopen controls across the width. On narrow screens, stack the three sections at full width. Panel controls expose expanded state and work with a keyboard.

Selecting a gallery tile loads saved analysis for that ID without submitting a new upload. Result and Suggestion refer to the same selection. A failed or superseded selection cannot display stale mistakes for the active image.
