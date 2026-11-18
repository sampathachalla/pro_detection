# Implementation plan for the three-panel workspace

## Summary

Build a full-width React workspace with Result on the left, Data in the middle and Suggestion on the right. Use Tailwind for layout and Astryx Neutral for interactive controls. Keep the existing detection and image-quality review flow.

## Technical context

- React 19 and Vite with Tailwind 4 and the installed Astryx 0.7.16 packages.
- Work in app/src/App.jsx, app/src/App.css and the React entry point and stylesheet as needed for Astryx setup.
- Use the existing upload, confirm, retake and image detail endpoints. Configure explicitly approved existing test image IDs for the initial gallery. Reopen saved analysis by ID.
- Use the desktop layout in the [UI contract](contracts/workspace.md). Start with all three panels expanded and give Result the widest column.

## Constitution check

The repository constitution contains placeholders. The supplied AGENTS.md instructions govern scope, reuse and verification. The design uses the existing packages and API. Validation scenarios are documented for a separately authorized run.

## Phase 0 research

See [research.md](research.md) for the layout and gallery decisions.

## Phase 1 design

1. Wire the Astryx Neutral provider and required CSS into the current entry point. Check cascade order against Tailwind and existing app styles during implementation.
2. Put upload, camera and the recent photo grid in Data. Load configured test IDs and remember completed uploads in the current browser. Reuse one shared selection and request flow.
3. Move preview, overlay controls and detection details into Result. Keep the original local preview visible while a new image is analyzed. Move recognized text and spelling findings into Suggestion.
4. Apply responsive Tailwind grid tracks and independent accessible panel toggles. Preserve selection and in-flight requests when a panel collapses.
5. Use the [data model](data-model.md), [UI contract](contracts/workspace.md) and [quickstart](quickstart.md) to verify the flow.

## Post-design check

Existing API routes cover image selection and saved analysis. Populating the initial test gallery requires valid existing image IDs. The plan keeps the existing retention policy and handles expired entries.
