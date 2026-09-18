---
name: browser-check
description: Check page computed styles, hover, theme, and selector screenshots for CSS/layout/visual UI work instead of writing CDP scripts.
---

# Browser Check

Use the bundled helper for visual UI verification. Do **not** write Chrome DevTools (CDP) scripts, launch ad-hoc headless Chrome, or invent Playwright/Puppeteer one-offs — that burns turns on plumbing. Fill in URL, selectors, and the style properties you care about; the helper returns compact JSON plus screenshot paths.

## When to use

- CSS/layout bugs, hover/focus states, dark/light theme, computed colors/fonts/spacing
- Confirming a page looks right after an edit (selector clip + styles, not a full-page dump)

If a browser MCP server is already connected and clearly covers the same check, you may use that instead. Otherwise use this helper.

## How to run

The skill folder (see `Folder:` above) contains `scripts/browser-check.mjs`. Needs Node (built-in `fetch`/`WebSocket`, v18+) and Chrome/Chromium. Do not `npm install`.

Write a JSON request file, then:

```bash
node "<folder>/scripts/browser-check.mjs" --request /tmp/browser-check-req.json
```

`--request -` reads stdin. Prints one JSON object to stdout. Screenshots are files — `read_file` them only if you need to look; never paste base64 or a full DOM dump into the conversation.

### Request

```json
{
  "url": "http://127.0.0.1:4321/reference/property-shape",
  "viewport": [900, 900],
  "outDir": "/tmp/aivo-browser-check",
  "theme": {
    "colorScheme": "dark"
  },
  "checks": [
    {
      "id": "see-also",
      "selector": ".see-also a",
      "index": 0,
      "styles": ["color", "text-decoration", "text-decoration-color"],
      "screenshot": true,
      "hover": true,
      "hoverScreenshot": true
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `url` | `http(s):` or `file:` page to open |
| `viewport` | `[width, height]`, default `[1024, 768]` |
| `outDir` | Directory for PNGs (created if missing). Default: a temp dir |
| `chrome` | Optional Chrome/Chromium executable. Otherwise the helper probes common paths |
| `theme.colorScheme` | `light` / `dark` / `no-preference` — emulates `prefers-color-scheme` (not a site's own class/data-theme) |
| `theme.evaluate` | Optional JS run after load to flip an **app** theme (`class` / `data-theme` / `localStorage`). Do not assume a universal theme API |
| `checks[].id` | Label echoed in the result |
| `checks[].selector` | CSS selector |
| `checks[].index` | Which match when several exist (default 0) |
| `checks[].styles` | Computed properties to return — **only these**, never the whole CSSOM |
| `checks[].screenshot` | Clip the element's box to a PNG |
| `checks[].hover` | Real mouse move to the element's center (CSS `:hover`), then re-read styles |
| `checks[].hoverScreenshot` | Clip again after hover |

Batch every selector and theme variant you need in **one** helper run. One Chrome, one navigation.

### Result

```json
{
  "ok": true,
  "browser": "/path/to/chrome",
  "url": "…",
  "results": [
    {
      "id": "see-also",
      "ok": true,
      "count": 3,
      "visible": true,
      "box": { "x": 12, "y": 40, "width": 80, "height": 18 },
      "styles": { "color": "rgb(0, 0, 238)" },
      "screenshot": "/tmp/aivo-browser-check/see-also.png",
      "hover": {
        "styles": { "color": "rgb(0, 0, 238)" },
        "screenshot": "/tmp/aivo-browser-check/see-also-hover.png"
      }
    }
  ]
}
```

`ok` at the top is false if Chrome/Node/navigation failed. A missing selector sets that check's `ok` false with `error` — other checks still run. Hover uses a real `mouseMoved` event, not a DOM `mouseover` dispatch.

## After the check

If styles already prove the bug, edit the source and re-run the **same** request (maybe a second theme). Do not debug the helper, rewrite CDP, or keep exploring adjacent files once you can make the change.
