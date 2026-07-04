# website — Aloy landing page

The public marketing site for **Aloy** — a single **self-contained static page**
(`index.html`): all CSS/JS inline, system fonts, no build step, no dependencies.
Open it in a browser, or drop it on any static host (Vercel / Netlify / GitHub
Pages / Cloudflare Pages).

## Preview

```bash
# just open it
open website/index.html          # or double-click

# or serve it
python -m http.server -d website 4321   # http://localhost:4321
```

## Design

Calm, modern-SaaS identity: warm off-white ground, soft neutrals, a single teal
accent (`#0F8571`), crisp system-font type, airy spacing. The hero graphic renders
the moat directly — org → team → personal knowledge, most-specific wins (personal
highlighted in the accent). Sections: hero · four pillars · the Pori kernel ·
capabilities · CTA · footer. Real product copy; deliberately not the AI-default
blue-purple-gradient look.

Not a product surface and not in the Python import graph. (Supersedes the earlier
plan to import `../pori_website`, which was the Pori marketing site.)
