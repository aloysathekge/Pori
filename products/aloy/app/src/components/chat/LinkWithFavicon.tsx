import { useState, type ReactNode } from 'react';
import { Globe } from 'lucide-react';

/**
 * A Markdown link rendered with the destination site's favicon before the text
 * — the pattern ChatGPT/modern chat UIs use to make external links scannable.
 *
 * The favicon is an INLINE image (not a flex box) followed by the link text, so
 * a long link wraps across lines naturally — the favicon stays before the first
 * word and the text flows to the next line, exactly like ChatGPT.
 *
 * Only absolute http(s) links get a favicon; anything else (relative links,
 * anchors, mailto:, invalid URLs) falls back to a plain anchor. If the favicon
 * fails to load, we swap in a neutral globe icon so the row never looks broken.
 */

const FAVICON_SIZE = 64; // request a crisp source; we render it at 16px (h-4 w-4)

// Favicon URLs are a pure function of the domain — memoize so repeated links to
// the same site don't recompute the string. (The image bytes themselves are
// cached by the browser's HTTP cache.)
const faviconCache = new Map<string, string>();

function faviconUrl(domain: string): string {
  const cached = faviconCache.get(domain);
  if (cached) return cached;
  const url = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(
    domain,
  )}&sz=${FAVICON_SIZE}`;
  faviconCache.set(domain, url);
  return url;
}

/** The hostname for an absolute http(s) URL, or null for anything we shouldn't
 *  fetch a favicon for (relative, anchor, mailto:, tel:, invalid). */
function externalDomain(href: string | undefined): string | null {
  if (!href) return null;
  try {
    const url = new URL(href); // no base → relative URLs throw, which we want
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    return url.hostname || null;
  } catch {
    return null;
  }
}

// ChatGPT-style: coloured text, no underline until hover; the favicon carries
// the "this is a link" signal alongside the colour.
const ANCHOR_CLASS =
  'text-accent-600 no-underline underline-offset-2 hover:underline hover:text-accent-500';

// Inline favicon aligned to sit on the text baseline like an emoji would.
const ICON_CLASS = 'mr-1 inline-block h-4 w-4 shrink-0 rounded-sm align-[-0.15em]';

export function LinkWithFavicon({
  href,
  children,
}: {
  href?: string;
  children: ReactNode;
}) {
  const domain = externalDomain(href);
  const [iconFailed, setIconFailed] = useState(false);

  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={ANCHOR_CLASS}>
      {domain &&
        (iconFailed ? (
          <Globe
            size={16}
            aria-hidden="true"
            className={`${ICON_CLASS} text-zinc-500`}
          />
        ) : (
          <img
            src={faviconUrl(domain)}
            alt="" // decorative: the link text carries the meaning
            aria-hidden="true"
            width={16}
            height={16}
            loading="lazy"
            onError={() => setIconFailed(true)}
            className={ICON_CLASS}
          />
        ))}
      {children}
    </a>
  );
}
