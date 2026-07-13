import { useState, type ReactNode } from 'react';
import { Globe } from 'lucide-react';

/**
 * A Markdown link rendered with the destination site's favicon before the text
 * — the pattern modern chat UIs use to make external links scannable.
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

const ANCHOR_CLASS =
  'text-accent-600 underline underline-offset-2 hover:text-accent-500';

export function LinkWithFavicon({
  href,
  children,
}: {
  href?: string;
  children: ReactNode;
}) {
  const domain = externalDomain(href);
  const [iconFailed, setIconFailed] = useState(false);

  // No favicon for non-external links — render exactly the previous plain link.
  if (!domain) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={ANCHOR_CLASS}>
        {children}
      </a>
    );
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`group inline-flex max-w-full items-center gap-1 align-middle ${ANCHOR_CLASS}`}
    >
      {iconFailed ? (
        <Globe
          size={16}
          aria-hidden="true"
          className="shrink-0 text-zinc-500 transition-transform group-hover:scale-110"
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
          className="h-4 w-4 shrink-0 rounded-sm opacity-80 transition-all group-hover:scale-110 group-hover:opacity-100"
        />
      )}
      <span className="break-words">{children}</span>
    </a>
  );
}
