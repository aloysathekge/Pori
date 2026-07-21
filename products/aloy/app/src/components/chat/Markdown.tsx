import { memo } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { LinkWithFavicon } from './LinkWithFavicon';

/**
 * Renders assistant message text as GitHub-flavored Markdown — bold, italics,
 * lists, tables, links, and code blocks. Raw HTML is NOT rendered (react-markdown
 * default), so message content can never inject markup.
 */

const components: Components = {
  p: ({ children }) => <p className="mb-4 leading-7 last:mb-0">{children}</p>,
  strong: ({ children }) => (
    <strong className="font-semibold text-zinc-100">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,
  a: ({ children, href }) => <LinkWithFavicon href={href}>{children}</LinkWithFavicon>,
  ul: ({ children }) => (
    <ul className="mb-4 list-disc space-y-1.5 pl-5 last:mb-0">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-4 list-decimal space-y-1.5 pl-5 last:mb-0">{children}</ol>
  ),
  li: ({ children }) => <li className="leading-7">{children}</li>,
  h1: ({ children }) => (
    <h1 className="mb-3 mt-7 text-xl font-semibold text-zinc-100 first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-3 mt-7 text-lg font-semibold text-zinc-100 first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-2 mt-6 text-base font-semibold text-zinc-100 first:mt-0">{children}</h3>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mb-4 border-l-2 border-zinc-600 pl-4 italic leading-7 text-zinc-400 last:mb-0">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-6 border-zinc-800" />,
  code: ({ className, children, ...props }) => {
    // Inline code has no language class and no newline; block code is fenced.
    const isBlock = /language-/.test(className || '') || String(children).includes('\n');
    if (!isBlock) {
      return (
        <code
          className="rounded bg-zinc-950/60 px-1.5 py-0.5 font-mono text-[0.85em] text-accent-500"
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        className={`block font-mono text-[0.85em] leading-relaxed ${className || ''}`}
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="mb-3 overflow-x-auto rounded-lg border border-zinc-700 bg-zinc-950 p-3 last:mb-0">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="mb-3 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-zinc-700 bg-zinc-800 px-2 py-1 text-left font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-zinc-700 px-2 py-1">{children}</td>
  ),
};

export const Markdown = memo(function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {children}
    </ReactMarkdown>
  );
});
