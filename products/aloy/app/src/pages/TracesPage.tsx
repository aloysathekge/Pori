import { useState, useEffect, useCallback } from 'react';
import { Activity, ChevronRight, ChevronDown, Trash2, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import { listTraces, getTrace, deleteTrace } from '@/api/traces';
import type { TraceListItem, TraceDetail, TraceSpan } from '@/types';

function SpanTree({ span, depth = 0 }: { span: TraceSpan; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = span.children.length > 0;

  return (
    <div style={{ marginLeft: depth * 16 }}>
      <button
        onClick={() => hasChildren && setExpanded(!expanded)}
        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-zinc-800"
      >
        {hasChildren ? (
          expanded ? (
            <ChevronDown size={14} className="text-zinc-500" />
          ) : (
            <ChevronRight size={14} className="text-zinc-500" />
          )
        ) : (
          <span className="w-3.5" />
        )}
        <Badge
          color={
            span.status === 'ok'
              ? 'green'
              : span.status === 'error'
                ? 'red'
                : 'gray'
          }
        >
          {span.type || 'span'}
        </Badge>
        <span className="text-zinc-300">{span.name}</span>
        <span className="ml-auto text-xs text-zinc-500">{span.duration}</span>
      </button>
      {expanded &&
        span.children.map((child) => (
          <SpanTree key={child.span_id} span={child} depth={depth + 1} />
        ))}
    </div>
  );
}

function groupTraces(traces: TraceListItem[]): [string, TraceListItem[]][] {
  const groups = new Map<string, TraceListItem[]>();
  for (const t of traces) {
    const key = t.conversation_id ?? 'none';
    const list = groups.get(key) ?? [];
    list.push(t);
    groups.set(key, list);
  }
  return Array.from(groups.entries());
}

export function TracesPage() {
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setTraces(await listTraces());
    } catch {
      // handle silently
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial load on mount
    load();
  }, [load]);

  async function openTrace(id: string) {
    setDetailLoading(true);
    try {
      setDetail(await getTrace(id));
    } catch {
      // handle silently
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Delete this trace?')) return;
    try {
      await deleteTrace(id);
      if (detail?.id === id) setDetail(null);
      await load();
    } catch {
      // handle silently
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (detail) {
    return (
      <div className="h-full overflow-y-auto px-4 py-5 sm:p-6 lg:p-8">
        <Button variant="ghost" size="sm" onClick={() => setDetail(null)} className="mb-4">
          <ArrowLeft size={14} /> Back to traces
        </Button>

        {detailLoading ? (
          <div className="flex justify-center py-12">
            <Spinner className="h-8 w-8" />
          </div>
        ) : (
          <Card className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-zinc-100">
                {detail.trace_data.name}
              </h2>
              <Badge
                color={detail.status === 'ok' ? 'green' : 'red'}
              >
                {detail.status}
              </Badge>
            </div>

            <div className="grid gap-4 text-sm sm:grid-cols-3">
              <div>
                <p className="text-zinc-500">Duration</p>
                <p className="text-zinc-200">{detail.trace_data.duration}</p>
              </div>
              <div>
                <p className="text-zinc-500">Spans</p>
                <p className="text-zinc-200">{detail.total_spans}</p>
              </div>
              <div>
                <p className="text-zinc-500">Created</p>
                <p className="text-zinc-200">
                  {new Date(detail.created_at).toLocaleString()}
                </p>
              </div>
            </div>

            {detail.trace_data.input && (
              <div>
                <p className="mb-1 text-xs font-medium text-zinc-500">Input</p>
                <p className="rounded-lg bg-zinc-800 p-3 text-sm text-zinc-300">
                  {detail.trace_data.input}
                </p>
              </div>
            )}

            {detail.trace_data.output && (
              <div>
                <p className="mb-1 text-xs font-medium text-zinc-500">Output</p>
                <p className="rounded-lg bg-zinc-800 p-3 text-sm text-zinc-300">
                  {detail.trace_data.output}
                </p>
              </div>
            )}

            <div>
              <p className="mb-2 text-xs font-medium text-zinc-500">
                Span Tree
              </p>
              <div className="rounded-lg border border-zinc-800 p-2">
                {detail.trace_data.tree.map((span) => (
                  <SpanTree key={span.span_id} span={span} />
                ))}
              </div>
            </div>
          </Card>
        )}
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-4 py-5 sm:p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-zinc-100">Traces</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Agent execution traces for debugging
        </p>
      </div>

      {traces.length === 0 ? (
        <Card className="py-12 text-center text-zinc-500">
          <Activity size={32} className="mx-auto mb-3 text-zinc-600" />
          <p>No traces yet. They appear automatically when agents run.</p>
        </Card>
      ) : (
        <div className="space-y-6">
          {groupTraces(traces).map(([convId, group]) => (
            <div key={convId}>
              <div className="mb-2 flex items-center gap-2">
                <h2 className="truncate text-sm font-medium text-zinc-300">
                  {group[0]?.conversation_title || 'Untitled conversation'}
                </h2>
                <span className="shrink-0 text-xs text-zinc-600">
                  {group.length} run{group.length === 1 ? '' : 's'}
                </span>
                {group[0]?.conversation_id && (
                  <a
                    href={`/chat/${group[0].conversation_id}`}
                    className="shrink-0 text-xs text-accent-600 hover:underline"
                  >
                    open chat →
                  </a>
                )}
              </div>
              <div className="space-y-2">
                {group.map((trace) => (
                  <Card
                    key={trace.id}
                    className="flex cursor-pointer items-center gap-4 transition-colors hover:border-zinc-700"
                  >
                    <button
                      className="flex flex-1 items-center gap-4"
                      onClick={() => openTrace(trace.id)}
                    >
                      <Badge color={trace.status === 'ok' ? 'green' : 'red'}>
                        {trace.status}
                      </Badge>
                      <div className="flex-1 text-left">
                        <p className="text-sm text-zinc-300">
                          {trace.total_spans} spans &middot;{' '}
                          {trace.duration_seconds.toFixed(2)}s
                        </p>
                        <p className="text-xs text-zinc-500">
                          {trace.id.slice(0, 12)}…
                        </p>
                      </div>
                      <span className="text-xs text-zinc-500">
                        {new Date(trace.created_at).toLocaleString(undefined, {
                          month: 'short',
                          day: 'numeric',
                          hour: 'numeric',
                          minute: '2-digit',
                        })}
                      </span>
                    </button>
                    <button onClick={() => handleDelete(trace.id)}>
                      <Trash2
                        size={14}
                        className="text-zinc-500 hover:text-red-600"
                      />
                    </button>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
