import { useState, useEffect, useCallback } from 'react';
import { BarChart3 } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Select } from '@/components/ui/Select';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { getUsageSummary, getUsageHistory, getUsageRecords } from '@/api/usage';
import type { UsageSummary, UsageHistoryEntry, UsageRecord } from '@/types';

export function UsagePage() {
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [history, setHistory] = useState<UsageHistoryEntry[]>([]);
  const [records, setRecords] = useState<UsageRecord[]>([]);
  const [days, setDays] = useState(30);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    // allSettled: one failing endpoint must not blank the whole page (a 500
    // on /history once hid perfectly good summary + records data).
    const [s, h, r] = await Promise.allSettled([
      getUsageSummary(days),
      getUsageHistory(days),
      getUsageRecords(),
    ]);
    if (s.status === 'fulfilled') setSummary(s.value);
    if (h.status === 'fulfilled') setHistory(h.value);
    if (r.status === 'fulfilled') setRecords(r.value);
    setLoading(false);
  }, [days]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load on mount/days change
    load();
  }, [load]);

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-4 py-5 sm:p-6 lg:p-8">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">Usage & Billing</h1>
          <p className="mt-1 text-sm text-zinc-400">
            Token usage and cost tracking
          </p>
        </div>
        <Select
          options={[
            { value: '7', label: 'Last 7 days' },
            { value: '30', label: 'Last 30 days' },
            { value: '90', label: 'Last 90 days' },
            { value: '365', label: 'Last year' },
          ]}
          value={String(days)}
          onChange={(e) => setDays(parseInt(e.target.value))}
        />
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="mb-8 grid gap-4 sm:grid-cols-3">
          <Card>
            <p className="text-sm text-zinc-400">Total Tokens</p>
            <p className="mt-1 text-2xl font-bold text-zinc-100">
              {(summary.total_tokens ?? 0).toLocaleString()}
            </p>
          </Card>
          <Card>
            <p className="text-sm text-zinc-400">Total Cost</p>
            <p className="mt-1 text-2xl font-bold text-zinc-100">
              ${(summary.total_cost ?? 0).toFixed(4)}
            </p>
          </Card>
          <Card>
            <p className="text-sm text-zinc-400">Total Requests</p>
            <p className="mt-1 text-2xl font-bold text-zinc-100">
              {summary.total_requests}
            </p>
          </Card>
        </div>
      )}

      {/* By Model */}
      {summary && Object.keys(summary.by_model).length > 0 && (
        <Card className="mb-8">
          <h2 className="mb-4 text-sm font-semibold text-zinc-300">
            Usage by Model
          </h2>
          <div className="space-y-2">
            {Object.entries(summary.by_model).map(([model, data]) => (
              <div
                key={model}
                className="flex items-center justify-between rounded-lg bg-zinc-800/50 px-4 py-2"
              >
                <span className="text-sm text-zinc-300">{model}</span>
                <div className="flex gap-4">
                  <Badge color="blue">
                    {data.tokens.toLocaleString()} tokens
                  </Badge>
                  <Badge color="green">${data.cost.toFixed(4)}</Badge>
                  <Badge color="gray">{data.requests} requests</Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* History Chart */}
      {history.length > 0 && (
        <Card className="mb-8">
          <h2 className="mb-4 text-sm font-semibold text-zinc-300">
            Daily Usage
          </h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E9E5DE" />
                <XAxis dataKey="date" stroke="#838A90" fontSize={12} />
                <YAxis stroke="#838A90" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E9E5DE',
                    borderRadius: 8,
                    color: '#191C1E',
                  }}
                />
                <Legend />
                <Bar
                  dataKey="tokens"
                  fill="#0F8571"
                  radius={[4, 4, 0, 0]}
                  name="Tokens"
                />
                <Bar
                  dataKey="requests"
                  fill="#3AA88D"
                  radius={[4, 4, 0, 0]}
                  name="Requests"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Recent Records */}
      <Card>
        <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-zinc-300">
          <BarChart3 size={16} /> Recent Records
        </h2>
        {records.length === 0 ? (
          <p className="py-4 text-center text-sm text-zinc-500">No records</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-xs text-zinc-500">
                  <th className="pb-2 pr-4">Model</th>
                  <th className="pb-2 pr-4">Input</th>
                  <th className="pb-2 pr-4">Output</th>
                  <th className="pb-2 pr-4">Total</th>
                  <th className="pb-2 pr-4">Cost</th>
                  <th className="pb-2">Date</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r) => (
                  <tr
                    key={r.id}
                    className="border-b border-zinc-800/50 text-zinc-300"
                  >
                    <td className="py-2 pr-4 text-xs">
                      {r.provider}/{r.model}
                    </td>
                    <td className="py-2 pr-4">
                      {r.input_tokens.toLocaleString()}
                    </td>
                    <td className="py-2 pr-4">
                      {r.output_tokens.toLocaleString()}
                    </td>
                    <td className="py-2 pr-4">
                      {r.total_tokens.toLocaleString()}
                    </td>
                    <td className="py-2 pr-4">
                      ${r.estimated_cost.toFixed(4)}
                    </td>
                    <td className="py-2 text-xs text-zinc-500">
                      {new Date(r.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
