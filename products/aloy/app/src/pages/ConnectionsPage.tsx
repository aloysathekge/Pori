import { useCallback, useEffect, useState } from 'react';
import { Link2, Mail, Plug, Unplug } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import {
  disconnect,
  listConnections,
  startConnection,
  type ProviderInfo,
} from '@/api/connections';

const ICONS: Record<string, typeof Mail> = { google: Mail };

export function ConnectionsPage() {
  const [providers, setProviders] = useState<ProviderInfo[] | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setProviders(await listConnections());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load connections');
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
    // Returning from the OAuth redirect lands on /connections?connected=… —
    // refresh so the just-connected account shows immediately.
    const params = new URLSearchParams(window.location.search);
    if (params.get('connected') && params.get('connected') !== 'error') {
      window.history.replaceState({}, '', '/connections');
    }
  }, [load]);

  async function connect(provider: string) {
    setBusy(provider);
    setError('');
    try {
      const { authorize_url } = await startConnection(provider);
      window.location.assign(authorize_url); // full-page redirect to consent
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start connection');
      setBusy(null);
    }
  }

  async function remove(provider: string) {
    if (!window.confirm(`Disconnect ${provider}?`)) return;
    setBusy(provider);
    setError('');
    try {
      await disconnect(provider);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not disconnect');
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="mx-auto max-w-2xl p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-lg font-semibold text-zinc-100">Connections</h1>
        <p className="text-sm text-zinc-400">
          Connect your accounts so Aloy can act on them. Your tokens are
          encrypted and stay with Aloy — nothing is shared with third parties.
        </p>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {!providers ? (
        <div className="flex justify-center py-12">
          <Spinner className="h-8 w-8" />
        </div>
      ) : providers.length === 0 ? (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6 text-center text-sm text-zinc-500">
          <Plug size={20} className="mx-auto mb-2 text-zinc-600" />
          No integrations are configured on this server yet.
        </div>
      ) : (
        <div className="grid gap-3">
          {providers.map((p) => {
            const Icon = ICONS[p.provider] ?? Link2;
            const isBusy = busy === p.provider;
            return (
              <div
                key={p.provider}
                className="flex items-center gap-4 rounded-xl border border-zinc-800 bg-zinc-900 p-4"
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-50 text-accent-600">
                  <Icon size={20} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-zinc-100">{p.label}</span>
                    {p.connected && (
                      <span className="rounded-full bg-accent-50 px-2 py-0.5 text-xs text-accent-700">
                        connected
                      </span>
                    )}
                    {p.status === 'error' && (
                      <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
                        reconnect
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 truncate text-sm text-zinc-400">
                    {p.connected && p.account_email
                      ? p.account_email
                      : p.description}
                  </p>
                </div>
                {p.connected ? (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={isBusy}
                    onClick={() => remove(p.provider)}
                  >
                    <Unplug size={14} /> Disconnect
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    disabled={isBusy}
                    onClick={() => connect(p.provider)}
                  >
                    {isBusy ? <Spinner className="h-4 w-4" /> : <Plug size={14} />}
                    Connect
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
