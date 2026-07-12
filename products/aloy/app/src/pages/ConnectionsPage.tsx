import { useCallback, useEffect, useState } from 'react';
import { Building2, Link2, Mail, Plug, Unplug, User } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import {
  disconnect,
  listConnections,
  startConnection,
  type ConnectionScope,
  type ProviderInfo,
} from '@/api/connections';
import { McpServersSection } from '@/components/McpServersSection';

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
    const params = new URLSearchParams(window.location.search);
    if (params.get('connected') && params.get('connected') !== 'error') {
      window.history.replaceState({}, '', '/connections');
    }
  }, [load]);

  async function connect(provider: string, scope: ConnectionScope) {
    setBusy(`${scope}:${provider}`);
    setError('');
    try {
      const { authorize_url } = await startConnection(provider, scope);
      window.location.assign(authorize_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start connection');
      setBusy(null);
    }
  }

  async function remove(provider: string, scope: ConnectionScope) {
    if (!window.confirm(`Disconnect ${provider}?`)) return;
    setBusy(`${scope}:${provider}`);
    setError('');
    try {
      await disconnect(provider, scope);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not disconnect');
    } finally {
      setBusy(null);
    }
  }

  const showOrg =
    providers?.some((p) => p.can_manage_org || p.org_connected) ?? false;

  return (
    <div className="mx-auto max-w-2xl p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-lg font-semibold text-zinc-100">Connections</h1>
        <p className="text-sm text-zinc-400">
          Connect accounts so Aloy can act on them. Tokens are encrypted and stay
          with Aloy — nothing is shared with third parties.
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
          <p className="font-medium text-zinc-400">
            Account integrations aren't set up on this server yet.
          </p>
          <p className="mt-1">
            Connecting accounts (Google — Gmail &amp; Calendar) requires the
            server operator to configure OAuth credentials. Once that's done,
            the connect buttons appear here. MCP servers below work
            regardless.
          </p>
        </div>
      ) : (
        <div className="space-y-8">
          <Section icon={User} title="My connections">
            {providers.map((p) => (
              <ProviderCard
                key={p.provider}
                provider={p}
                connected={p.connected}
                status={p.status}
                accountEmail={p.account_email}
                manageable
                busy={busy === `user:${p.provider}`}
                onConnect={() => connect(p.provider, 'user')}
                onDisconnect={() => remove(p.provider, 'user')}
              />
            ))}
          </Section>

          {showOrg && (
            <Section
              icon={Building2}
              title="Organization connections"
              subtitle="Shared with everyone in your organization."
            >
              {providers.map((p) => (
                <ProviderCard
                  key={p.provider}
                  provider={p}
                  connected={p.org_connected}
                  status={p.org_status}
                  accountEmail={p.org_account_email}
                  manageable={p.can_manage_org}
                  readOnlyLabel="Provided by your organization"
                  busy={busy === `org:${p.provider}`}
                  onConnect={() => connect(p.provider, 'org')}
                  onDisconnect={() => remove(p.provider, 'org')}
                />
              ))}
            </Section>
          )}
        </div>
      )}

      {/* MCP servers are independent of OAuth providers — always shown
          (previously nested in the providers branch, so a server with no
          OAuth configured hid the entire MCP UI). */}
      <div className="mt-8">
        <McpServersSection />
      </div>
    </div>
  );
}

function Section({
  icon: Icon,
  title,
  subtitle,
  children,
}: {
  icon: typeof User;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <Icon size={15} className="text-zinc-500" />
        <h2 className="text-sm font-medium text-zinc-300">{title}</h2>
      </div>
      {subtitle && <p className="-mt-2 mb-3 text-xs text-zinc-500">{subtitle}</p>}
      <div className="grid gap-3">{children}</div>
    </div>
  );
}

function ProviderCard({
  provider,
  connected,
  status,
  accountEmail,
  manageable,
  readOnlyLabel,
  busy,
  onConnect,
  onDisconnect,
}: {
  provider: ProviderInfo;
  connected: boolean;
  status: string | null;
  accountEmail: string | null;
  manageable: boolean;
  readOnlyLabel?: string;
  busy: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}) {
  const Icon = ICONS[provider.provider] ?? Link2;
  return (
    <div className="flex items-center gap-4 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-50 text-accent-600">
        <Icon size={20} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-zinc-100">{provider.label}</span>
          {connected && (
            <span className="rounded-full bg-accent-50 px-2 py-0.5 text-xs text-accent-700">
              connected
            </span>
          )}
          {status === 'error' && (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
              reconnect
            </span>
          )}
        </div>
        <p className="mt-0.5 truncate text-sm text-zinc-400">
          {connected && accountEmail ? accountEmail : provider.description}
        </p>
      </div>
      {!manageable ? (
        <span className="text-xs text-zinc-500">{readOnlyLabel}</span>
      ) : status === 'error' ? (
        <Button size="sm" disabled={busy} onClick={onConnect}>
          {busy ? <Spinner className="h-4 w-4" /> : <Plug size={14} />}
          Reconnect
        </Button>
      ) : connected ? (
        <Button variant="outline" size="sm" disabled={busy} onClick={onDisconnect}>
          <Unplug size={14} /> Disconnect
        </Button>
      ) : (
        <Button size="sm" disabled={busy} onClick={onConnect}>
          {busy ? <Spinner className="h-4 w-4" /> : <Plug size={14} />}
          Connect
        </Button>
      )}
    </div>
  );
}
