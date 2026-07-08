import { useCallback, useEffect, useState } from 'react';
import { Building2, Plus, Server, Trash2, User } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import {
  createMcpServer,
  deleteMcpServer,
  listMcpServers,
  setMcpServerEnabled,
  type McpServerInfo,
} from '@/api/mcpServers';
import type { ConnectionScope } from '@/api/connections';

export function McpServersSection() {
  const [servers, setServers] = useState<McpServerInfo[] | null>(null);
  const [error, setError] = useState('');
  const [adding, setAdding] = useState(false);

  const load = useCallback(async () => {
    try {
      setServers(await listMcpServers());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load MCP servers');
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  const canManageOrg = servers?.some((s) => s.scope === 'org' && s.account_managed);
  const mine = servers?.filter((s) => s.scope === 'user') ?? [];
  const org = servers?.filter((s) => s.scope === 'org') ?? [];

  async function toggle(s: McpServerInfo) {
    setServers((prev) =>
      prev?.map((x) => (x.id === s.id ? { ...x, enabled: !x.enabled } : x)) ?? prev,
    );
    try {
      await setMcpServerEnabled(s.id, !s.enabled);
    } catch {
      await load();
    }
  }

  async function remove(s: McpServerInfo) {
    if (!window.confirm(`Delete MCP server "${s.name}"?`)) return;
    try {
      await deleteMcpServer(s.id);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete');
    }
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server size={15} className="text-zinc-500" />
          <h2 className="text-sm font-medium text-zinc-300">MCP servers</h2>
        </div>
        <Button size="sm" variant="outline" onClick={() => setAdding((v) => !v)}>
          <Plus size={14} /> Add server
        </Button>
      </div>
      <p className="-mt-1 mb-3 text-xs text-zinc-500">
        Connect external tool servers (Model Context Protocol). The agent gains
        their tools for your runs.
      </p>

      {error && (
        <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {adding && (
        <AddServerForm
          canManageOrg={!!canManageOrg}
          onAdded={() => {
            setAdding(false);
            load();
          }}
          onError={setError}
        />
      )}

      {!servers ? (
        <div className="flex justify-center py-6">
          <Spinner className="h-6 w-6" />
        </div>
      ) : servers.length === 0 ? (
        <p className="rounded-lg border border-dashed border-zinc-800 px-4 py-6 text-center text-sm text-zinc-500">
          No MCP servers yet.
        </p>
      ) : (
        <div className="space-y-4">
          <ServerList icon={User} label="Mine" servers={mine} onToggle={toggle} onRemove={remove} />
          {org.length > 0 && (
            <ServerList
              icon={Building2}
              label="Organization"
              servers={org}
              onToggle={toggle}
              onRemove={remove}
            />
          )}
        </div>
      )}
    </div>
  );
}

function ServerList({
  icon: Icon,
  label,
  servers,
  onToggle,
  onRemove,
}: {
  icon: typeof User;
  label: string;
  servers: McpServerInfo[];
  onToggle: (s: McpServerInfo) => void;
  onRemove: (s: McpServerInfo) => void;
}) {
  if (servers.length === 0) return null;
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-500">
        <Icon size={12} /> {label}
      </div>
      <div className="grid gap-2">
        {servers.map((s) => (
          <div
            key={s.id}
            className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-3"
          >
            <div className="min-w-0 flex-1">
              <div className="font-medium text-zinc-100">{s.name}</div>
              <div className="truncate text-xs text-zinc-500">{s.url}</div>
            </div>
            {s.account_managed ? (
              <>
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-zinc-400">
                  <input
                    type="checkbox"
                    checked={s.enabled}
                    onChange={() => onToggle(s)}
                  />
                  enabled
                </label>
                <button
                  onClick={() => onRemove(s)}
                  className="text-zinc-500 hover:text-red-500"
                  aria-label="Delete server"
                >
                  <Trash2 size={15} />
                </button>
              </>
            ) : (
              <span className="text-xs text-zinc-500">
                {s.enabled ? 'enabled' : 'disabled'}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function AddServerForm({
  canManageOrg,
  onAdded,
  onError,
}: {
  canManageOrg: boolean;
  onAdded: () => void;
  onError: (msg: string) => void;
}) {
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [secret, setSecret] = useState('');
  const [scope, setScope] = useState<ConnectionScope>('user');
  const [saving, setSaving] = useState(false);

  async function submit() {
    if (!name.trim() || !url.trim()) return;
    setSaving(true);
    try {
      await createMcpServer({
        name: name.trim(),
        url: url.trim(),
        scope,
        auth_kind: secret.trim() ? 'static' : 'none',
        static_secret: secret.trim() || undefined,
      });
      onAdded();
    } catch (e) {
      onError(e instanceof Error ? e.message : 'Could not add server');
    } finally {
      setSaving(false);
    }
  }

  const input =
    'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600';

  return (
    <div className="mb-3 grid gap-2 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <input
        className={input}
        placeholder="Name (e.g. notion)"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <input
        className={input}
        placeholder="Server URL (https://…/mcp)"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <input
        className={input}
        placeholder="Bearer token (optional)"
        type="password"
        value={secret}
        onChange={(e) => setSecret(e.target.value)}
      />
      {canManageOrg && (
        <label className="flex items-center gap-2 text-sm text-zinc-400">
          <input
            type="checkbox"
            checked={scope === 'org'}
            onChange={(e) => setScope(e.target.checked ? 'org' : 'user')}
          />
          Share with the whole organization
        </label>
      )}
      <div className="flex justify-end">
        <Button size="sm" disabled={saving} onClick={submit}>
          {saving ? <Spinner className="h-4 w-4" /> : null} Add
        </Button>
      </div>
    </div>
  );
}
