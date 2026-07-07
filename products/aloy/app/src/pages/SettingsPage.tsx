import { useCallback, useState, useEffect } from 'react';
import { Save, ShieldCheck, ShieldAlert, User } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Spinner } from '@/components/ui/Spinner';
import { Badge } from '@/components/ui/Badge';
import { getProfile, updateProfile } from '@/api/profile';
import { getExecutionStatus, type ExecutionStatus } from '@/api/system';
import type { UserProfileResponse } from '@/types';
import { useAuth } from '@/contexts/AuthContext';
import { SoulEditor } from '@/components/settings/SoulEditor';

export function SettingsPage() {
  const { session } = useAuth();
  const [profile, setProfile] = useState<UserProfileResponse | null>(null);
  const [execution, setExecution] = useState<ExecutionStatus | null>(null);
  const [displayName, setDisplayName] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    try {
      const p = await getProfile();
      setProfile(p);
      setDisplayName(p.display_name || '');
      setAvatarUrl(p.avatar_url || '');
    } catch {
      // handle silently
    }
    try {
      setExecution(await getExecutionStatus());
    } catch {
      // execution status is best-effort; hide the card if it fails
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  async function handleSave() {
    setSaving(true);
    setSaved(false);
    try {
      const updated = await updateProfile({
        display_name: displayName || undefined,
        avatar_url: avatarUrl || undefined,
      });
      setProfile(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      // handle silently
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl p-6 lg:p-8">
      <h1 className="mb-6 text-xl font-bold text-zinc-100">Settings</h1>

      <Card className="mb-6 space-y-4">
        <div className="flex items-center gap-4">
          {avatarUrl ? (
            <img
              src={avatarUrl}
              alt="Avatar"
              className="h-14 w-14 rounded-full border border-zinc-700 object-cover"
            />
          ) : (
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-zinc-800">
              <User size={24} className="text-zinc-500" />
            </div>
          )}
          <div>
            <p className="font-medium text-zinc-200">
              {displayName || session?.user?.email || 'User'}
            </p>
            <p className="text-sm text-zinc-500">{session?.user?.email}</p>
          </div>
        </div>

        <Input
          label="Display Name"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="Your name"
        />
        <Input
          label="Avatar URL"
          value={avatarUrl}
          onChange={(e) => setAvatarUrl(e.target.value)}
          placeholder="https://..."
        />

        <div className="flex items-center gap-3">
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <Spinner className="h-4 w-4" />
            ) : (
              <Save size={16} />
            )}
            Save
          </Button>
          {saved && (
            <span className="text-sm text-emerald-400">Saved!</span>
          )}
        </div>
      </Card>

      {execution && (
        <Card className="mb-6 space-y-3">
          <h2 className="text-sm font-semibold text-zinc-300">
            Secure execution
          </h2>
          <div className="flex items-start gap-3">
            <span
              className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                execution.isolated
                  ? 'bg-accent-50 text-accent-600'
                  : 'bg-zinc-800 text-zinc-500'
              }`}
            >
              {execution.isolated ? (
                <ShieldCheck size={18} />
              ) : (
                <ShieldAlert size={18} />
              )}
            </span>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-zinc-200">
                  {execution.label}
                </span>
                {execution.isolated && (
                  <Badge color="accent">isolated</Badge>
                )}
              </div>
              <p className="mt-1 text-sm text-zinc-400">{execution.detail}</p>
            </div>
          </div>
          <p className="text-xs text-zinc-500">
            Managed by Aloy — nothing to configure here.
          </p>
        </Card>
      )}

      <div className="mb-6">
        <SoulEditor />
      </div>

      {profile && (
        <Card className="space-y-3">
          <h2 className="text-sm font-semibold text-zinc-300">Account</h2>
          <div>
            <p className="text-xs text-zinc-500">Member Since</p>
            <p className="text-sm text-zinc-200">
              {new Date(profile.created_at).toLocaleDateString()}
            </p>
          </div>
          <div className="pt-2">
            <Badge color="gray">ID: {profile.id.slice(0, 12)}...</Badge>
          </div>
        </Card>
      )}
    </div>
  );
}
