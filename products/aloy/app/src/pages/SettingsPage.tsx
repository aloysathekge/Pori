import {
  Bell,
  CalendarClock,
  Check,
  ChevronRight,
  CircleUserRound,
  Download,
  Eye,
  Link2,
  LockKeyhole,
  MessageCircle,
  Monitor,
  Paintbrush,
  Save,
  ShieldCheck,
  SlidersHorizontal,
} from 'lucide-react';
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { useNavigate } from 'react-router-dom';
import {
  createKnowledgeEntry,
  exportMemory,
  listKnowledgeEntries,
} from '@/api/memory';
import { getProfile, updateProfile } from '@/api/profile';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Modal } from '@/components/ui/Modal';
import { Spinner } from '@/components/ui/Spinner';
import { TextArea } from '@/components/ui/TextArea';
import { useAuth } from '@/contexts/useAuth';
import { useTheme } from '@/hooks/useTheme';
import type { UserProfileResponse } from '@/types';

const WORKING_AGREEMENT_KEY = 'aloy.global.working-agreement';
const DEFAULT_AGREEMENT =
  'Aloy should help me stay organized while asking before consequential actions.';

type Dialog = 'working-agreement' | 'privacy' | 'appearance' | 'account' | null;
type AttentionPreference = 'show_today_suggestions';

function prettify(value: string) {
  return value
    .replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function initialFor(name: string) {
  return name.trim().charAt(0).toUpperCase() || 'A';
}

function Toggle({
  checked,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`inline-flex h-9 min-w-[88px] shrink-0 items-center justify-between gap-2 rounded-full border px-2.5 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-wait disabled:opacity-60 ${
        checked
          ? 'border-accent-500/70 bg-accent-500/15 text-accent-200'
          : 'border-zinc-700 bg-zinc-800 text-zinc-400'
      }`}
    >
      <span>{checked ? 'On' : 'Off'}</span>
      <span
        aria-hidden="true"
        className={`flex h-5 w-9 items-center rounded-full border p-0.5 transition-colors ${
          checked
            ? 'justify-end border-accent-500 bg-accent-600'
            : 'justify-start border-zinc-600 bg-zinc-700'
        }`}
      >
        <span className="h-3.5 w-3.5 rounded-full bg-white shadow-sm" />
      </span>
    </button>
  );
}

function Section({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border border-zinc-800 bg-zinc-900/70 ${className}`}
    >
      {children}
    </section>
  );
}

export function SettingsPage() {
  const navigate = useNavigate();
  const { session, signOut } = useAuth();
  const { theme, setTheme } = useTheme();
  const [profile, setProfile] = useState<UserProfileResponse | null>(null);
  const [workingAgreement, setWorkingAgreement] = useState(DEFAULT_AGREEMENT);
  const [draftName, setDraftName] = useState('');
  const [draftAgreement, setDraftAgreement] = useState(DEFAULT_AGREEMENT);
  const [dialog, setDialog] = useState<Dialog>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savingPreference, setSavingPreference] =
    useState<AttentionPreference | null>(null);
  const [exporting, setExporting] = useState(false);
  const [avatarFailed, setAvatarFailed] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    const [profileResult, knowledgeResult] = await Promise.allSettled([
      getProfile(),
      listKnowledgeEntries(24),
    ]);

    if (profileResult.status === 'fulfilled') {
      const nextProfile = profileResult.value;
      setProfile(nextProfile);
      setDraftName(nextProfile.display_name || '');
    }
    const profileAgreement =
      profileResult.status === 'fulfilled' &&
      typeof profileResult.value.preferences?.aloy_working_agreement ===
        'string'
        ? profileResult.value.preferences.aloy_working_agreement
        : '';
    const acceptedAgreement =
      knowledgeResult.status === 'fulfilled'
        ? knowledgeResult.value.find(
            (entry) => entry.conflict_key === WORKING_AGREEMENT_KEY,
          )?.content
        : '';
    const resolvedAgreement =
      acceptedAgreement || profileAgreement || DEFAULT_AGREEMENT;
    setWorkingAgreement(resolvedAgreement);
    setDraftAgreement(resolvedAgreement);

    if (
      profileResult.status === 'rejected' &&
      knowledgeResult.status === 'rejected'
    ) {
      setError('Aloy could not load your controls. Try refreshing this page.');
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    // The async load owns the initial remote state for this route.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  const displayName = useMemo(() => {
    const metadataName =
      typeof session?.user?.user_metadata?.full_name === 'string'
        ? session.user.user_metadata.full_name
        : '';
    return (
      profile?.display_name ||
      metadataName ||
      session?.user?.email?.split('@')[0] ||
      'You'
    );
  }, [profile?.display_name, session]);

  const preferences = profile?.preferences ?? {};
  const attention = (key: AttentionPreference, fallback = true) =>
    typeof preferences[key] === 'boolean'
      ? (preferences[key] as boolean)
      : fallback;

  function openWorkingAgreement() {
    setDraftName(profile?.display_name || displayName);
    setDraftAgreement(workingAgreement);
    setError('');
    setDialog('working-agreement');
  }

  async function saveWorkingAgreement() {
    const agreement = draftAgreement.trim();
    if (!agreement) {
      setError('Describe how you want Aloy to work with you.');
      return;
    }
    setSaving(true);
    setError('');
    try {
      const nextPreferences = {
        ...preferences,
        aloy_working_agreement: agreement,
      };
      const jobs: [
        ReturnType<typeof updateProfile>,
        ReturnType<typeof createKnowledgeEntry> | Promise<null>,
      ] = [
        updateProfile({
          display_name: draftName.trim() || undefined,
          preferences: nextPreferences,
        }),
        agreement !== workingAgreement
          ? createKnowledgeEntry({
              content: agreement,
              tags: ['working-agreement', 'preference'],
              importance: 5,
              kind: 'procedural',
              source: 'user',
              sensitivity: 'internal',
              conflict_key: WORKING_AGREEMENT_KEY,
              conflict_policy: 'supersede',
            })
          : Promise.resolve(null),
      ];
      const [updatedProfile] = await Promise.all(jobs);
      setProfile(updatedProfile);
      setWorkingAgreement(agreement);
      setDialog(null);
      setNotice('Your Aloy preferences were saved.');
      window.setTimeout(() => setNotice(''), 2400);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Aloy could not save your preferences.',
      );
    } finally {
      setSaving(false);
    }
  }

  async function saveAttentionPreference(
    key: AttentionPreference,
    next: boolean,
  ) {
    setSavingPreference(key);
    setError('');
    try {
      const updated = await updateProfile({
        preferences: { ...preferences, [key]: next },
      });
      setProfile(updated);
      setNotice('Attention preference updated.');
      window.setTimeout(() => setNotice(''), 1800);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Aloy could not update this preference.',
      );
    } finally {
      setSavingPreference(null);
    }
  }

  async function handleExport() {
    setExporting(true);
    setError('');
    try {
      const data = await exportMemory();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `aloy-memory-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'Memory export failed.',
      );
    } finally {
      setExporting(false);
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-full items-center justify-center py-24">
        <Spinner className="h-7 w-7" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto w-full max-w-6xl px-5 py-7 sm:px-7 lg:px-10 lg:py-10">
      <header className="mb-7">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-zinc-100">
          Aloy &amp; you
        </h1>
        <p className="mt-1.5 text-sm text-zinc-400">
          Control how Aloy works with you and what matters most.
        </p>
      </header>

      {(notice || error) && (
        <div
          role="status"
          className={`mb-5 flex items-center gap-2 rounded-xl border px-4 py-3 text-sm ${
            error
              ? 'border-red-500/30 bg-red-500/10 text-red-300'
              : 'border-accent-500/25 bg-accent-500/10 text-accent-300'
          }`}
        >
          {error ? <Bell size={16} /> : <Check size={16} />}
          {error || notice}
        </div>
      )}

      <Section className="mb-7 p-5 sm:p-7">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
          {profile?.avatar_url && !avatarFailed ? (
            <img
              src={profile.avatar_url}
              alt=""
              onError={() => setAvatarFailed(true)}
              className="h-20 w-20 shrink-0 rounded-full border border-zinc-700 object-cover sm:h-24 sm:w-24"
            />
          ) : (
            <div
              aria-hidden="true"
              className="flex h-20 w-20 shrink-0 items-center justify-center rounded-full border border-accent-500/20 bg-accent-500/10 font-display text-2xl font-semibold text-accent-300 sm:h-24 sm:w-24"
            >
              {initialFor(displayName)}
            </div>
          )}

          <div className="min-w-0 flex-1">
            <h2 className="font-display text-2xl font-semibold text-zinc-100">
              {displayName}
            </h2>
            <p className="mt-2 max-w-xl text-sm leading-6 text-zinc-300">
              {workingAgreement}
            </p>
          </div>

          <Button
            variant="outline"
            onClick={openWorkingAgreement}
            className="shrink-0 self-start sm:self-center"
          >
            <SlidersHorizontal size={16} />
            Edit how Aloy works with me
          </Button>
        </div>
      </Section>

      <Section className="p-5 sm:p-7">
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">
              When Aloy needs me
            </h2>
            <p className="mt-1 text-sm text-zinc-400">
              Choose how Aloy brings important changes to you.
            </p>
          </div>

          <div className="mt-4 divide-y divide-zinc-800 border-t border-zinc-800">
            <div className="flex items-start gap-3 py-4">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500/10 text-accent-300">
                <Bell size={17} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-zinc-200">
                  Approvals always notify
                </p>
                <p className="mt-1 text-xs leading-5 text-zinc-500">
                  Protected actions never continue silently.
                </p>
              </div>
              <span className="mt-1 rounded-full border border-accent-500/25 bg-accent-500/10 px-2.5 py-1 text-[11px] font-medium text-accent-300">
                Always on
              </span>
            </div>

            <div className="flex items-start gap-3 py-4">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500/10 text-accent-300">
                <CalendarClock size={17} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-zinc-200">
                  Schedule notifications
                </p>
                <p className="mt-1 text-xs leading-5 text-zinc-500">
                  Choose important-only or every outcome per Schedule.
                </p>
              </div>
              <button
                type="button"
                onClick={() => navigate('/schedules')}
                className="mt-1 flex items-center gap-1 text-xs font-medium text-accent-300 hover:text-accent-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
              >
                Per schedule
                <ChevronRight size={14} />
              </button>
            </div>

            <div className="flex items-start gap-3 py-4">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500/10 text-accent-300">
                <MessageCircle size={17} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-zinc-200">
                  Suggestions in Today
                </p>
                <p className="mt-1 text-xs leading-5 text-zinc-500">
                  Show useful follow-ups in your daily focus view.
                </p>
              </div>
              <Toggle
                checked={attention('show_today_suggestions')}
                disabled={savingPreference === 'show_today_suggestions'}
                label="Show suggestions in Today"
                onChange={(next) =>
                  void saveAttentionPreference('show_today_suggestions', next)
                }
              />
            </div>
          </div>
      </Section>

      <Section className="mt-7 overflow-hidden">
        <div className="border-b border-zinc-800 px-5 py-4 sm:px-7">
          <h2 className="text-base font-semibold text-zinc-100">
            What Aloy may do
          </h2>
          <p className="mt-1 text-xs text-zinc-500">
            Your safety boundary stays clear even when individual Events have
            their own controls.
          </p>
        </div>
        <div className="grid divide-y divide-zinc-800 md:grid-cols-2 md:divide-y-0">
          <button
            type="button"
            onClick={() => navigate('/schedules')}
            className="flex items-center gap-3 px-5 py-4 text-left hover:bg-zinc-800/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500 sm:px-7 md:border-r md:border-zinc-800"
          >
            <CalendarClock size={18} className="text-zinc-500" />
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-medium text-zinc-200">
                Scheduled work
              </span>
              <span className="block text-xs text-zinc-500">
                Authority is chosen per Event Schedule
              </span>
            </span>
            <span className="text-xs font-medium text-accent-300">
              Review
            </span>
            <ChevronRight size={16} className="text-zinc-600" />
          </button>
          <button
            type="button"
            onClick={() => navigate('/connections')}
            className="flex items-center gap-3 px-5 py-4 text-left hover:bg-zinc-800/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500 sm:px-7"
          >
            <LockKeyhole size={18} className="text-zinc-500" />
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-medium text-zinc-200">
                Actions outside Aloy
              </span>
              <span className="block text-xs text-zinc-500">
                Consequential actions always require approval
              </span>
            </span>
            <span className="text-xs font-medium text-accent-300">
              Always ask
            </span>
          </button>
        </div>
      </Section>

      <nav
        aria-label="More settings"
        className="mt-7 grid overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/70 sm:grid-cols-2 lg:grid-cols-5"
      >
        {[
          {
            label: 'Privacy & data',
            icon: ShieldCheck,
            action: () => setDialog('privacy'),
          },
          {
            label: 'Connected services',
            icon: Link2,
            action: () => navigate('/connections'),
          },
          {
            label: 'Appearance',
            icon: Paintbrush,
            action: () => setDialog('appearance'),
          },
          {
            label: 'Account',
            icon: CircleUserRound,
            action: () => setDialog('account'),
          },
          {
            label: 'Sign out',
            icon: LockKeyhole,
            action: () => void signOut(),
          },
        ].map((item, index) => {
          const Icon = item.icon;
          return (
            <button
              type="button"
              key={item.label}
              onClick={item.action}
              className={`flex min-h-14 items-center gap-2.5 border-b border-zinc-800 px-4 text-left text-sm text-zinc-300 transition-colors hover:bg-zinc-800/55 hover:text-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500 lg:border-b-0 ${
                index % 2 === 0 && index < 4 ? 'sm:border-r' : ''
              } ${index < 4 ? 'lg:border-r' : 'lg:border-r-0'}`}
            >
              <Icon size={17} className="text-zinc-500" />
              <span className="min-w-0 flex-1 truncate">{item.label}</span>
              <ChevronRight size={15} className="text-zinc-600" />
            </button>
          );
        })}
      </nav>

      <p className="mt-6 flex items-start gap-2 text-xs leading-5 text-zinc-500">
        <ShieldCheck size={15} className="mt-0.5 shrink-0" />
        Event-specific memory, schedules, connections, and authority are
        controlled inside each Event.
      </p>

      <Modal
        open={dialog === 'working-agreement'}
        onClose={() => setDialog(null)}
        title="How Aloy works with you"
        panelClassName="max-w-xl"
      >
        <div className="space-y-4">
          <Input
            label="Your name"
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
            placeholder="What should Aloy call you?"
          />
          <TextArea
            label="Working agreement"
            value={draftAgreement}
            onChange={(event) => setDraftAgreement(event.target.value)}
            rows={5}
            maxLength={500}
            placeholder="Describe how you want Aloy to help, communicate, and pause for you."
          />
          <p className="text-xs leading-5 text-zinc-500">
            This becomes accepted global memory. You can review, correct, or
            forget it from Memory at any time.
          </p>
          {error && <p className="text-sm text-red-400">{error}</p>}
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="ghost" onClick={() => setDialog(null)}>
              Cancel
            </Button>
            <Button onClick={() => void saveWorkingAgreement()} disabled={saving}>
              {saving ? <Spinner className="h-4 w-4" /> : <Save size={16} />}
              Save preferences
            </Button>
          </div>
        </div>
      </Modal>

      <Modal
        open={dialog === 'privacy'}
        onClose={() => setDialog(null)}
        title="Privacy & data"
      >
        <div className="space-y-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-4">
            <p className="text-sm font-medium text-zinc-200">
              Your memory stays scoped
            </p>
            <p className="mt-1 text-xs leading-5 text-zinc-500">
              Global memory belongs to you. Event memory stays inside its Event
              unless you explicitly promote it.
            </p>
          </div>
          <Button
            variant="secondary"
            className="w-full"
            onClick={() => navigate('/memory')}
          >
            <Eye size={16} />
            Review or forget memory
          </Button>
          <Button
            variant="outline"
            className="w-full"
            onClick={() => void handleExport()}
            disabled={exporting}
          >
            {exporting ? (
              <Spinner className="h-4 w-4" />
            ) : (
              <Download size={16} />
            )}
            Export my memory
          </Button>
        </div>
      </Modal>

      <Modal
        open={dialog === 'appearance'}
        onClose={() => setDialog(null)}
        title="Appearance"
      >
        <div className="space-y-3">
          <p className="text-sm text-zinc-400">
            Choose how Aloy looks on this device.
          </p>
          <div className="grid grid-cols-2 gap-3">
            {(['light', 'dark'] as const).map((choice) => (
              <button
                type="button"
                key={choice}
                onClick={() => setTheme(choice)}
                className={`flex items-center gap-3 rounded-xl border p-4 text-left transition-colors ${
                  theme === choice
                    ? 'border-accent-500 bg-accent-500/10 text-accent-200'
                    : 'border-zinc-800 bg-zinc-950/40 text-zinc-300 hover:border-zinc-700'
                }`}
              >
                <Monitor size={18} />
                <span className="text-sm font-medium">{prettify(choice)}</span>
                {theme === choice && <Check size={16} className="ml-auto" />}
              </button>
            ))}
          </div>
        </div>
      </Modal>

      <Modal
        open={dialog === 'account'}
        onClose={() => setDialog(null)}
        title="Account"
      >
        <dl className="divide-y divide-zinc-800">
          <div className="py-3">
            <dt className="text-xs text-zinc-500">Signed in as</dt>
            <dd className="mt-1 text-sm text-zinc-200">
              {session?.user?.email || 'Unknown'}
            </dd>
          </div>
          <div className="py-3">
            <dt className="text-xs text-zinc-500">Member since</dt>
            <dd className="mt-1 text-sm text-zinc-200">
              {profile?.created_at
                ? new Date(profile.created_at).toLocaleDateString()
                : 'Unavailable'}
            </dd>
          </div>
        </dl>
        <Button
          variant="secondary"
          className="mt-5 w-full"
          onClick={() => void signOut()}
        >
          Sign out
        </Button>
      </Modal>
      </div>
    </div>
  );
}
