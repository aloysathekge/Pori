import { useEffect, useRef, useState } from 'react';
import { Archive, CalendarClock, ImagePlus } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { updateEvent, uploadEventCover, type EventSummary } from '@/api/events';
import { listSchedules } from '@/api/schedules';
import { MemoryIcon } from '@/components/icons';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/contexts/toast';
import { EventCover } from './EventCover';
import { EventMemoryPanel } from './EventMemoryPanel';

interface EventSettingsPanelProps {
  event: EventSummary;
  refreshKey?: string;
  onEventChanged: () => Promise<void>;
}

type SettingsSection = 'general' | 'memory';

export function EventSettingsPanel({
  event,
  refreshKey,
  onEventChanged,
}: EventSettingsPanelProps) {
  const navigate = useNavigate();
  const { showToast } = useToast();
  const [section, setSection] = useState<SettingsSection>('general');
  const [scheduleCount, setScheduleCount] = useState<number | null>(null);
  const [uploadingCover, setUploadingCover] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [phase, setPhase] = useState(event.phase);
  const [savingPhase, setSavingPhase] = useState(false);
  const [archiveOpen, setArchiveOpen] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const coverInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    void listSchedules()
      .then((schedules) => {
        if (active) setScheduleCount(schedules.filter((schedule) => schedule.event_id === event.id).length);
      })
      .catch(() => {
        if (active) setScheduleCount(null);
      });
    return () => { active = false; };
  }, [event.id, refreshKey]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- synchronize the saved Event phase returned by the host
    setPhase(event.phase);
  }, [event.phase]);

  async function changeCover(file: File | undefined) {
    if (!file) return;
    setUploadingCover(true);
    setUploadProgress(0);
    setError('');
    try {
      await uploadEventCover(event.id, file, setUploadProgress);
      await onEventChanged();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The cover could not be uploaded.');
    } finally {
      setUploadingCover(false);
      if (coverInputRef.current) coverInputRef.current.value = '';
    }
  }

  async function savePhase() {
    if (phase.trim() === event.phase) return;
    setSavingPhase(true);
    setError('');
    try {
      await updateEvent(event.id, { phase: phase.trim() });
      await onEventChanged();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The Event phase could not be updated.');
    } finally {
      setSavingPhase(false);
    }
  }

  async function archiveEvent() {
    setArchiving(true);
    setError('');
    try {
      await updateEvent(event.id, { lifecycle: 'archived' });
      showToast({
        tone: 'success',
        title: `${event.title} archived`,
        description: 'Its work is paused. Restore or permanently delete it from Archived Events.',
      });
      navigate('/today', { replace: true });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The Event could not be archived.');
      setArchiveOpen(false);
    } finally {
      setArchiving(false);
    }
  }

  return (
    <div>
      <div className="mb-5 grid grid-cols-2 rounded-lg border border-zinc-800 bg-zinc-950 p-0.5">
        <button
          type="button"
          onClick={() => setSection('general')}
          className={`rounded-md px-3 py-2 text-xs font-medium transition-colors ${section === 'general' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
          aria-pressed={section === 'general'}
        >
          General
        </button>
        <button
          type="button"
          onClick={() => setSection('memory')}
          className={`flex items-center justify-center gap-1.5 rounded-md px-3 py-2 text-xs font-medium transition-colors ${section === 'memory' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
          aria-pressed={section === 'memory'}
        >
          <MemoryIcon size={14} /> Memory
        </button>
      </div>

      {section === 'general' ? (
        <div className="space-y-4">
          {error && (
            <div role="alert" className="rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-xs leading-5 text-red-500">
              {error}
            </div>
          )}

          <section className="rounded-xl border border-zinc-800 bg-zinc-950/45 p-3.5">
            <div className="flex items-start gap-3">
              <EventCover event={event} className="h-14 w-20 shrink-0 rounded-lg border border-zinc-800" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-zinc-200">{event.title}</p>
                <p className="mt-1 line-clamp-2 text-xs leading-5 text-zinc-500">
                  {event.summary || 'No Event summary yet.'}
                </p>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] capitalize text-zinc-400">{event.lifecycle}</span>
                  {event.phase && <span className="rounded-full bg-accent-600/10 px-2 py-0.5 text-[10px] text-accent-700">{event.phase}</span>}
                </div>
              </div>
            </div>
          </section>

          {!event.is_life && (
            <section className="rounded-xl border border-zinc-800 p-3.5">
              <label htmlFor={`event-phase-${event.id}`} className="text-sm font-medium text-zinc-200">
                Current phase
              </label>
              <p className="mt-1 text-xs leading-5 text-zinc-500">
                A meaningful phase change can prompt Aloy to suggest adapting the Surface.
              </p>
              <div className="mt-3 flex items-center gap-2">
                <input
                  id={`event-phase-${event.id}`}
                  value={phase}
                  onChange={(inputEvent) => setPhase(inputEvent.target.value)}
                  placeholder="For example: Exam preparation"
                  maxLength={200}
                  className="min-h-11 min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-accent-600 focus:outline-none"
                />
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={savingPhase || phase.trim() === event.phase}
                  onClick={() => void savePhase()}
                >
                  {savingPhase ? 'Saving…' : 'Save'}
                </Button>
              </div>
            </section>
          )}

          {!event.is_life && (
            <section className="rounded-xl border border-zinc-800 p-3.5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-zinc-200">Event cover</p>
                  <p className="mt-1 text-xs leading-5 text-zinc-500">Give this Event a recognizable visual identity.</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => coverInputRef.current?.click()}
                  disabled={uploadingCover}
                >
                  {uploadingCover ? <Spinner className="h-3.5 w-3.5" /> : <ImagePlus size={14} />}
                  {uploadingCover ? `${uploadProgress}%` : 'Change'}
                </Button>
                <input
                  ref={coverInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  className="hidden"
                  onChange={(inputEvent) => void changeCover(inputEvent.target.files?.[0])}
                />
              </div>
            </section>
          )}

          <section className="rounded-xl border border-zinc-800 p-3.5">
            <div className="flex items-start gap-3">
              <CalendarClock size={17} className="mt-0.5 shrink-0 text-zinc-500" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-medium text-zinc-200">Schedules and autonomy</p>
                  <span className="text-[11px] text-zinc-600">
                    {scheduleCount === null ? '—' : scheduleCount}
                  </span>
                </div>
                <p className="mt-1 text-xs leading-5 text-zinc-500">
                  Control the durable reasons Aloy may wake and work on this Event.
                </p>
                <Link to="/schedules" className="mt-2 inline-block text-xs font-medium text-accent-700 hover:text-accent-600">
                  Manage schedules
                </Link>
              </div>
            </div>
          </section>

          {!event.is_life && (
            <section className="rounded-xl border border-zinc-800 p-3.5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-zinc-200">Archive Event</p>
                  <p className="mt-1 text-xs leading-5 text-zinc-500">
                    Pause this workspace and remove it from your sidebar without losing its history.
                  </p>
                </div>
                <Button size="sm" variant="outline" onClick={() => setArchiveOpen(true)}>
                  <Archive size={14} /> Archive
                </Button>
              </div>
            </section>
          )}
        </div>
      ) : (
        <EventMemoryPanel eventId={event.id} refreshKey={refreshKey} />
      )}

      <Modal
        open={archiveOpen}
        onClose={() => { if (!archiving) setArchiveOpen(false); }}
        title={`Archive ${event.title}?`}
      >
        <p className="text-sm leading-6 text-zinc-400">
          Aloy will stop waking for this Event, and it will disappear from Today and your normal sidebar. Its conversation, tasks, files, memory, Trail, and Surface remain intact.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setArchiveOpen(false)} disabled={archiving}>Cancel</Button>
          <Button variant="secondary" onClick={() => void archiveEvent()} disabled={archiving}>
            {archiving ? <Spinner className="h-4 w-4" /> : <Archive size={16} />}
            Archive Event
          </Button>
        </div>
      </Modal>
    </div>
  );
}
