import { useEffect, useState } from 'react';
import { ImageIcon } from 'lucide-react';
import { getEventCoverBlob, type EventSummary } from '@/api/events';

export function EventCover({ event, className = '' }: { event: EventSummary; className?: string }) {
  const [source, setSource] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    let objectUrl: string | null = null;
    if (event.cover?.status !== 'ready') {
      return;
    }
    getEventCoverBlob(event.id)
      .then((blob) => {
        if (disposed) return;
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      })
      .catch(() => setSource(null));
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [event.cover?.status, event.id, event.updated_at]);

  if (event.cover?.status === 'ready' && source) {
    return <img src={source} alt={event.cover.alt_text || `${event.title} cover`} className={`object-cover ${className}`} />;
  }
  return (
    <span
      title={event.cover?.status === 'queued' ? 'Aloy will create this cover in the background' : undefined}
      className={`flex items-center justify-center bg-gradient-to-br from-accent-600/15 via-zinc-800 to-zinc-900 text-accent-600 ${event.cover?.status === 'queued' ? 'animate-pulse' : ''} ${className}`}
    >
      <ImageIcon size={16} />
    </span>
  );
}
