import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import { GripHorizontal, X } from 'lucide-react';
import { AloyMark } from '@/components/icons';

interface FloatingAloyPanelProps {
  open: boolean;
  status: string;
  storageKey: string;
  onOpen: () => void;
  onClose: () => void;
  children: ReactNode;
}

interface Position {
  x: number;
  y: number;
}

const EDGE_GAP = 12;

function storedPosition(key: string): Position | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(key) || 'null') as Partial<Position> | null;
    if (!value || !Number.isFinite(value.x) || !Number.isFinite(value.y)) return null;
    return { x: Number(value.x), y: Number(value.y) };
  } catch {
    return null;
  }
}

function clampPosition(position: Position, element: HTMLElement, parent: HTMLElement): Position {
  return {
    x: Math.min(Math.max(EDGE_GAP, position.x), Math.max(EDGE_GAP, parent.clientWidth - element.offsetWidth - EDGE_GAP)),
    y: Math.min(Math.max(EDGE_GAP, position.y), Math.max(EDGE_GAP, parent.clientHeight - element.offsetHeight - EDGE_GAP)),
  };
}

function samePosition(left: Position, right: Position) {
  return left.x === right.x && left.y === right.y;
}

export function FloatingAloyPanel({
  open,
  status,
  storageKey,
  onOpen,
  onClose,
  children,
}: FloatingAloyPanelProps) {
  const buttonStorageKey = `${storageKey}:button`;
  const panelStorageKey = `${storageKey}:panel`;
  const [buttonPosition, setButtonPosition] = useState<Position | null>(() => storedPosition(buttonStorageKey));
  const [panelPosition, setPanelPosition] = useState<Position | null>(() => storedPosition(panelStorageKey));
  const [desktopPanel, setDesktopPanel] = useState(() => window.matchMedia('(min-width: 640px)').matches);
  const rootRef = useRef<HTMLDivElement | HTMLElement | null>(null);
  const buttonPositionRef = useRef(buttonPosition);
  const suppressOpenRef = useRef(false);

  useEffect(() => {
    const media = window.matchMedia('(min-width: 640px)');
    const update = () => setDesktopPanel(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    buttonPositionRef.current = buttonPosition;
  }, [buttonPosition]);

  useLayoutEffect(() => {
    const element = rootRef.current;
    const parent = element?.parentElement;
    if (!element || !parent) return;

    const constrain = () => {
      if (open) {
        if (!desktopPanel) return;
        setPanelPosition((current) => {
          if (!current) {
            const anchor = buttonPositionRef.current;
            const nearLeft = anchor ? anchor.x < parent.clientWidth / 2 : false;
            const nearTop = anchor ? anchor.y < parent.clientHeight / 2 : false;
            return clampPosition({
              x: nearLeft ? EDGE_GAP : parent.clientWidth - element.offsetWidth - EDGE_GAP,
              y: nearTop ? EDGE_GAP : parent.clientHeight - element.offsetHeight - EDGE_GAP,
            }, element, parent);
          }
          const next = clampPosition(current, element, parent);
          return samePosition(current, next) ? current : next;
        });
      } else {
        setButtonPosition((current) => {
          if (!current) return current;
          const next = clampPosition(current, element, parent);
          return samePosition(current, next) ? current : next;
        });
      }
    };

    constrain();
    const observer = new ResizeObserver(constrain);
    observer.observe(parent);
    window.addEventListener('resize', constrain);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', constrain);
    };
  }, [desktopPanel, open]);

  function beginDrag(
    event: ReactPointerEvent<HTMLElement>,
    kind: 'button' | 'panel',
  ) {
    if (kind === 'panel' && !desktopPanel) return;
    if (event.button !== 0) return;
    const element = rootRef.current;
    const parent = element?.parentElement;
    if (!element || !parent) return;
    event.preventDefault();

    const elementBounds = element.getBoundingClientRect();
    const parentBounds = parent.getBoundingClientRect();
    const start = {
      x: elementBounds.left - parentBounds.left,
      y: elementBounds.top - parentBounds.top,
    };
    const pointer = { x: event.clientX, y: event.clientY };
    let latest = start;
    let moved = false;

    const move = (pointerEvent: PointerEvent) => {
      const deltaX = pointerEvent.clientX - pointer.x;
      const deltaY = pointerEvent.clientY - pointer.y;
      if (Math.abs(deltaX) + Math.abs(deltaY) > 4) moved = true;
      const next = clampPosition(
        { x: start.x + deltaX, y: start.y + deltaY },
        element,
        parent,
      );
      latest = next;
      if (kind === 'button') setButtonPosition(next);
      else setPanelPosition(next);
    };

    const stop = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
      if (moved) {
        window.localStorage.setItem(
          kind === 'button' ? buttonStorageKey : panelStorageKey,
          JSON.stringify(latest),
        );
        if (kind === 'button') suppressOpenRef.current = true;
      }
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop, { once: true });
    window.addEventListener('pointercancel', stop, { once: true });
  }

  function openPanel() {
    if (suppressOpenRef.current) {
      suppressOpenRef.current = false;
      return;
    }
    onOpen();
  }

  if (!open) {
    return (
      <div
        ref={rootRef as React.RefObject<HTMLDivElement>}
        className="pointer-events-none absolute z-40"
        style={buttonPosition
          ? { left: buttonPosition.x, top: buttonPosition.y }
          : { bottom: 'max(0.75rem, env(safe-area-inset-bottom))', right: '1rem' }}
      >
        <button
          type="button"
          onPointerDown={(event) => beginDrag(event, 'button')}
          onClick={openPanel}
          className="pointer-events-auto flex min-h-12 touch-none select-none items-center gap-2.5 rounded-full border border-zinc-700 bg-zinc-900/95 py-2 pl-2.5 pr-4 text-sm font-semibold text-zinc-100 shadow-2xl backdrop-blur transition hover:border-accent-600/50 hover:bg-zinc-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
          aria-label={`Open Aloy conversation. Drag to move. ${status}`}
          aria-expanded="false"
          title="Drag to move · Click to open Aloy"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-600 text-white">
            <AloyMark size={18} />
          </span>
          <span>{status}</span>
        </button>
      </div>
    );
  }

  return (
    <section
      ref={rootRef as React.RefObject<HTMLElement>}
      className="absolute inset-x-2 bottom-[max(0.5rem,env(safe-area-inset-bottom))] z-40 flex h-[min(78dvh,42rem)] min-h-0 flex-col overflow-hidden rounded-3xl border border-zinc-700 bg-zinc-950/98 shadow-2xl backdrop-blur sm:inset-x-auto sm:right-3 sm:h-[min(42rem,calc(100%-1.5rem))] sm:w-[min(28rem,calc(100%-1.5rem))]"
      style={desktopPanel && panelPosition
        ? { bottom: 'auto', left: panelPosition.x, right: 'auto', top: panelPosition.y }
        : undefined}
      aria-label="Aloy conversation over the Workbench"
    >
      <header className="flex h-14 min-h-14 shrink-0 items-center gap-3 border-b border-zinc-800 px-3">
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-600 text-white">
          <AloyMark size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-zinc-100">Aloy</p>
          <p className="truncate text-[10px] text-zinc-500" role="status" aria-live="polite">
            {status}
          </p>
        </div>
        <button
          type="button"
          onPointerDown={(event) => beginDrag(event, 'panel')}
          className="hidden h-10 w-10 touch-none cursor-move items-center justify-center rounded-xl text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 sm:flex"
          aria-label="Drag Aloy conversation"
          title="Drag to move"
        >
          <GripHorizontal size={17} />
        </button>
        <button
          type="button"
          onClick={onClose}
          className="flex h-10 w-10 items-center justify-center rounded-xl text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
          aria-label="Collapse Aloy conversation"
        >
          <X size={17} />
        </button>
      </header>
      {children}
    </section>
  );
}
