import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import {
  Download,
  Ellipsis,
  FolderMinus,
  FolderPlus,
  MessageSquareText,
  PanelRightOpen,
  Trash2,
} from 'lucide-react';
import {
  deleteStoredFile,
  getStoredFileBlob,
  removeFromLibrary,
  saveToLibrary,
} from '@/api/files';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { useToast } from '@/contexts/toast';

interface FileActionsMenuProps {
  file: {
    id: string;
    name: string;
    size_bytes: number;
    kind?: string;
    in_library?: boolean;
  };
  onOpen?: () => void;
  onAskAloy?: (reference: StoredFileReference) => void;
  onLibraryChanged?: (inLibrary: boolean) => void;
  onDeleted?: (fileId: string) => void;
}

const MENU_WIDTH = 240;
const MENU_HEIGHT_ESTIMATE = 230;

export function FileActionsMenu({
  file,
  onOpen,
  onAskAloy,
  onLibraryChanged,
  onDeleted,
}: FileActionsMenuProps) {
  const { showToast } = useToast();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ left: 8, top: 8 });
  const [libraryOverride, setLibraryOverride] = useState<{
    fileId: string;
    value: boolean;
  } | null>(null);
  const [busyAction, setBusyAction] = useState<'download' | 'library' | 'delete' | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [error, setError] = useState('');
  const inLibrary = libraryOverride?.fileId === file.id
    ? libraryOverride.value
    : Boolean(file.in_library);

  useEffect(() => {
    if (!open) return;

    function closeOnOutsidePointer(event: PointerEvent) {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target) || menuRef.current?.contains(target)) return;
      setOpen(false);
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== 'Escape') return;
      setOpen(false);
      triggerRef.current?.focus();
    }

    function closeOnViewportChange() {
      setOpen(false);
    }

    document.addEventListener('pointerdown', closeOnOutsidePointer);
    document.addEventListener('keydown', closeOnEscape);
    window.addEventListener('resize', closeOnViewportChange);
    window.addEventListener('scroll', closeOnViewportChange, true);
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePointer);
      document.removeEventListener('keydown', closeOnEscape);
      window.removeEventListener('resize', closeOnViewportChange);
      window.removeEventListener('scroll', closeOnViewportChange, true);
    };
  }, [open]);

  function openMenu() {
    const bounds = triggerRef.current?.getBoundingClientRect();
    if (!bounds) return;
    const left = Math.max(8, Math.min(window.innerWidth - MENU_WIDTH - 8, bounds.right - MENU_WIDTH));
    const below = bounds.bottom + 6;
    const top = below + MENU_HEIGHT_ESTIMATE <= window.innerHeight
      ? below
      : Math.max(8, bounds.top - MENU_HEIGHT_ESTIMATE - 6);
    setPosition({ left, top });
    setError('');
    setConfirmingDelete(false);
    setOpen(true);
  }

  function runAndClose(action: () => void) {
    action();
    setOpen(false);
  }

  async function download() {
    setBusyAction('download');
    setError('');
    try {
      const blob = await getStoredFileBlob(file.id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = file.name;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      setOpen(false);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Could not download file';
      setError(message);
      showToast({ title: 'Download failed', description: message, tone: 'error' });
    } finally {
      setBusyAction(null);
    }
  }

  async function toggleLibrary() {
    setBusyAction('library');
    setError('');
    try {
      const updated = inLibrary
        ? await removeFromLibrary(file.id)
        : await saveToLibrary(file.id);
      setLibraryOverride({ fileId: file.id, value: updated.in_library });
      onLibraryChanged?.(updated.in_library);
      setOpen(false);
      showToast({
        title: updated.in_library ? 'Saved to My Files' : 'Removed from My Files',
        description: file.name,
        tone: 'success',
      });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Could not update My Files';
      setError(message);
      showToast({ title: 'Could not update My Files', description: message, tone: 'error' });
    } finally {
      setBusyAction(null);
    }
  }

  async function deleteFile() {
    setBusyAction('delete');
    setError('');
    try {
      await deleteStoredFile(file.id);
      setOpen(false);
      showToast({ title: 'File deleted', description: file.name, tone: 'success' });
      onDeleted?.(file.id);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Could not delete file';
      setError(message);
      showToast({ title: 'Could not delete file', description: message, tone: 'error' });
    } finally {
      setBusyAction(null);
    }
  }

  const menu = open ? (
    <div
      ref={menuRef}
      role="menu"
      aria-label={`Actions for ${file.name}`}
      className="fixed z-[100] w-60 rounded-2xl border border-zinc-700 bg-zinc-900 p-1.5 shadow-2xl"
      style={position}
    >
      {onOpen && (
        <MenuButton icon={<PanelRightOpen size={16} />} onClick={() => runAndClose(onOpen)}>
          Open in Workbench
        </MenuButton>
      )}
      {onAskAloy && (
        <MenuButton
          icon={<MessageSquareText size={16} />}
          onClick={() => runAndClose(() => onAskAloy({ file_id: file.id, name: file.name, size: file.size_bytes }))}
        >
          Ask Aloy about this
        </MenuButton>
      )}
      <MenuButton icon={<Download size={16} />} onClick={() => void download()} disabled={busyAction !== null}>
        {busyAction === 'download' ? 'Downloading…' : 'Download'}
      </MenuButton>
      <div className="my-1 border-t border-zinc-800" />
      <MenuButton
        icon={inLibrary ? <FolderMinus size={16} /> : <FolderPlus size={16} />}
        onClick={() => void toggleLibrary()}
        disabled={busyAction !== null}
      >
        {busyAction === 'library'
          ? 'Updating…'
          : inLibrary
            ? 'Remove from My Files'
            : 'Save to My Files'}
      </MenuButton>
      {file.kind === 'upload' && (
        <>
          <div className="my-1 border-t border-zinc-800" />
          {confirmingDelete ? (
            <div className="rounded-xl bg-red-500/10 p-3">
              <p className="text-xs leading-5 text-red-300">
                Delete this file from the Event permanently?
              </p>
              <div className="mt-2 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setConfirmingDelete(false)}
                  disabled={busyAction !== null}
                  className="rounded-lg px-2.5 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => void deleteFile()}
                  disabled={busyAction !== null}
                  className="rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-400 disabled:opacity-50"
                >
                  {busyAction === 'delete' ? 'Deleting…' : 'Delete'}
                </button>
              </div>
            </div>
          ) : (
            <MenuButton
              icon={<Trash2 size={16} />}
              onClick={() => setConfirmingDelete(true)}
              disabled={busyAction !== null}
              danger
            >
              Delete from Event…
            </MenuButton>
          )}
        </>
      )}
      {error && <p className="px-3 py-2 text-xs leading-5 text-red-500">{error}</p>}
    </div>
  ) : null;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          if (open) setOpen(false);
          else openMenu();
        }}
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60"
        aria-label={`More actions for ${file.name}`}
        aria-haspopup="menu"
        aria-expanded={open}
        title="File actions"
      >
        <Ellipsis size={18} />
      </button>
      {menu && createPortal(menu, document.body)}
    </>
  );
}

function MenuButton({
  icon,
  children,
  onClick,
  disabled = false,
  danger = false,
}: {
  icon: ReactNode;
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      className={`flex min-h-11 w-full items-center gap-3 rounded-xl px-3 text-left text-sm transition-colors hover:bg-zinc-800 focus-visible:bg-zinc-800 focus-visible:outline-none disabled:opacity-50 ${danger ? 'text-red-400' : 'text-zinc-200'}`}
    >
      <span className={danger ? 'text-red-500' : 'text-zinc-500'}>{icon}</span>
      <span>{children}</span>
    </button>
  );
}
