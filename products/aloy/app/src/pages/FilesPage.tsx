import { useEffect, useState } from 'react';
import { Download, FileText, FolderOpen, Trash2 } from 'lucide-react';
import {
  listMyFiles,
  removeFromLibrary,
  type StoredFileView,
} from '@/api/files';
import { apiStreamFetch } from '@/api/client';
import { Spinner } from '@/components/ui/Spinner';

function humanSize(n: number) {
  if (n > 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n > 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

async function download(f: StoredFileView) {
  try {
    const res = await apiStreamFetch(
      `/files/${f.file_id}`,
      undefined,
      undefined,
      'GET',
    );
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = f.name;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    // transient — the row stays for a retry
  }
}

export function FilesPage() {
  const [files, setFiles] = useState<StoredFileView[] | null>(null);

  useEffect(() => {
    listMyFiles()
      .then(setFiles)
      .catch(() => setFiles([]));
  }, []);

  async function handleRemove(f: StoredFileView) {
    try {
      await removeFromLibrary(f.file_id);
      setFiles((prev) => (prev ?? []).filter((x) => x.file_id !== f.file_id));
    } catch {
      // keep the row on failure
    }
  }

  return (
    <div className="h-full overflow-y-auto px-4 py-5 sm:p-6">
      <div className="mx-auto max-w-3xl">
      <div className="mb-1 flex items-center gap-2">
        <FolderOpen size={20} className="text-accent-600" />
        <h1 className="text-lg font-semibold text-zinc-100">My files</h1>
      </div>
      <p className="mb-6 text-sm text-zinc-400">
        Files saved here are remembered across every chat — say “use my CV”
        anywhere and the agent fetches it into its workspace.
      </p>

      {files === null ? (
        <div className="flex justify-center py-12">
          <Spinner className="h-8 w-8" />
        </div>
      ) : files.length === 0 ? (
        <div className="rounded-xl border border-dashed border-zinc-700 p-8 text-center text-sm text-zinc-500">
          Nothing saved yet. Attach a file in a chat, then use the bookmark on
          its chip to save it to your library.
        </div>
      ) : (
        <ul className="divide-y divide-zinc-800 rounded-xl border border-zinc-800 bg-zinc-900/40">
          {files.map((f) => (
            <li key={f.file_id} className="flex items-center gap-3 px-4 py-3">
              <FileText size={16} className="shrink-0 text-accent-600" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-zinc-200">
                  {f.name}
                </p>
                <p className="text-xs text-zinc-500">
                  {humanSize(f.size_bytes)} · saved{' '}
                  {new Date(f.created_at).toLocaleDateString()}
                </p>
              </div>
              <button
                type="button"
                onClick={() => download(f)}
                title="Download"
                className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
              >
                <Download size={15} />
              </button>
              <button
                type="button"
                onClick={() => handleRemove(f)}
                title="Remove from library (the agent forgets it exists)"
                className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-red-400"
              >
                <Trash2 size={15} />
              </button>
            </li>
          ))}
        </ul>
      )}
      </div>
    </div>
  );
}
