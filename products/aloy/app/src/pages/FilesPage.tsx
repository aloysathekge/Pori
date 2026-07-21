import { useEffect, useRef, useState } from 'react';
import { Download, FolderOpen, LoaderCircle, Trash2, Upload } from 'lucide-react';
import {
  listMyFiles,
  removeFromLibrary,
  uploadLibraryFile,
  type StoredFileView,
} from '@/api/files';
import { apiStreamFetch } from '@/api/client';
import { Button } from '@/components/ui/Button';
import { FileThumbnail } from '@/components/files/FileVisual';
import { Spinner } from '@/components/ui/Spinner';

function humanSize(bytes: number) {
  if (bytes > 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes > 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function fileOrigin(file: StoredFileView) {
  if (file.event_is_life) return 'Uploaded to My Files';
  if (file.event_title) return `From ${file.event_title}`;
  return 'Original Event unavailable';
}

async function download(file: StoredFileView) {
  try {
    const response = await apiStreamFetch(
      `/files/${file.file_id}`,
      undefined,
      undefined,
      'GET',
    );
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = file.name;
    anchor.click();
    URL.revokeObjectURL(url);
  } catch {
    // The retained row remains available for another attempt.
  }
}

export function FilesPage() {
  const [files, setFiles] = useState<StoredFileView[] | null>(null);
  const [uploading, setUploading] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listMyFiles()
      .then(setFiles)
      .catch(() => setFiles([]));
  }, []);

  async function handleRemove(file: StoredFileView) {
    try {
      await removeFromLibrary(file.file_id);
      setFiles((current) => (current ?? []).filter((item) => item.file_id !== file.file_id));
    } catch {
      // Keep the row when the request fails.
    }
  }

  async function handleUpload(selected: FileList | null) {
    const pending = Array.from(selected ?? []);
    if (pending.length === 0) return;
    setUploadError('');
    for (const file of pending) {
      setUploading(file.name);
      setUploadProgress(0);
      try {
        const stored = await uploadLibraryFile(file, setUploadProgress);
        setFiles((current) => [
          stored,
          ...(current ?? []).filter((item) => item.file_id !== stored.file_id),
        ]);
      } catch (cause) {
        setUploadError(cause instanceof Error ? cause.message : `Could not upload ${file.name}`);
        break;
      }
    }
    setUploading('');
    setUploadProgress(0);
  }

  return (
    <div className="h-full overflow-y-auto px-4 py-5 sm:p-6">
      <div className="mx-auto max-w-3xl">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <FolderOpen size={20} className="text-accent-600" />
            <h1 className="text-lg font-semibold text-zinc-100">My files</h1>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => {
              void handleUpload(event.target.files);
              event.target.value = '';
            }}
          />
          <Button onClick={() => fileInputRef.current?.click()} disabled={!!uploading}>
            {uploading ? <LoaderCircle size={16} className="animate-spin" /> : <Upload size={16} />}
            {uploading ? `${uploadProgress}%` : 'Upload files'}
          </Button>
        </div>
        <p className="mb-6 text-sm text-zinc-400">
          Upload once, then attach a file in Life with @ or the + menu. Event uploads stay with their Event.
        </p>

        {uploading && (
          <div className="mb-4 rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3">
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="min-w-0 truncate text-zinc-300">Uploading {uploading}</span>
              <span className="shrink-0 text-zinc-500">{uploadProgress}%</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full rounded-full bg-accent-600 transition-[width]"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}
        {uploadError && (
          <p className="mb-4 rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {uploadError}
          </p>
        )}

        {files === null ? (
          <div className="flex justify-center py-12">
            <Spinner className="h-8 w-8" />
          </div>
        ) : files.length === 0 ? (
          <div className="rounded-xl border border-dashed border-zinc-700 p-8 text-center text-sm text-zinc-500">
            Nothing saved yet. Upload a file here or save one from a conversation.
          </div>
        ) : (
          <ul className="divide-y divide-zinc-800 rounded-xl border border-zinc-800 bg-zinc-900/40">
            {files.map((file) => (
              <li key={file.file_id} className="flex items-center gap-3 px-3 py-3 sm:px-4">
                <FileThumbnail file={file} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-zinc-200">{file.name}</p>
                  <p className="mt-0.5 truncate text-xs font-medium text-accent-700">
                    {fileOrigin(file)}
                  </p>
                  <p className="mt-0.5 text-[11px] text-zinc-500">
                    {humanSize(file.size_bytes)} · saved {new Date(file.created_at).toLocaleDateString()}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void download(file)}
                  title="Download"
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 sm:h-9 sm:w-9"
                  aria-label={`Download ${file.name}`}
                >
                  <Download size={15} />
                </button>
                <button
                  type="button"
                  onClick={() => void handleRemove(file)}
                  title="Remove from My files"
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-red-400 sm:h-9 sm:w-9"
                  aria-label={`Remove ${file.name} from My files`}
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
