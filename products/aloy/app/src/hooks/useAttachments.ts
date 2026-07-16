import { useState } from 'react';
import { uploadConversationFile } from '@/api/conversations';
import type { MessageImage, PendingFile } from '@/types';

export interface StoredFileReference {
  file_id: string;
  name: string;
  size: number;
}

const MAX_IMAGES = 3;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5MB per image (backend-enforced too)
const MAX_FILES = 3;
const MAX_FILE_CHARS = 200_000; // ~200KB of text (backend-enforced too)
const DOC_MIMES: Record<string, string> = {
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};
const MAX_DOC_BYTES = 10 * 1024 * 1024; // 10MB per document
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // durable-upload rung cap (backend too)
const TEXT_EXTENSIONS =
  /\.(txt|md|markdown|csv|tsv|json|jsonl|js|jsx|ts|tsx|py|rb|go|rs|java|c|h|cpp|cs|php|html|css|scss|xml|yml|yaml|toml|ini|cfg|conf|env|sh|bash|ps1|sql|log|diff|patch)$/i;

/**
 * Owns the composer's pending attachments and the whole attachment ladder
 * (inline text → native doc → durable upload). Page-agnostic: takes the
 * target conversation id as a parameter (durable uploads need it) and never
 * touches routing.
 */
export function useAttachments(activeId: string | null) {
  const [pendingImages, setPendingImages] = useState<MessageImage[]>([]);
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([]);

  /** Durable-upload rung: too big (or too binary) for inline/native — the
   *  file streams to object storage NOW (eager, while the user types) and
   *  the message will carry only its reference. */
  /** Durable copy for an existing chip: streams the bytes to object storage
   *  and attaches file_id to the chip (which is what makes it bookmarkable
   *  into My Files). Failure just clears the uploading flag — the chip still
   *  works through its inline/native ride. */
  function uploadDurable(file: File, key: string) {
    const convId = activeId;
    if (!convId) return;
    uploadConversationFile(convId, file, (pct) =>
      setPendingFiles((prev) =>
        prev.map((f) => (f.key === key ? { ...f, progress: pct } : f)),
      ),
    )
      .then((res) =>
        setPendingFiles((prev) =>
          prev.map((f) =>
            f.key === key
              ? { ...f, uploading: false, progress: 100, file_id: res.file_id }
              : f,
          ),
        ),
      )
      .catch((err) => {
        console.error('[aloy] durable upload failed:', err);
        setPendingFiles((prev) =>
          prev.map((f) =>
            f.key === key ? { ...f, uploading: false, error: true } : f,
          ),
        );
      });
  }

  function uploadAttachment(file: File) {
    if (!activeId || file.size > MAX_UPLOAD_BYTES) return;
    if (pendingFiles.length >= MAX_FILES) return;
    const key = `${file.name}-${Date.now()}-${Math.random()}`;
    setPendingFiles((prev) =>
      prev.length >= MAX_FILES
        ? prev
        : [
            ...prev,
            { key, name: file.name, size: file.size, uploading: true, progress: 0 },
          ],
    );
    uploadDurable(file, key);
  }

  /** Route attachments: images render for the model's eyes; text-like files
   *  embed their content into the task; everything bigger or binary takes
   *  the durable-upload rung. Docs and text files ALSO get a durable copy
   *  (same chip), so any attachment can be saved to My Files — and later
   *  turns can still reach the bytes in the sandbox. */
  function addAttachments(files: Iterable<File>) {
    for (const file of files) {
      if (file.type.startsWith('image/')) {
        if (file.size > MAX_IMAGE_BYTES) {
          uploadAttachment(file);
          continue;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = String(reader.result || '').split(',')[1] ?? '';
          if (!base64) return;
          setPendingImages((prev) =>
            prev.length >= MAX_IMAGES
              ? prev
              : [...prev, { data: base64, media_type: file.type }],
          );
        };
        reader.readAsDataURL(file);
      } else if (DOC_MIMES[file.name.split('.').pop()?.toLowerCase() ?? '']) {
        if (file.size > MAX_DOC_BYTES) {
          uploadAttachment(file);
          continue;
        }
        if (pendingFiles.length >= MAX_FILES) continue;
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        const mediaType = DOC_MIMES[file.name.split('.').pop()!.toLowerCase()];
        setPendingFiles((prev) =>
          prev.length >= MAX_FILES
            ? prev
            : [
                ...prev,
                {
                  key,
                  name: file.name,
                  size: file.size,
                  media_type: mediaType,
                  uploading: true,
                  progress: 0,
                },
              ],
        );
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = String(reader.result || '').split(',')[1] ?? '';
          if (!base64) return;
          setPendingFiles((prev) =>
            prev.map((f) => (f.key === key ? { ...f, data: base64 } : f)),
          );
        };
        reader.readAsDataURL(file);
        uploadDurable(file, key);
      } else if (file.type.startsWith('text/') || TEXT_EXTENSIONS.test(file.name)) {
        if (file.size > MAX_FILE_CHARS) {
          // Too big to inline without truncating — store it whole instead.
          uploadAttachment(file);
          continue;
        }
        if (pendingFiles.length >= MAX_FILES) continue;
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        setPendingFiles((prev) =>
          prev.length >= MAX_FILES
            ? prev
            : [
                ...prev,
                {
                  key,
                  name: file.name,
                  size: file.size,
                  uploading: true,
                  progress: 0,
                },
              ],
        );
        const reader = new FileReader();
        reader.onload = () => {
          const text = String(reader.result || '').slice(0, MAX_FILE_CHARS);
          if (!text) return;
          setPendingFiles((prev) =>
            prev.map((f) =>
              f.key === key ? { ...f, content: text, size: text.length } : f,
            ),
          );
        };
        reader.readAsText(file);
        uploadDurable(file, key);
      } else {
        // Any other type (zip, sqlite, parquet, unknown binaries…): the
        // durable-upload rung — the agent works on it in the sandbox.
        uploadAttachment(file);
      }
    }
  }

  function removeImage(index: number) {
    setPendingImages((prev) => prev.filter((_, idx) => idx !== index));
  }

  function removeFile(index: number) {
    setPendingFiles((prev) => prev.filter((_, idx) => idx !== index));
  }

  function attachStoredFile(reference: StoredFileReference) {
    setPendingFiles((prev) => {
      if (prev.some((file) => file.file_id === reference.file_id)) return prev;
      if (prev.length >= MAX_FILES) return prev;
      return [
        ...prev,
        {
          key: `stored:${reference.file_id}`,
          file_id: reference.file_id,
          name: reference.name,
          size: reference.size,
        },
      ];
    });
  }

  /** Clear everything staged (called after the send captures the arrays). */
  function resetAttachments() {
    setPendingImages([]);
    setPendingFiles([]);
  }

  return {
    pendingImages,
    pendingFiles,
    addAttachments,
    attachStoredFile,
    removeImage,
    removeFile,
    resetAttachments,
    /** Some chip is still streaming to object storage — sends must wait. */
    uploadsInFlight: pendingFiles.some((f) => f.uploading),
    /** Both attachment budgets exhausted (drives the composer's attach UI). */
    attachmentsFull:
      pendingImages.length >= MAX_IMAGES && pendingFiles.length >= MAX_FILES,
  };
}
