export interface FileVisualDescriptor {
  id?: string;
  file_id?: string;
  name: string;
  content_type?: string;
  kind?: string;
  size?: number;
  size_bytes?: number;
}

export type FileVisualKind =
  | 'pdf'
  | 'document'
  | 'spreadsheet'
  | 'slides'
  | 'image'
  | 'video'
  | 'audio'
  | 'archive'
  | 'code'
  | 'markdown'
  | 'text'
  | 'artifact'
  | 'unknown';

const IMAGE_EXTENSIONS = new Set(['avif', 'bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'svg', 'webp']);
const VIDEO_EXTENSIONS = new Set(['avi', 'm4v', 'mkv', 'mov', 'mp4', 'ogv', 'webm']);
const AUDIO_EXTENSIONS = new Set(['aac', 'flac', 'm4a', 'mp3', 'oga', 'ogg', 'opus', 'wav']);
const ARCHIVE_EXTENSIONS = new Set(['7z', 'bz2', 'gz', 'rar', 'tar', 'tgz', 'xz', 'zip']);
const CODE_EXTENSIONS = new Set(['c', 'cpp', 'css', 'go', 'h', 'html', 'java', 'js', 'jsx', 'php', 'py', 'rb', 'rs', 'sh', 'sql', 'swift', 'ts', 'tsx']);
const TEXT_EXTENSIONS = new Set(['csv', 'env', 'ini', 'json', 'jsonl', 'log', 'toml', 'tsv', 'txt', 'xml', 'yaml', 'yml']);

export function fileExtension(name: string) {
  const clean = name.split(/[?#]/, 1)[0] ?? name;
  const part = clean.split(/[/\\]/).pop() ?? clean;
  const dot = part.lastIndexOf('.');
  return dot > 0 ? part.slice(dot + 1).toLowerCase() : '';
}

export function resolveFileVisual(file: FileVisualDescriptor): FileVisualKind {
  const ext = fileExtension(file.name);
  const mediaType = ((file.content_type ?? '').split(';', 1)[0] ?? '').toLowerCase();
  // Prefer a recognized filename extension over ambiguous OS MIME metadata.
  // Windows commonly reports TypeScript `.ts` files as MPEG `video/mp2t`.
  if (ext === 'pdf') return 'pdf';
  if (IMAGE_EXTENSIONS.has(ext)) return 'image';
  if (VIDEO_EXTENSIONS.has(ext)) return 'video';
  if (AUDIO_EXTENSIONS.has(ext)) return 'audio';
  if (ext === 'doc' || ext === 'docx') return 'document';
  if (['xls', 'xlsx', 'xlsm'].includes(ext)) return 'spreadsheet';
  if (['ppt', 'pptx'].includes(ext)) return 'slides';
  if (ARCHIVE_EXTENSIONS.has(ext)) return 'archive';
  if (ext === 'md' || ext === 'mdx') return 'markdown';
  if (CODE_EXTENSIONS.has(ext)) return 'code';
  if (TEXT_EXTENSIONS.has(ext)) return 'text';

  if (mediaType === 'application/pdf') return 'pdf';
  if (mediaType.startsWith('image/')) return 'image';
  if (mediaType.startsWith('video/')) return 'video';
  if (mediaType.startsWith('audio/')) return 'audio';
  if (mediaType.includes('wordprocessingml')) return 'document';
  if (mediaType.includes('spreadsheetml')) return 'spreadsheet';
  if (mediaType.includes('presentationml')) return 'slides';
  if (mediaType === 'text/markdown') return 'markdown';
  if (mediaType.startsWith('text/')) return 'text';
  if (file.kind === 'artifact') return 'artifact';
  return 'unknown';
}
