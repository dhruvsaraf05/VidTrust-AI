/**
 * Client-side file validation.
 *
 * This mirrors ALLOWED_EXTENSIONS and MAX_FILE_BYTES in backend/config.py.
 * It is a convenience only: it saves a 50 MB upload that the server would
 * reject anyway. The server remains authoritative, and its UNSUPPORTED_FORMAT
 * / FILE_TOO_LARGE responses are still handled -- never assume a file that
 * passes here will be accepted (the backend also rejects empty files and
 * anything Pillow/OpenCV cannot decode, which we cannot check from here).
 */

export const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
export const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi']
export const ALLOWED_EXTENSIONS = [...IMAGE_EXTENSIONS, ...VIDEO_EXTENSIONS]

export const MAX_FILE_BYTES = 50 * 1024 * 1024

/** The `accept` attribute for the file input. */
export const ACCEPT_ATTR = ALLOWED_EXTENSIONS.join(',')

export function extensionOf(filename) {
  const index = filename.lastIndexOf('.')
  return index === -1 ? '' : filename.slice(index).toLowerCase()
}

export function mediaTypeOf(filename) {
  const extension = extensionOf(filename)
  if (IMAGE_EXTENSIONS.includes(extension)) return 'image'
  if (VIDEO_EXTENSIONS.includes(extension)) return 'video'
  return null
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Returns { ok: true } or { ok: false, message } for inline display. */
export function validateFile(file) {
  const extension = extensionOf(file.name)

  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    return {
      ok: false,
      message: `${extension || file.name} is not a supported format. Accepted: ${ALLOWED_EXTENSIONS.join(' ')}`,
    }
  }

  if (file.size > MAX_FILE_BYTES) {
    return {
      ok: false,
      message: `${formatBytes(file.size)} exceeds the 50 MB limit.`,
    }
  }

  if (file.size === 0) {
    return { ok: false, message: 'That file is empty.' }
  }

  return { ok: true }
}
