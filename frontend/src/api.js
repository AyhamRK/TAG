const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')
export const questionLimit = 2000

async function request(path, options) {
  let response
  try {
    response = await fetch(base + path, options)
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new Error('Could not reach the assistant server. Check that the backend is running and try again.')
  }

  const body = await response.json().catch(() => null)
  if (!response.ok || !body) {
    const detail = body?.detail
    const validationError = Array.isArray(detail) && detail.map((item) => item.msg).join(' ')
    const message = typeof detail === 'string' ? detail : validationError
    throw new Error(message || 'The assistant server could not complete the request. Try again shortly.')
  }
  return body
}

export const api = {
  health: () => request('/health', { signal: AbortSignal.timeout(12000) }),
  books: () => request('/api/v1/books', { signal: AbortSignal.timeout(12000) }),
  ask: (question, signal) =>
    request('/api/v1/chat/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal,
    }),
  upload: (file) => {
    const data = new FormData()
    data.append('file', file, file.name.replace(/\.pdf$/i, '.pdf'))
    return request('/api/v1/books/upload', { method: 'POST', body: data })
  },
  remove: (id) => request('/api/v1/books/' + encodeURIComponent(id), { method: 'DELETE' }),
}

const titles = new Map([
  ['ANTI_MONEY_LAUNDERING_COUNTER_FINANCING_OF_TERRORISM', 'Anti-Money Laundering Counter Financing of Terrorism'],
  ['because_i_like_the_truth', 'Because I Like the Truth'],
  ['Book_AI', 'AI'],
  ['Book_Moral_Message', 'Moral Message'],
])

export function bookTitle(filename) {
  const stem = filename.replace(/\.pdf$/i, '')
  const title = titles.get(stem) || stem.replace(/_/g, ' ').replace(/\b[A-Z]{3,}\b/g, (word) =>
    word[0] + word.slice(1).toLowerCase(),
  )
  return title.charAt(0).toUpperCase() + title.slice(1)
}
