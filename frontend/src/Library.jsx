import React, { useRef, useState } from 'react'
import { BookOpen, Check, CircleAlert, FileText, LibraryBig, LoaderCircle, RefreshCw, Search, Trash2, Upload } from 'lucide-react'
import { api, bookTitle } from './api'

export function Collection({ books, loading, error, openLibrary }) {
  return (
    <aside className="collection" aria-labelledby="collection-title">
      <div className="collection-heading">
        <div className="collection-icon"><LibraryBig size={20} strokeWidth={1.75} /></div>
        <div>
          <h2 id="collection-title">Book collection</h2>
          <p>{books ? `${books.length} indexed ${books.length === 1 ? 'title' : 'titles'}` : 'Indexed titles'}</p>
        </div>
      </div>
      {books?.length ? (
        <ul className="collection-list">
          {books.slice(0, 4).map((book) => (
            <li key={book.book_id}><FileText size={17} /><span>{bookTitle(book.filename)}</span></li>
          ))}
        </ul>
      ) : (
        <p className="collection-empty">
          {loading ? 'Loading books…' : error ? 'The collection is unavailable.' : 'The library has no indexed books yet.'}
        </p>
      )}
      <button type="button" onClick={openLibrary} className="collection-button">View library</button>
    </aside>
  )
}

export default function Library({ books, loading, error, refresh, onUpload, onRemove, onAsk }) {
  const [search, setSearch] = useState('')
  const [busy, setBusy] = useState(null)
  const [notice, setNotice] = useState(null)
  const fileInput = useRef(null)
  const confirmDialog = useRef(null)
  const [bookToRemove, setBookToRemove] = useState(null)

  const term = search.trim().toLowerCase()
  const filtered = (books || []).filter(({ filename }) =>
    bookTitle(filename).toLowerCase().includes(term) || filename.toLowerCase().includes(term),
  )

  async function upload(event) {
    const file = event.target.files[0]
    event.target.value = ''
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf') || !file.size) {
      setNotice({ error: true, text: 'Choose a PDF file with content.' })
      return
    }
    setBusy({ type: 'upload', label: bookTitle(file.name) })
    setNotice(null)
    try {
      const result = await api.upload(file)
      onUpload(result)
      setSearch('')
      setNotice({ text: `${bookTitle(result.filename)} was indexed.` })
    } catch (uploadError) {
      setNotice({ error: true, text: uploadError.message })
    } finally {
      setBusy(null)
    }
  }

  async function remove() {
    const book = bookToRemove
    if (!book) return
    confirmDialog.current.close()
    setNotice(null)
    setBusy({ type: 'remove', id: book.book_id })
    try {
      const result = await api.remove(book.book_id)
      onRemove(result.book_id)
      setNotice({ text: `${bookTitle(book.filename)} was removed.` })
    } catch (removeError) {
      setNotice({ error: true, text: removeError.message })
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="library-page" aria-labelledby="library-heading">
      <div className="library-header">
        <div>
          <div className="collection-icon"><LibraryBig size={23} /></div>
          <h1 id="library-heading">Book library</h1>
          <p>Browse the titles the assistant uses to answer questions.</p>
        </div>
        <input type="file" accept=".pdf,application/pdf" ref={fileInput} onChange={upload} hidden />
        <button className="primary-button" type="button" disabled={Boolean(busy)} onClick={() => fileInput.current.click()}>
          {busy?.type === 'upload' ? <LoaderCircle className="spin" size={18} /> : <Upload size={18} />}
          {busy?.type === 'upload' ? 'Indexing PDF…' : 'Upload PDF'}
        </button>
      </div>
      {busy?.type === 'upload' && <p className="library-notice" role="status">Indexing {busy.label}…</p>}
      {notice && (
        <p className={'library-notice' + (notice.error ? ' is-error' : '')} role={notice.error ? 'alert' : 'status'}>
          {notice.error ? <CircleAlert size={18} /> : <Check size={18} />} {notice.text}
        </p>
      )}
      <div className="library-toolbar">
        <label className="search-input"><Search size={18} /><span className="sr-only">Search books</span>
          <input type="search" placeholder="Search titles…" value={search} onChange={(event) => setSearch(event.target.value)} />
        </label>
        <p>{books && `${term ? `${filtered.length} of ` : ''}${books.length} ${books.length === 1 ? 'book' : 'books'}`}</p>
      </div>
      {error && (
        <div className="library-notice is-error" role="alert">
          <CircleAlert size={18} /><span>{error}</span>
          <button className="text-button" type="button" onClick={refresh} disabled={loading}><RefreshCw size={15} /> Try again</button>
        </div>
      )}
      {filtered.length ? (
        <ul className="book-list">
          {filtered.map((book) => (
            <li key={book.book_id}>
              <span className="book-file-icon" aria-hidden="true"><FileText size={20} /></span>
              <div className="book-info">
                <h2>{bookTitle(book.filename)}</h2>
                <p>{book.chunk_count.toLocaleString()} indexed {book.chunk_count === 1 ? 'passage' : 'passages'}</p>
              </div>
              <button type="button" className="text-button ask-book" onClick={() => onAsk(bookTitle(book.filename))}>
                Ask about this book
              </button>
              <button
                type="button"
                className="icon-button delete-button"
                aria-label={`Remove ${bookTitle(book.filename)}`}
                onClick={() => {
                  setBookToRemove(book)
                  confirmDialog.current.showModal()
                }}
                disabled={Boolean(busy)}
              >
                {busy?.type === 'remove' && busy.id === book.book_id ? <LoaderCircle className="spin" size={17} /> : <Trash2 size={17} />}
              </button>
            </li>
          ))}
        </ul>
      ) : !error && (
        <div className="library-empty" role="status">
          {loading ? <LoaderCircle className="spin" size={24} /> : <BookOpen size={24} />}
          <p>{loading ? 'Loading books…' : term ? 'No books match your search.' : 'Upload a PDF to add the first book.'}</p>
        </div>
      )}
      <dialog className="remove-dialog" ref={confirmDialog} aria-labelledby="remove-heading">
        <h2 id="remove-heading">Remove book?</h2>
        <p>This will remove <strong>{bookToRemove && bookTitle(bookToRemove.filename)}</strong> and its indexed passages from the library. The server will also delete its copy of the PDF.</p>
        <div>
          <form method="dialog"><button className="secondary-button" type="submit" autoFocus>Cancel</button></form>
          <button className="danger-button" type="button" onClick={remove}>Remove book</button>
        </div>
      </dialog>
    </section>
  )
}
