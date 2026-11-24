import React, { useCallback, useEffect, useRef, useState } from 'react'
import { BookOpen, LibraryBig, Menu, MessageCircle, Plus, RefreshCw, X } from 'lucide-react'
import Chat, { LibraryMark } from './Chat'
import Library, { Collection } from './Library'
import { api, questionLimit } from './api'

const historyKey = 'talal-library-conversations-v1'

function readThreads() {
  try {
    const stored = JSON.parse(localStorage.getItem(historyKey) || '[]')
    return stored.map((thread) => ({
      ...thread,
      turns: thread.turns.map((turn) => turn.status === 'pending'
        ? { ...turn, status: 'error', error: 'The request was interrupted. You can retry this question.' }
        : turn),
    }))
  } catch {
    return []
  }
}

const sortBooks = (books) => [...books].sort((a, b) => a.filename.localeCompare(b.filename))

function Sidebar({ open, close, page, threads, threadId, onNew, onNavigate, onThread }) {
  return (
    <>
      {open && <button className="sidebar-backdrop" type="button" onClick={close} aria-label="Close navigation" />}
      <aside className={`sidebar${open ? ' sidebar-open' : ''}`} aria-label="Workspace navigation">
        <div className="brand">
          <LibraryMark />
          <div><strong>Talal Library</strong><span>AI book assistant</span></div>
          <button className="icon-button close-menu" type="button" onClick={close} aria-label="Close navigation"><X size={20} /></button>
        </div>
        <button className="new-chat" type="button" onClick={onNew}><Plus size={19} /> New chat</button>
        <nav aria-label="Workspace">
          <button type="button" aria-current={page === 'chat' ? 'page' : undefined} onClick={() => onNavigate('chat')}>
            <MessageCircle size={19} /> Chat
          </button>
          <button type="button" aria-current={page === 'library' ? 'page' : undefined} onClick={() => onNavigate('library')}>
            <LibraryBig size={19} /> Book library
          </button>
        </nav>
        <div className="history">
          <h2>Recent chats</h2>
          {threads.length ? (
            <ul>
              {threads.map((thread) => (
                <li key={thread.id}>
                  <button type="button" title={thread.turns[0].question} onClick={() => onThread(thread.id)}
                    aria-current={page === 'chat' && thread.id === threadId ? 'true' : undefined}>
                    <MessageCircle size={15} /><span>{thread.turns[0].question}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : <p>Your chats will appear here.</p>}
        </div>
        <div className="sidebar-footer"><BookOpen size={17} /> Writings of Talal Abu Ghazaleh</div>
      </aside>
    </>
  )
}

export default function App() {
  const [page, setPage] = useState('chat')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [threads, setThreads] = useState(readThreads)
  const [threadId, setThreadId] = useState(null)
  const [draft, setDraft] = useState('')
  const [focusChange, setFocusChange] = useState(0)
  const [books, setBooks] = useState(null)
  const [connection, setConnection] = useState('checking')
  const [loadingBooks, setLoadingBooks] = useState(true)
  const [bookError, setBookError] = useState('')
  const [pending, setPending] = useState(null)
  const request = useRef(null)
  const refreshId = useRef(0)
  const menuButton = useRef(null)
  const thread = threads.find(({ id }) => id === threadId)

  useEffect(() => {
    try {
      localStorage.setItem(historyKey, JSON.stringify(threads))
    } catch {
      // The active conversation remains usable if browser storage is unavailable.
    }
  }, [threads])

  const refresh = useCallback(async () => {
    const id = ++refreshId.current
    setLoadingBooks(true)
    const [healthResult, booksResult] = await Promise.allSettled([api.health(), api.books()])
    if (id !== refreshId.current) return
    setConnection(healthResult.status === 'fulfilled' && healthResult.value.status === 'ok' ? 'online' : 'offline')
    if (booksResult.status === 'fulfilled') {
      setBooks(sortBooks(booksResult.value.books))
      setBookError('')
    } else {
      setBookError(booksResult.reason.message)
    }
    setLoadingBooks(false)
  }, [])

  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    if (!sidebarOpen) return
    function closeOnEscape(event) {
      if (event.key === 'Escape') {
        setSidebarOpen(false)
        menuButton.current?.focus()
      }
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [sidebarOpen])

  function navigate(target) {
    setPage(target)
    setSidebarOpen(false)
    if (target === 'library' && books === null && !loadingBooks) refresh()
  }

  function openChat(id = null, question = '') {
    setThreadId(id)
    setDraft(question.slice(0, questionLimit))
    setFocusChange((count) => count + 1)
    navigate('chat')
  }

  function updateTurn(id, turnId, data) {
    setThreads((previous) => previous.map((item) => item.id === id
      ? { ...item, turns: item.turns.map((turn) => turn.id === turnId ? { ...turn, ...data } : turn) }
      : item))
  }

  async function resolveQuestion(id, turn) {
    const controller = new AbortController()
    request.current = controller
    setPending({ threadId: id, turnId: turn.id })
    try {
      const result = await api.ask(turn.question, controller.signal)
      updateTurn(id, turn.id, { status: 'done', result })
      if (connection !== 'online') refresh()
    } catch (error) {
      const message = error.name === 'AbortError'
        ? 'You stopped waiting for the answer. You can retry this question.'
        : error.message
      updateTurn(id, turn.id, { status: 'error', error: message })
      if (error.name !== 'AbortError') refresh()
    } finally {
      request.current = null
      setPending(null)
    }
  }

  function send(question) {
    if (request.current) return
    const id = threadId || crypto.randomUUID()
    const turn = { id: crypto.randomUUID(), question, status: 'pending' }
    setThreadId(id)
    setDraft('')
    setThreads((previous) => {
      const existing = previous.find((item) => item.id === id)
      const updated = {
        id,
        turns: [...(existing?.turns || []), turn],
      }
      return [updated, ...previous.filter((item) => item.id !== id)].slice(0, 30)
    })
    resolveQuestion(id, turn)
  }

  function retry(id, turn) {
    if (request.current) return
    updateTurn(id, turn.id, { status: 'pending', error: undefined })
    resolveQuestion(id, turn)
  }

  function onUpload(result) {
    setBooks((previous) => previous && sortBooks([
      ...previous.filter(({ book_id }) => book_id !== result.book_id),
      { book_id: result.book_id, filename: result.filename, chunk_count: result.chunks_created },
    ]))
    refresh()
  }

  function onRemove(id) {
    setBooks((previous) => previous?.filter(({ book_id }) => book_id !== id) ?? null)
    refresh()
  }

  return (
    <div className="app-shell">
      <Sidebar
        open={sidebarOpen}
        close={() => { setSidebarOpen(false); menuButton.current?.focus() }}
        page={page}
        threads={threads}
        threadId={threadId}
        onNew={() => openChat()}
        onNavigate={navigate}
        onThread={openChat}
      />
      <main className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            <button ref={menuButton} className="icon-button menu-button" type="button" onClick={() => setSidebarOpen(true)}
              aria-label="Open navigation" aria-expanded={sidebarOpen}><Menu size={23} /></button>
            {page === 'chat' ? <MessageCircle size={19} /> : <LibraryBig size={19} />}
            <span>{page === 'chat' ? 'Book chat' : 'Book library'}</span>
          </div>
          <button
            className={`connection connection-${connection}`}
            onClick={refresh}
            type="button"
            title="Refresh backend connection"
            disabled={loadingBooks}
          >
            {connection === 'offline' ? <RefreshCw size={13} /> : <span className="connection-dot" />}
            {connection === 'online' ? 'Backend connected' : connection === 'offline' ? 'Reconnect backend' : 'Connecting…'}
          </button>
        </header>
        {page === 'chat' ? (
          <div className="chat-layout">
            <Chat
              thread={thread}
              draft={draft}
              onDraft={setDraft}
              send={send}
              retry={retry}
              busy={Boolean(pending)}
              active={Boolean(pending && pending.threadId === threadId)}
              onStop={() => request.current?.abort()}
              focusChange={focusChange}
              openLibrary={() => navigate('library')}
            />
            <Collection books={books} loading={loadingBooks} error={bookError} openLibrary={() => navigate('library')} />
          </div>
        ) : (
          <Library
            books={books}
            loading={loadingBooks}
            error={bookError}
            refresh={refresh}
            onUpload={onUpload}
            onRemove={onRemove}
            onAsk={(title) => openChat(null, `What are the main ideas in "${title}"?`)}
          />
        )}
      </main>
    </div>
  )
}
