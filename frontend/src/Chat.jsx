import React, { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  ArrowUp, AudioLines, BookOpen, BriefcaseBusiness, ChevronDown, CircleAlert,
  Copy, Cpu, Layers3, RefreshCw, Scale, Square, UserRound,
} from 'lucide-react'
import { bookTitle, questionLimit } from './api'

const suggestions = [
  { title: 'Knowledge economy', question: 'What is the knowledge economy?', Icon: Layers3 },
  { title: 'AI and society', question: 'How might artificial intelligence change society?', Icon: Cpu },
  { title: 'Business leadership', question: 'What makes a good business leader?', Icon: BriefcaseBusiness },
  { title: 'Ethics and values', question: 'What does Talal Abu Ghazaleh say about ethics and values?', Icon: Scale },
]

export function LibraryMark({ className = '' }) {
  return <span className={'library-mark ' + className} aria-hidden="true"><BookOpen size={23} strokeWidth={1.8} /></span>
}

function Sources({ sources }) {
  if (!sources?.length) return null
  return (
    <div className="sources">
      <h3><BookOpen size={15} /> Sources <span>{sources.length}</span></h3>
      {sources.map((source, index) => (
        <details className="source" key={index}>
          <summary>
            <span className="source-name">{bookTitle(source.book)}</span>
            <span className="source-page">Page {source.page}</span>
            <ChevronDown size={15} aria-hidden="true" />
          </summary>
          <blockquote>{source.chunk}</blockquote>
        </details>
      ))}
    </div>
  )
}

function Turn({ turn, retry, blocked }) {
  const [copyState, setCopyState] = React.useState('')
  const answer = turn.result?.answer

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(answer)
      setCopyState('Copied')
    } catch {
      setCopyState('Copy unavailable')
    }
  }

  return (
    <article className="turn">
      <div className="message question">
        <span className="avatar user-avatar" aria-hidden="true"><UserRound size={16} /></span>
        <div className="message-body"><div className="message-name">You</div><p>{turn.question}</p></div>
      </div>
      <div className="message answer">
        <LibraryMark className="avatar" />
        <div className="message-body">
          <div className="message-name">Talal Library</div>
          {turn.status === 'pending' && (
            <p className="thinking" role="status"><AudioLines size={17} /> Searching the books…</p>
          )}
          {turn.status === 'error' && (
            <div className="response-error">
              <p role="status"><CircleAlert size={17} /> {turn.error}</p>
              <button className="text-button" type="button" onClick={retry} disabled={blocked}>
                <RefreshCw size={14} /> Retry question
              </button>
            </div>
          )}
          {turn.status === 'done' && (turn.result.allowed === false ? (
            <p className="refusal" role="status">{turn.result.message}</p>
          ) : (
            <>
              <div className="markdown" role="status">
                <ReactMarkdown skipHtml disallowedElements={['img']}>{answer}</ReactMarkdown>
              </div>
              <Sources sources={turn.result.sources} />
              {answer && (
                <button className="text-button copy-button" type="button" onClick={copyAnswer}>
                  <Copy size={15} /> {copyState || 'Copy answer'}
                </button>
              )}
            </>
          ))}
        </div>
      </div>
    </article>
  )
}

export default function Chat({
  thread, draft, onDraft, send, retry, busy, onStop, active, focusChange, openLibrary,
}) {
  const input = useRef(null)
  const scroll = useRef(null)
  const turns = thread?.turns || []
  const lastTurn = turns.at(-1)
  const trimmedDraft = draft.trim()

  useEffect(() => {
    if (focusChange) input.current?.focus()
  }, [focusChange])

  useEffect(() => {
    const viewport = scroll.current
    if (!viewport || !lastTurn) return
    if (lastTurn.status === 'pending' || viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 220) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [thread?.id, lastTurn])

  function submit(event) {
    event.preventDefault()
    if (trimmedDraft && !busy) send(trimmedDraft)
  }

  function handleKey(event) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      submit(event)
    }
  }

  return (
    <section className="chat-area" aria-label="Book chat">
      <div className="chat-scroll" ref={scroll}>
        {turns.length ? (
          <div className="conversation">
            {turns.map((turn) => (
              <Turn key={turn.id} turn={turn} retry={() => retry(thread.id, turn)} blocked={busy} />
            ))}
          </div>
        ) : (
          <div className="welcome">
            <LibraryMark className="hero-mark" />
            <h1>Ask the books<span>.</span></h1>
            <p className="welcome-copy">Explore the writings of Talal Abu Ghazaleh. Ask a question and trace the answer back to its source.</p>
            <h2 className="suggestions-label">Try a question</h2>
            <div className="suggestions">
              {suggestions.map(({ title, question, Icon }) => (
                <button className="suggestion" key={title} type="button" onClick={() => {
                  onDraft(question)
                  input.current?.focus()
                }}>
                  <span className="suggestion-icon"><Icon size={18} strokeWidth={1.8} /></span>
                  <span className="suggestion-text"><strong>{title}</strong><span>{question}</span></span>
                </button>
              ))}
            </div>
            <button className="text-button mobile-library" onClick={openLibrary} type="button">
              <BookOpen size={16} /> Browse the library
            </button>
          </div>
        )}
      </div>
      <div className="composer-wrap">
        <form className="composer" onSubmit={submit}>
          <label className="sr-only" htmlFor="question">Question about the books</label>
          <textarea
            ref={input}
            id="question"
            rows={2}
            maxLength={questionLimit}
            placeholder="Ask a question about the books…"
            value={draft}
            onChange={(event) => onDraft(event.target.value)}
            onKeyDown={handleKey}
          />
          {active ? (
            <button className="send-button" type="button" onClick={onStop} aria-label="Stop waiting" title="Stop waiting">
              <Square size={17} fill="currentColor" />
            </button>
          ) : (
            <button className="send-button" type="submit" disabled={!trimmedDraft || busy} aria-label="Send question" title="Send question">
              <ArrowUp size={21} />
            </button>
          )}
        </form>
        <div className="composer-note">
          <span>{busy && !active ? 'Another question is being answered.' : 'Answers include excerpts from the books when sources are found.'}</span>
          {draft.length >= 1700 && <span>{draft.length} / {questionLimit}</span>}
        </div>
      </div>
    </section>
  )
}
