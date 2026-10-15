import { useEffect, useRef, useState } from "react";
import { api } from "./api";

const ACCEPTED = ".md,.markdown,.txt,.docx,.pdf";

// --- render an answer, turning [S1] markers into inline citation chips ------
function AnswerText({ text, citations }) {
  const bySource = Object.fromEntries((citations || []).map((c) => [c.marker, c]));
  const parts = text.split(/(\[S\d+\])/g); // keep the markers as separate tokens
  return (
    <span>
      {parts.map((part, i) => {
        const m = part.match(/^\[(S\d+)\]$/);
        if (m) {
          const c = bySource[m[1]];
          const title = c ? `${c.source}\n\n"${c.snippet}"` : m[1];
          return (
            <sup key={i} className="chip" title={title}>
              {m[1]}
            </sup>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

// --- one assistant turn: answer + metadata + collapsible sources ------------
function AssistantMessage({ turn }) {
  const [open, setOpen] = useState(false);
  const faithPct = Math.round((turn.faithfulness ?? 1) * 100);
  return (
    <div className="msg assistant">
      <div className="bubble">
        {turn.no_answer ? (
          <em className="noanswer">{turn.answer}</em>
        ) : (
          <AnswerText text={turn.answer} citations={turn.citations} />
        )}
        <div className="meta">
          <span className={`tag intent-${turn.intent}`}>{turn.intent}</span>
          {!turn.no_answer && turn.intent === "retrieve" && (
            <>
              <span className="tag">faithfulness {faithPct}%</span>
              <span className="tag">{turn.latency_ms} ms</span>
            </>
          )}
          {turn.citations?.length > 0 && (
            <button className="link" onClick={() => setOpen((o) => !o)}>
              {open ? "hide" : "show"} sources ({turn.citations.length})
            </button>
          )}
        </div>
        {open && (
          <div className="sources">
            {turn.citations.map((c) => (
              <div className="source" key={c.marker}>
                <span className="chip">{c.marker}</span>
                <div>
                  <div className="source-name">{c.source}</div>
                  <div className="snippet">{c.snippet}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [role, setRole] = useState("admin");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const scrollRef = useRef(null);
  const fileRef = useRef(null);

  const refreshStatus = () => api.status().then(setStatus).catch(() => {});
  const ingestRoles = status?.ingest_roles || ["admin", "owner"];
  const canIngest = ingestRoles.includes(role);

  useEffect(() => {
    refreshStatus();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages, busy]);

  async function sendText(raw) {
    const text = (raw ?? "").trim();
    if (!text || busy) return;
    setInput("");
    setMessages((m) => [...m, { who: "user", text }]);
    setBusy(true);
    try {
      const turn = await api.chat(text, role);
      setMessages((m) => [...m, { who: "assistant", turn }]);
      if (turn.intent === "ingest") refreshStatus();
    } catch (e) {
      setMessages((m) => [
        ...m,
        { who: "assistant", turn: { answer: `Error: ${e.message}`, intent: "error", no_answer: true, citations: [] } },
      ]);
    } finally {
      setBusy(false);
    }
  }

  const send = () => sendText(input);

  async function reindex() {
    setBusy(true);
    setNotice("");
    try {
      const res = await api.ingest(null, role);
      setNotice(res.message);
      refreshStatus();
    } catch (err) {
      setNotice(`Re-index failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file later
    if (!file) return;
    setBusy(true);
    setNotice(`Uploading "${file.name}"…`);
    try {
      const res = await api.upload(file, role);
      setNotice(res.message);
      refreshStatus();
    } catch (err) {
      setNotice(`Upload failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function newChat() {
    await api.reset();
    setMessages([]);
    setNotice("");
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>Multi-Agent RAG</h1>
          <span className="subtitle">document chat with grounded citations</span>
        </div>
        <div className="controls">
          {status && (
            <span className="status-badge" title={`collection: ${status.collection}`}>
              ● {status.indexed} chunks indexed · {status.documents} docs · {status.strategy}
            </span>
          )}
          <label className="role">
            role&nbsp;
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              {(status?.roles || ["viewer", "analyst", "admin", "owner"]).map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </label>
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED}
            style={{ display: "none" }}
            onChange={onUpload}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={busy || !canIngest}
            title={canIngest
              ? "Upload a .md / .txt / .docx / .pdf file and index it"
              : `Only ${ingestRoles.join(" / ")} can add documents — switch role`}
          >
            ⬆ Upload doc
          </button>
          <button
            onClick={reindex}
            disabled={busy || !canIngest}
            title={canIngest ? "Re-index the whole corpus" : `Only ${ingestRoles.join(" / ")} can re-index`}
          >
            Re-index
          </button>
          {canIngest && (
            <button
              onClick={() => sendText("/documents")}
              disabled={busy}
              title="List every document in the vector DB with chunk details"
            >
              Documents
            </button>
          )}
          <button onClick={newChat} disabled={busy}>New chat</button>
        </div>
      </header>

      {notice && <div className="notice">{notice}</div>}

      <main ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask a question about the indexed knowledge base, e.g.:</p>
            <ul>
              <li onClick={() => setInput("/help")}>
                <b>/help</b> — see which topics I can help with
              </li>
              <li onClick={() => setInput("Which encryption standard is used for data at rest?")}>
                Which encryption standard is used for data at rest?
              </li>
              <li onClick={() => setInput("How much does the Team edition cost per user per month?")}>
                How much does the Team edition cost per user per month?
              </li>
              <li onClick={() => setInput("What is the capital of France?")}>
                What is the capital of France? <em>(tests graceful no-answer)</em>
              </li>
            </ul>
            <p className="hint">
              Tip: ask <i>"Who should I contact about a security breach?"</i> as{" "}
              <b>admin</b>, then switch <b>role</b> to <b>viewer</b> and ask again —
              the viewer gets a graceful no-answer because the breach contact lives
              in a <b>confidential</b> document it isn't permitted to retrieve.
            </p>
          </div>
        )}
        {messages.map((m, i) =>
          m.who === "user" ? (
            <div className="msg user" key={i}>
              <div className="bubble">{m.text}</div>
            </div>
          ) : (
            <AssistantMessage turn={m.turn} key={i} />
          )
        )}
        {busy && <div className="msg assistant"><div className="bubble typing">…thinking</div></div>}
      </main>

      <footer>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask a question, type /help, or (admin/owner) 'ingest ./corpus/file.txt'…"
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim()}>Send</button>
      </footer>
    </div>
  );
}
