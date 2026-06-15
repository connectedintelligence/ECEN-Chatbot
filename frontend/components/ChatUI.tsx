"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import { Send, Square, User, ExternalLink, RefreshCw, Flag, X, Check, ThumbsUp, ThumbsDown } from "lucide-react";

/** EIRA — ECE Information & Resource Assistant — illustrated avatar (frontend/public/ellie-avatar.png). */
function EllieAvatar({ size = 30 }: { size?: number }) {
  return (
    <img
      src="/ellie-avatar.png?v=3"
      alt="EIRA"
      width={size}
      height={size}
      style={{ borderRadius: "50%", objectFit: "cover", display: "block", flexShrink: 0, border: `1px solid ${BORDER}` }}
    />
  );
}
import ReactMarkdown from "react-markdown";

interface Source { url: string; title: string; section: string; }
interface Message { role: "user" | "assistant"; content: string; sources?: Source[]; suggestions?: string[]; feedback?: "up" | "down"; loading?: boolean; }

import { parseAnswer } from "../lib/parseAnswer";

const MAROON = "#500000";
const BG = "#ffffff";
const CARD = "#f5f5f5";
const BORDER = "#e5e5e5";

const SECTION_COLORS: Record<string, { bg: string; color: string }> = {
  people:     { bg: "#f3e8ff", color: "#7e22ce" },
  research:   { bg: "#dbeafe", color: "#1d4ed8" },
  academics:  { bg: "#dcfce7", color: "#15803d" },
  admissions: { bg: "#fef9c3", color: "#a16207" },
  news:       { bg: "#fee2e2", color: "#b91c1c" },
  events:     { bg: "#ffedd5", color: "#c2410c" },
  about:      { bg: "#f3f4f6", color: "#4b5563" },
};

const TOPICS = [
  { label: "Research areas",     prompt: "What research areas does TAMU ECE specialize in?" },
  { label: "Faculty",            prompt: "Who are the faculty members in TAMU ECE?" },
  { label: "Graduate programs",  prompt: "What graduate programs are offered in TAMU ECE?" },
  { label: "Admissions",         prompt: "How do I apply to TAMU ECE?" },
  { label: "Patents",            prompt: "What patents has TAMU ECE filed?" },
  { label: "Scholarships",       prompt: "What scholarships and financial aid are available?" },
];

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [section] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const hasMessages = messages.length > 0;

  const [reportOpen, setReportOpen] = useState(false);
  const [reportText, setReportText] = useState("");
  const [reportStatus, setReportStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function submitReport() {
    if (!reportText.trim() || reportStatus === "sending") return;
    setReportStatus("sending");
    const lastUser = [...messages].reverse().find(m => m.role === "user");
    const lastAssistant = [...messages].reverse().find(m => m.role === "assistant" && !m.loading);
    const context = [
      lastUser ? `Q: ${lastUser.content}` : "",
      lastAssistant ? `A: ${lastAssistant.content}` : "",
    ].filter(Boolean).join("\n\n") || undefined;
    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: reportText.trim(), context }),
      });
      if (!res.ok) throw new Error();
      setReportStatus("sent");
      setReportText("");
      setTimeout(() => { setReportOpen(false); setReportStatus("idle"); }, 1800);
    } catch {
      setReportStatus("error");
    }
  }

  const reportButton = (
    <button onClick={() => setReportOpen(true)} title="Report a problem"
      style={{ display: "flex", alignItems: "center", gap: "6px", padding: "5px 12px", borderRadius: "999px", fontSize: "0.75rem", border: `1px solid ${BORDER}`, backgroundColor: "transparent", color: "#6b7280", cursor: "pointer" }}>
      <Flag size={12} /> Report a problem
    </button>
  );

  const reportModal = reportOpen ? (
    <div onClick={() => reportStatus !== "sending" && setReportOpen(false)}
      style={{ position: "fixed", inset: 0, backgroundColor: "rgba(0,0,0,0.4)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50, padding: "1rem" }}>
      <div onClick={e => e.stopPropagation()}
        style={{ width: "100%", maxWidth: "460px", backgroundColor: BG, borderRadius: "16px", border: `1px solid ${BORDER}`, padding: "20px", boxShadow: "0 10px 40px rgba(0,0,0,0.2)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px" }}>
          <span style={{ fontWeight: 600, color: "#111111", fontSize: "0.95rem" }}>Report a problem</span>
          <button onClick={() => setReportOpen(false)} style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af" }}><X size={16} /></button>
        </div>
        {reportStatus === "sent" ? (
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#15803d", padding: "12px 0", fontSize: "0.875rem" }}>
            <Check size={16} /> Thanks! Your report was sent to the team.
          </div>
        ) : (
          <>
            <p style={{ color: "#6b7280", fontSize: "0.8rem", margin: "0 0 10px" }}>
              Tell us what went wrong (wrong answer, missing info, an error, etc.). Your most recent question is attached automatically.
            </p>
            <textarea
              value={reportText}
              onChange={e => setReportText(e.target.value)}
              placeholder="e.g. I asked for power & energy faculty and the list was cut off."
              rows={4}
              style={{ width: "100%", boxSizing: "border-box", resize: "vertical", padding: "10px 12px", borderRadius: "10px", border: `1px solid ${BORDER}`, fontSize: "0.85rem", fontFamily: "inherit", outline: "none", color: "#111111" }}
            />
            {reportStatus === "error" && (
              <p style={{ color: "#b91c1c", fontSize: "0.78rem", margin: "8px 0 0" }}>Couldn't send the report. Please try again.</p>
            )}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "12px" }}>
              <button onClick={() => setReportOpen(false)}
                style={{ padding: "8px 14px", borderRadius: "999px", border: `1px solid ${BORDER}`, background: "transparent", color: "#6b7280", fontSize: "0.8rem", cursor: "pointer" }}>Cancel</button>
              <button onClick={submitReport} disabled={!reportText.trim() || reportStatus === "sending"}
                style={{ padding: "8px 16px", borderRadius: "999px", border: "none", backgroundColor: reportText.trim() ? MAROON : BORDER, color: "white", fontSize: "0.8rem", cursor: reportText.trim() ? "pointer" : "default" }}>
                {reportStatus === "sending" ? "Sending…" : "Send report"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  ) : null;

  async function send(question: string) {
    if (!question.trim() || streaming) return;
    setInput("");
    // Last 3 turns (6 messages) so the backend can resolve follow-up questions.
    const history = messages
      .filter(m => !m.loading && m.content &&
        m.content !== "Sorry, something went wrong. Please try again.")
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content.slice(0, 1500) }));
    setMessages(prev => [...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", loading: true },
    ]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          section_filter: section || undefined,
          history: history.length ? history : undefined,
        }),
        signal: controller.signal,
      });
      if (res.status === 429) {
        setMessages(prev => { const u = [...prev]; u[u.length - 1] = { role: "assistant", content: "Whoa, that's a lot of questions at once! Give me a minute to catch up, then ask again.", loading: false }; return u; });
        setStreaming(false); abortRef.current = null;
        return;
      }
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "", sources: Source[] = [], answer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") break;
          try { const p = JSON.parse(data); if (Array.isArray(p)) { sources = p; continue; } } catch {}
          answer += data;
          const parsed = parseAnswer(answer);
          setMessages(prev => { const u = [...prev]; u[u.length - 1] = { role: "assistant", content: parsed.display, sources, suggestions: parsed.suggestions, loading: false }; return u; });
        }
      }
    } catch {
      setMessages(prev => {
        const u = [...prev];
        const last = u[u.length - 1];
        u[u.length - 1] = controller.signal.aborted
          ? { ...last, content: last.content || "*Response stopped.*", loading: false }
          : { role: "assistant", content: "Sorry, something went wrong. Please try again.", loading: false };
        return u;
      });
    } finally { setStreaming(false); abortRef.current = null; }
  }

  function stopStreaming() { abortRef.current?.abort(); }

  function sendFeedback(i: number, rating: "up" | "down") {
    const msg = messages[i];
    if (!msg || msg.feedback) return;
    const q = [...messages.slice(0, i)].reverse().find(m => m.role === "user")?.content ?? "";
    setMessages(prev => { const u = [...prev]; u[i] = { ...u[i], feedback: rating }; return u; });
    fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating, question: q.slice(0, 1000), answer: msg.content.slice(0, 8000) }),
    }).catch(() => {});
  }

  function onSubmit(e: FormEvent) { e.preventDefault(); send(input.trim()); }

  /* ── LANDING ── */
  if (!hasMessages) return (
    <div className="app-viewport" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", width: "100%", backgroundColor: BG, fontFamily: "system-ui, sans-serif" }}>

      <div style={{ marginBottom: "1.25rem" }}><EllieAvatar size={96} /></div>
      <h1 className="landing-title" style={{ color: "#111111", fontSize: "clamp(1.4rem, 6vw, 2.25rem)", fontWeight: 400, margin: "0 0 0.5rem", letterSpacing: "-0.02em", textAlign: "center", maxWidth: "100%", padding: "0 1rem", boxSizing: "border-box" }}>
        Howdy, I'm EIRA!
      </h1>
      <p style={{ color: "#6b7280", fontSize: "1rem", margin: "0 0 2rem", textAlign: "center", padding: "0 1rem" }}>
        Your guide to everything ECE at Texas A&M — ask me anything.
      </p>

      <form onSubmit={onSubmit} style={{ width: "100%", maxWidth: "640px", padding: "0 1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", backgroundColor: CARD, border: `1px solid ${BORDER}`, borderRadius: "999px", padding: "14px 20px", boxShadow: "0 1px 6px rgba(0,0,0,0.08)" }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask anything"
            style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#111111", fontSize: "1rem" }}
          />
          <button type="submit" disabled={!input.trim()} style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "36px", height: "36px", borderRadius: "50%", backgroundColor: input.trim() ? MAROON : BORDER, border: "none", cursor: input.trim() ? "pointer" : "default", flexShrink: 0, transition: "background 0.2s" }}>
            <Send size={15} color="white" />
          </button>
        </div>
      </form>

      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "10px", marginTop: "1.5rem", maxWidth: "640px", padding: "0 1rem" }}>
        {TOPICS.map(t => (
          <button key={t.label} onClick={() => send(t.prompt)}
            style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 16px", borderRadius: "999px", backgroundColor: CARD, border: `1px solid ${BORDER}`, color: "#111111", fontSize: "0.875rem", cursor: "pointer" }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = MAROON; (e.currentTarget as HTMLButtonElement).style.color = "white"; (e.currentTarget as HTMLButtonElement).style.borderColor = MAROON; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = CARD; (e.currentTarget as HTMLButtonElement).style.color = "#111111"; (e.currentTarget as HTMLButtonElement).style.borderColor = BORDER; }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ marginTop: "1.75rem" }}>{reportButton}</div>
      {reportModal}
    </div>
  );

  /* ── CHAT ── */
  return (
    <div className="app-viewport" style={{ display: "flex", flexDirection: "column", backgroundColor: BG, fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div className="chat-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 24px", borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <EllieAvatar size={36} />
          <span style={{ color: "#111111", fontWeight: 600, fontSize: "0.9rem" }}>EIRA<span className="header-tagline"> · ECE Information &amp; Resource Assistant</span></span>
          {reportButton}
        </div>
      </div>

      {/* Messages */}
      <div className="chat-scroll" style={{ flex: 1, overflowY: "auto", padding: "24px 16px" }}>
        <div style={{ maxWidth: "720px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "20px" }}>
          {messages.map((msg, i) => (
            <div key={i} style={{ display: "flex", gap: "12px", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
              {msg.role === "assistant" && (
                <div className="msg-avatar" style={{ flexShrink: 0, marginTop: "4px" }}>
                  <EllieAvatar size={36} />
                </div>
              )}
              <div className="bubble-col" style={{ maxWidth: "75%" }}>
                <div style={{ padding: "12px 16px", borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "4px 18px 18px 18px", backgroundColor: msg.role === "user" ? MAROON : CARD, border: msg.role === "assistant" ? `1px solid ${BORDER}` : "none", color: msg.role === "user" ? "white" : "#111111", fontSize: "0.875rem", lineHeight: 1.6 }}>
                  {msg.loading ? (
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#6b7280" }}>
                      <RefreshCw size={13} style={{ animation: "spin 1s linear infinite" }} />
                      <span>EIRA is looking that up…</span>
                    </div>
                  ) : msg.role === "user" ? (
                    msg.content
                  ) : (
                    <div className="markdown-body">
                      <ReactMarkdown>{msg.content.replace(/\\n/g, "\n")}</ReactMarkdown>
                    </div>
                  )}
                </div>
                {/* Feedback + follow-up chips, once the answer is complete */}
                {msg.role === "assistant" && !msg.loading && msg.content && !(streaming && i === messages.length - 1) && (
                  <div style={{ display: "flex", gap: "6px", marginTop: "6px", paddingLeft: "4px", alignItems: "center" }}>
                    <button onClick={() => sendFeedback(i, "up")} title="Good answer" disabled={!!msg.feedback}
                      style={{ background: "none", border: "none", cursor: msg.feedback ? "default" : "pointer", color: msg.feedback === "up" ? "#15803d" : "#9ca3af", padding: "2px" }}>
                      <ThumbsUp size={14} />
                    </button>
                    <button onClick={() => sendFeedback(i, "down")} title="Bad answer" disabled={!!msg.feedback}
                      style={{ background: "none", border: "none", cursor: msg.feedback ? "default" : "pointer", color: msg.feedback === "down" ? "#b91c1c" : "#9ca3af", padding: "2px" }}>
                      <ThumbsDown size={14} />
                    </button>
                    {msg.feedback && <span style={{ fontSize: "0.7rem", color: "#9ca3af" }}>Thanks for the feedback!</span>}
                  </div>
                )}
                {msg.suggestions && msg.suggestions.length > 0 && i === messages.length - 1 && !streaming && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "10px" }}>
                    {msg.suggestions.map((s, si) => (
                      <button key={si} onClick={() => send(s)}
                        style={{ padding: "6px 12px", borderRadius: "999px", backgroundColor: BG, border: `1px solid ${BORDER}`, color: MAROON, fontSize: "0.75rem", cursor: "pointer", textAlign: "left" }}>
                        {s}
                      </button>
                    ))}
                  </div>
                )}
                {/* Sources render only after the answer finishes streaming */}
                {msg.sources && msg.sources.length > 0 && !(streaming && i === messages.length - 1) && (
                  <div style={{ marginTop: "8px", display: "flex", flexDirection: "column", gap: "4px" }}>
                    <span style={{ fontSize: "0.7rem", color: "#6b7280", paddingLeft: "4px" }}>Sources</span>
                    {msg.sources.map((s, si) => {
                      const col = SECTION_COLORS[s.section] ?? SECTION_COLORS.about;
                      return (
                        <a key={si} href={s.url} target="_blank" rel="noopener noreferrer"
                          style={{ display: "flex", alignItems: "flex-start", gap: "8px", padding: "8px 12px", borderRadius: "8px", backgroundColor: CARD, border: `1px solid ${BORDER}`, color: "#6b7280", fontSize: "0.75rem", textDecoration: "none" }}>
                          <ExternalLink size={11} style={{ marginTop: "2px", flexShrink: 0 }} />
                          <div>
                            <span style={{ color: "#111111", fontWeight: 500 }}>{s.title}</span>
                            <span style={{ marginLeft: "8px", padding: "1px 6px", borderRadius: "4px", fontSize: "0.65rem", backgroundColor: col.bg, color: col.color }}>{s.section}</span>
                          </div>
                        </a>
                      );
                    })}
                  </div>
                )}
              </div>
              {msg.role === "user" && (
                <div className="msg-avatar" style={{ width: "30px", height: "30px", borderRadius: "50%", backgroundColor: BORDER, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "4px" }}>
                  <User size={14} color="#9ca3af" />
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div style={{ padding: "16px", borderTop: `1px solid ${BORDER}`, flexShrink: 0 }}>
        <form onSubmit={onSubmit} style={{ maxWidth: "720px", margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", backgroundColor: CARD, border: `1px solid ${BORDER}`, borderRadius: "999px", padding: "12px 18px" }}>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask anything"
              style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#111111", fontSize: "0.9rem" }}
            />
            {streaming ? (
              <button type="button" onClick={stopStreaming} title="Stop generating"
                style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "34px", height: "34px", borderRadius: "50%", backgroundColor: MAROON, border: "none", cursor: "pointer", flexShrink: 0, transition: "background 0.2s" }}>
                <Square size={12} color="white" fill="white" />
              </button>
            ) : (
              <button type="submit" disabled={!input.trim()}
                style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "34px", height: "34px", borderRadius: "50%", backgroundColor: input.trim() ? MAROON : "#2a2a2a", border: "none", cursor: input.trim() ? "pointer" : "default", flexShrink: 0, transition: "background 0.2s" }}>
                <Send size={14} color="white" />
              </button>
            )}
          </div>
        </form>
      </div>

      {reportModal}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
