/**
 * Split a raw streamed answer into display text and optional follow-up suggestions.
 * Exported separately so it can be unit-tested without mounting ChatUI.
 */
export function parseAnswer(raw: string): { display: string; suggestions?: string[] } {
  const idx = raw.indexOf("|||SUGGEST");
  if (idx === -1) return { display: raw };
  const display = raw.slice(0, idx).trimEnd();
  const suggestions = raw
    .slice(idx + 10)
    .replace(/^:?\s*/, "")
    .split("|")
    .map((s) => s.trim())
    .filter((s) => s && !/^<q\d>$/i.test(s))
    .slice(0, 3);
  return { display, suggestions: suggestions.length ? suggestions : undefined };
}
