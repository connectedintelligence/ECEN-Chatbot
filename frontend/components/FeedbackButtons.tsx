"use client";

import { ThumbsUp, ThumbsDown } from "lucide-react";

/**
 * Thumbs up/down feedback control for an assistant answer.
 *
 * Behaviour (issue #15): feedback is changeable, not write-once. Only the
 * thumb matching the CURRENT rating is disabled, so the user can always switch
 * their vote (up -> down or down -> up). Clicking the already-selected thumb is
 * a no-op, which the parent enforces in sendFeedback.
 */
export function FeedbackButtons({
  feedback,
  onRate,
}: {
  feedback?: "up" | "down";
  onRate: (rating: "up" | "down") => void;
}) {
  return (
    <div style={{ display: "flex", gap: "6px", marginTop: "6px", paddingLeft: "4px", alignItems: "center" }}>
      <button
        onClick={() => onRate("up")}
        title="Good answer"
        aria-label="Good answer"
        aria-pressed={feedback === "up"}
        disabled={feedback === "up"}
        style={{ background: "none", border: "none", cursor: feedback === "up" ? "default" : "pointer", color: feedback === "up" ? "#15803d" : "#9ca3af", padding: "2px" }}
      >
        <ThumbsUp size={14} />
      </button>
      <button
        onClick={() => onRate("down")}
        title="Bad answer"
        aria-label="Bad answer"
        aria-pressed={feedback === "down"}
        disabled={feedback === "down"}
        style={{ background: "none", border: "none", cursor: feedback === "down" ? "default" : "pointer", color: feedback === "down" ? "#b91c1c" : "#9ca3af", padding: "2px" }}
      >
        <ThumbsDown size={14} />
      </button>
      {feedback && <span style={{ fontSize: "0.7rem", color: "#9ca3af" }}>Thanks for the feedback!</span>}
    </div>
  );
}
