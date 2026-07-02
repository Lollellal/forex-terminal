import { useState } from "react";
import { Send } from "lucide-react";
import { Card } from "./Card";
import { Pill, directionTone, statusTone } from "./Pill";
import type { CloseReason, JournalViewResponse } from "../api/types";
import { useAddJournalNote, useCloseAllocation } from "../api/hooks";
import { formatDate } from "../lib/format";

const CLOSE_OPTIONS: { label: string; reason: CloseReason; defaultR: string }[] = [
  { label: "Win", reason: "TP", defaultR: "" },
  { label: "Loss", reason: "SL", defaultR: "" },
  { label: "BE", reason: "MANUAL", defaultR: "0" },
];

export function JournalCard({ entry }: { entry: JournalViewResponse }) {
  const [noteText, setNoteText] = useState("");
  const [closing, setClosing] = useState<CloseReason | null>(null);
  const [realizedR, setRealizedR] = useState("");

  const addNote = useAddJournalNote();
  const closeAllocation = useCloseAllocation();

  const canClose = entry.status === "OPEN";

  function submitNote() {
    const text = noteText.trim();
    if (!text) return;
    addNote.mutate(
      { allocationId: entry.allocation_id, text },
      { onSuccess: () => setNoteText("") },
    );
  }

  function startClose(reason: CloseReason, defaultR: string) {
    setClosing(reason);
    setRealizedR(defaultR);
  }

  function confirmClose() {
    if (closing === null) return;
    const value = Number(realizedR);
    if (Number.isNaN(value)) return;
    closeAllocation.mutate(
      { allocationId: entry.allocation_id, closeReason: closing, realizedR: value },
      { onSuccess: () => setClosing(null) },
    );
  }

  return (
    <Card className="animate-in">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-lg font-bold text-ink">{entry.pair}</p>
          <div className="mt-2 flex gap-2">
            <Pill tone={directionTone(entry.direction)}>{entry.direction}</Pill>
            <Pill tone={statusTone(entry.status)}>{entry.status}</Pill>
          </div>
        </div>
        {entry.realized_r !== null && (
          <div className="text-right">
            <p className="text-xs font-medium text-ink-soft">Result</p>
            <p
              className={`text-lg font-bold ${
                Number(entry.realized_r) > 0
                  ? "text-positive"
                  : Number(entry.realized_r) < 0
                    ? "text-negative"
                    : "text-ink-soft"
              }`}
            >
              {Number(entry.realized_r) > 0 ? "+" : ""}
              {Number(entry.realized_r).toFixed(2)}R
            </p>
          </div>
        )}
      </div>

      {entry.notes.length > 0 && (
        <ul className="mt-4 flex flex-col gap-2 border-t border-hairline pt-3">
          {entry.notes.map((note) => (
            <li key={note.note_id} className="text-sm text-ink">
              <span className="text-ink-soft">{formatDate(note.created_at)} · </span>
              {note.text}
            </li>
          ))}
        </ul>
      )}

      <div className="mt-4 flex items-center gap-2">
        <input
          value={noteText}
          onChange={(e) => setNoteText(e.target.value)}
          placeholder="Kommentar hinzufügen…"
          className="flex-1 rounded-pill border border-hairline bg-app-bg px-3.5 py-2 text-sm text-ink placeholder:text-ink-soft focus:border-primary focus:outline-none"
        />
        <button
          onClick={submitNote}
          disabled={!noteText.trim() || addNote.isPending}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-white disabled:opacity-40"
          aria-label="Notiz senden"
        >
          <Send size={16} strokeWidth={2} />
        </button>
      </div>

      {canClose && (
        <div className="mt-4 border-t border-hairline pt-4">
          {closing === null ? (
            <div className="flex gap-2">
              {CLOSE_OPTIONS.map((option) => (
                <button
                  key={option.reason}
                  onClick={() => startClose(option.reason, option.defaultR)}
                  className="flex-1 rounded-pill border border-hairline py-2 text-sm font-semibold text-ink transition-colors active:bg-ink/5"
                >
                  {option.label}
                </button>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="0.1"
                value={realizedR}
                onChange={(e) => setRealizedR(e.target.value)}
                placeholder="R-Multiple"
                autoFocus
                className="w-24 rounded-pill border border-hairline bg-app-bg px-3.5 py-2 text-sm text-ink focus:border-primary focus:outline-none"
              />
              <button
                onClick={confirmClose}
                disabled={closeAllocation.isPending || realizedR === ""}
                className="flex-1 rounded-pill bg-ink py-2 text-sm font-semibold text-white disabled:opacity-40"
              >
                Close Trade
              </button>
              <button
                onClick={() => setClosing(null)}
                className="rounded-pill px-3 py-2 text-sm font-semibold text-ink-soft"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
