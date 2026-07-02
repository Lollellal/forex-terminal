import { NotebookPen } from "lucide-react";
import { Screen } from "../components/Screen";
import { EmptyState, ErrorState, SkeletonList } from "../components/States";
import { JournalCard } from "../components/JournalCard";
import { useJournal } from "../api/hooks";

export function Journal() {
  const journal = useJournal();

  if (journal.isError) {
    return (
      <Screen title="Journal">
        <ErrorState message={(journal.error as Error).message} />
      </Screen>
    );
  }

  return (
    <Screen title="Journal">
      {journal.isPending ? (
        <SkeletonList count={3} height={180} />
      ) : journal.data && journal.data.length > 0 ? (
        <div className="flex flex-col gap-4">
          {journal.data.map((entry) => (
            <JournalCard key={entry.allocation_id} entry={entry} />
          ))}
        </div>
      ) : (
        <EmptyState icon={NotebookPen} title="Journal ist leer" hint="Trades erscheinen hier automatisch." />
      )}
    </Screen>
  );
}
