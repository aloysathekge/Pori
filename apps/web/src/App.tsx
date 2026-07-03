import { Sparkles } from "lucide-react";
import { ChatView } from "./components/ChatView";

export function App() {
  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 border-b border-aloy-border px-4 py-3">
        <Sparkles className="h-5 w-5 text-aloy-accent" />
        <span className="font-semibold tracking-tight">Aloy</span>
        <span className="ml-1 text-sm text-aloy-muted">personal &amp; org OS agent</span>
      </header>
      <main className="min-h-0 flex-1">
        <ChatView />
      </main>
    </div>
  );
}
