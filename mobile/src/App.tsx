import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { BottomNav } from "./components/BottomNav";
import { Home } from "./screens/Home";
import { ActiveTrades } from "./screens/ActiveTrades";
import { Journal } from "./screens/Journal";
import { Empire } from "./screens/Empire";
import { WeeklyReports } from "./screens/WeeklyReports";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/trades" element={<ActiveTrades />} />
          <Route path="/journal" element={<Journal />} />
          <Route path="/empire" element={<Empire />} />
          <Route path="/reports" element={<WeeklyReports />} />
        </Routes>
        <BottomNav />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App
