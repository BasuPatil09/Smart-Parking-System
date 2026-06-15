import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Evaluation from "./pages/Evaluation";

export default function App() {
  const [page, setPage] = useState("dashboard");

  return (
    <>
      <div hidden={page !== "dashboard"}>
        <Dashboard onOpenEvaluation={() => setPage("evaluation")} />
      </div>
      <div hidden={page !== "evaluation"}>
        <Evaluation onBack={() => setPage("dashboard")} />
      </div>
    </>
  );
}
