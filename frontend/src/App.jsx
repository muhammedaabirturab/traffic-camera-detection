import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Analyze from "./pages/Analyze.jsx";
import AnalysisPage from "./pages/AnalysisPage.jsx";
import History from "./pages/History.jsx";
import Model from "./pages/Model.jsx";
import Rules from "./pages/Rules.jsx";
import About from "./pages/About.jsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/analysis/:id" element={<AnalysisPage />} />
        <Route path="/history" element={<History />} />
        <Route path="/model" element={<Model />} />
        <Route path="/rules" element={<Rules />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </Layout>
  );
}
