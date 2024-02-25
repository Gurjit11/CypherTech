import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import Home from "../components/Home";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import { StateProvider } from "./Provider/StateProvider";
import Dashboard from "../components/Dashboard";
import Entry from "../components/Entry";
import AskAI from "../components/AskAI";
import Youtube from "../components/Youtube";
import ChatAnalysis from "../components/ChatAnalysis";

function App() {
  return (
    <BrowserRouter>
      <div className="main z-[-1]">
        <div className="gradient" />
      </div>
      <StateProvider>
        <Sidebar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/home" element={<Entry />} />
          <Route path="/analytics" element={<Dashboard />} />
          <Route path="/askai" element={<AskAI />} />
          <Route path="/youtube" element={<Youtube />} />
          <Route path="/analysisai" element={<ChatAnalysis />} />

        </Routes>
      </StateProvider>
    </BrowserRouter>
  );
}

export default App;
