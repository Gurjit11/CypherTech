import { useContext, useState } from "react";
import { AvatarImage, AvatarFallback, Avatar } from "./ui/avatar";
import { TextToSpeech } from "tts-react";
import { StateContext } from "../src/Provider/StateProvider";
import { Link, useNavigate } from "react-router-dom";
import { UserButton, useUser } from "@clerk/clerk-react";
import { dark } from "@clerk/themes";

export default function Sidebar() {
  const navigate = useNavigate();
  const { isSignedIn, user, isLoaded } = useUser();
  const { selectedTab, setSelectedTab } = useContext(StateContext);
  const handleTabClick = (tabName) => {
    setSelectedTab(tabName);
    navigate(`/${tabName.toLowerCase()}`);
  };

  function noRefCheck() {}

  return (
    <nav className=" text-black top-0 fixed w-48 space-y-6 py-5 px-3 rounded-xl">
      <Link to="/">
        <h1 className="text-3xl font-bold">Analyser.ai</h1>
      </Link>
      <TextToSpeech
        align="horizontal"
        allowMuting
        markBackgroundColor="#55AD66"
        markColor="white"
        // lang="hi"
        markTextAsSpoken
        onBoundary={function noRefCheck() {}}
        onEnd={function noRefCheck() {}}
        onError={function noRefCheck() {}}
        onPause={function noRefCheck() {}}
        onPitchChange={function noRefCheck() {}}
        onRateChange={function noRefCheck() {}}
        onStart={function noRefCheck() {}}
        onVolumeChange={function noRefCheck() {}}
        position="topCenter"
        rate={1}
        size="medium"
        volume={1}
      >
        <div className="space-y-2">
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Home" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Home")}
          >
            <HomeIcon className="w-6 h-6" />
            <span>Home</span>
          </div>
          {/* <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Sprints" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Sprints")}
          >
            <CheckSquareIcon className="w-6 h-6" />
            <span>Sprints</span>
          </div>
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Calendar" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Calendar")}
          >
            <CalendarIcon className="w-6 h-6" />
            <span>Timeline</span>
          </div> */}
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "AskAI" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("AskAI")}
          >
            <LightbulbIcon className="w-6 h-6" />
            <span>Ask AI</span>
          </div>
          {/* <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Invite" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Invite")}
          >
            <UserPlusIcon className="w-6 h-6" />
            <span>Invite</span>
          </div> */}
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Analytics" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Analytics")}
          >
            <BarChartIcon className="w-6 h-6" />
            <span>Analytics</span>
          </div>
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "youtube" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("youtube")}
          >
            <VideoIcon className="w-6 h-6" />
            <span>Youtube AI</span>
          </div>
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "analysisai" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("analysisai")}
          >
            <LightbulbIcon className="w-6 h-6" />
            <span>Analysis AI Chat</span>
          </div>
          {/* <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Chat" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Chat")}
          >
            <MessageSquareIcon className="w-6 h-6" />
            <span>Chat</span>
          </div>
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Meet" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Meet")}
          >
            <VideoIcon className="w-6 h-6" />
            <span>Team Meet</span>
          </div>
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "LiveCode" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("LiveCode")}
          >
            <CodeIcon className="w-6 h-6" />
            <span>Live Code</span>
          </div>
          <div
            className={`flex cursor-pointer items-center space-x-3 text-sm font-medium p-2 rounded-md ${
              selectedTab === "Feedback" ? "bg-black text-white" : ""
            }`}
            onClick={() => handleTabClick("Feedback")}
          >
            <MessageCircleIcon className="w-6 h-6" />
            <span>Feedback API</span>
          </div> */}
        </div>
        <div className="flex py-3 px-3 rounded-full bg-white shadow-lg bottom-10 fixed justify-center items-center">
          <UserButton
            appearance={{
              baseTheme: dark,
            }}
          />
          <span className="ml-3">{user?.firstName}</span>
        </div>
      </TextToSpeech>
    </nav>
  );
}

function BarChartIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" x2="12" y1="20" y2="10" />
      <line x1="18" x2="18" y1="20" y2="4" />
      <line x1="6" x2="6" y1="20" y2="16" />
    </svg>
  );
}

function CalendarIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
      <line x1="16" x2="16" y1="2" y2="6" />
      <line x1="8" x2="8" y1="2" y2="6" />
      <line x1="3" x2="21" y1="10" y2="10" />
    </svg>
  );
}

function CheckSquareIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="9 11 12 14 22 4" />
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    </svg>
  );
}

function CodeIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

function HomeIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function LightbulbIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
      <path d="M9 18h6" />
      <path d="M10 22h4" />
    </svg>
  );
}

function LogOutIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" x2="9" y1="12" y2="12" />
    </svg>
  );
}

function MessageCircleIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z" />
    </svg>
  );
}

function MessageSquareIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function UserPlusIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <line x1="19" x2="19" y1="8" y2="14" />
      <line x1="22" x2="16" y1="11" y2="11" />
    </svg>
  );
}

function VideoIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m22 8-6 4 6 4V8Z" />
      <rect width="14" height="12" x="2" y="6" rx="2" ry="2" />
    </svg>
  );
}
