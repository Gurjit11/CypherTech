import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ResponsiveLine } from "@nivo/line";
import { ResponsivePie } from "@nivo/pie";
import { ResponsiveBar } from "@nivo/bar";
import { Link } from "react-router-dom";
import {
  SignIn,
  SignInButton,
  SignOutButton,
  SignedIn,
  SignedOut,
} from "@clerk/clerk-react";
import Section1 from "./Home/Section1";
import Section2 from "./Home/Section2";
import Section3 from "./Home/Section3";
import Footer from "./Home/Footer";


export default function Home() {
  return (
    <div className="bg-gradient-to-r from-white via-white relative z-20 text-gray-900">
      <nav className="flex justify-between sticky top-0 backdrop-blur-xl z-10 items-center py-4 px-8 border-b">
        <a className="text-xl font-bold" href="/">
          Analyser.ai
        </a>
        <div className="space-x-4">
          <Link className="hover:underline" to="#">
            About
          </Link>
          <Link className="hover:underline" to="#">
            Pricing
          </Link>
          <Link className="hover:underline" to="#">
            FAQs
          </Link>
          <Link className="hover:underline" to="#">
            Contact
          </Link>
          <SignedIn>
            <Link to="/home">
              <Button>Home</Button>
            </Link>
            <SignOutButton signOutCallback={() => redirect("/")} />
          </SignedIn>
          <SignedOut>
            <SignInButton
              appearance={{
                elements: {
                  formButtonPrimary:
                    "bg-slate-500 hover:bg-slate-400 text-sm normal-case",
                },
              }}
            />
          </SignedOut>
        </div>
      </nav>
      <header className="text-start flex justify-start items-center px-8 py-16">
        <div className="pl-10">
          <h1 className="text-5xl font-bold ">Artificial Intelligence for</h1>
          <h1 className="text-5xl text-blue-600 font-bold mb-8">
            Customer Review Analytics
          </h1>
          <p className="mb-6 text-xl">
            Convert reviews, chats, surveys, emails, support tickets and other
            textual customer feedback from multiple sources into actionable
            insights and trends.
          </p>
          <div className="flex justify-start items-center space-x-4">
            <Input className="w-1/2" placeholder="Enter your email" />
            <Button>Get Started</Button>
          </div>
        </div>
        <img src="/hero-optimized.svg" alt="Hero" className="w-1/2 h-auto" />
      </header>
      {/* <section className="px-8 py-16">
        <h2 className="text-3xl font-bold text-center mb-8">
          CUSTOMER FEEDBACK ANALYTICS
        </h2>
        <h3 className="text-2xl font-semibold mb-4">
          FreeText AI turns text into actionable insights and trends
        </h3>
        <div className="grid grid-cols-2 gap-8">
          <CurvedlineChart className="w-full h-[300px]" />
          <LabelledpieChart className="w-full h-[300px]" />
        </div>
        <div className="grid grid-cols-3 gap-8 mt-8">
          <BarChart className="w-full h-[300px]" />
          <BarChart className="w-full h-[300px]" />
          <LabelledpieChart className="w-full h-[300px]" />
        </div>
      </section> */}
      <img src="../src/assets/screenshot.svg" />
      <Section2 />
      <Section3 />
      <Section1 />
      <Footer />
    </div>
  );
}

function BarChart(props) {
  return (
    <div {...props}>
      <ResponsiveBar
        data={[
          { name: "Jan", count: 111 },
          { name: "Feb", count: 157 },
          { name: "Mar", count: 129 },
          { name: "Apr", count: 150 },
          { name: "May", count: 119 },
          { name: "Jun", count: 72 },
        ]}
        keys={["count"]}
        indexBy="name"
        margin={{ top: 0, right: 0, bottom: 40, left: 40 }}
        padding={0.3}
        colors={["#2563eb"]}
        axisBottom={{
          tickSize: 0,
          tickPadding: 16,
        }}
        axisLeft={{
          tickSize: 0,
          tickValues: 4,
          tickPadding: 16,
        }}
        gridYValues={4}
        theme={{
          tooltip: {
            chip: {
              borderRadius: "9999px",
            },
            container: {
              fontSize: "12px",
              textTransform: "capitalize",
              borderRadius: "6px",
            },
          },
          grid: {
            line: {
              stroke: "#f3f4f6",
            },
          },
        }}
        tooltipLabel={({ id }) => `${id}`}
        enableLabel={false}
        role="application"
        ariaLabel="A bar chart showing data"
      />
    </div>
  );
}

function CurvedlineChart(props) {
  return (
    <div {...props}>
      <ResponsiveLine
        data={[
          {
            id: "Desktop",
            data: [
              { x: "Jan", y: 43 },
              { x: "Feb", y: 137 },
              { x: "Mar", y: 61 },
              { x: "Apr", y: 145 },
              { x: "May", y: 26 },
              { x: "Jun", y: 154 },
            ],
          },
          {
            id: "Mobile",
            data: [
              { x: "Jan", y: 60 },
              { x: "Feb", y: 48 },
              { x: "Mar", y: 177 },
              { x: "Apr", y: 78 },
              { x: "May", y: 96 },
              { x: "Jun", y: 204 },
            ],
          },
        ]}
        margin={{ top: 10, right: 10, bottom: 40, left: 40 }}
        xScale={{
          type: "point",
        }}
        yScale={{
          type: "linear",
          min: 0,
          max: "auto",
        }}
        curve="monotoneX"
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 0,
          tickPadding: 16,
        }}
        axisLeft={{
          tickSize: 0,
          tickValues: 5,
          tickPadding: 16,
        }}
        colors={["#2563eb", "#e11d48"]}
        pointSize={6}
        useMesh={true}
        gridYValues={6}
        theme={{
          tooltip: {
            chip: {
              borderRadius: "9999px",
            },
            container: {
              fontSize: "12px",
              textTransform: "capitalize",
              borderRadius: "6px",
            },
          },
          grid: {
            line: {
              stroke: "#f3f4f6",
            },
          },
        }}
        role="application"
      />
    </div>
  );
}

function LabelledpieChart(props) {
  return (
    <div {...props}>
      <ResponsivePie
        data={[
          { id: "Jan", value: 111 },
          { id: "Feb", value: 157 },
          { id: "Mar", value: 129 },
          { id: "Apr", value: 150 },
          { id: "May", value: 119 },
          { id: "Jun", value: 72 },
        ]}
        sortByValue
        margin={{ top: 30, right: 50, bottom: 30, left: 50 }}
        innerRadius={0.5}
        padAngle={1}
        cornerRadius={3}
        activeOuterRadiusOffset={2}
        borderWidth={1}
        arcLinkLabelsThickness={1}
        enableArcLabels={false}
        colors={["#2563eb"]}
        theme={{
          tooltip: {
            chip: {
              borderRadius: "9999px",
            },
            container: {
              fontSize: "12px",
              textTransform: "capitalize",
              borderRadius: "6px",
            },
          },
        }}
        role="application"
      />
    </div>
  );
}
