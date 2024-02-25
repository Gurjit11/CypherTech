import { ResponsiveBar } from "@nivo/bar";
import { ResponsivePie } from "@nivo/pie";
import { Button } from "../ui/button";

function convertEmotionCountsToObject(emotionCounts) {
  return Object.entries(emotionCounts).map(([emotion, count]) => ({
    name: emotion,
    count: count,
  }));
}

function countSentiment(sentimentData) {
  let positiveCount = 0;
  let negativeCount = 0;

  sentimentData.forEach((result) => {
    if (result.label === "POSITIVE") {
      positiveCount++;
    } else if (result.label === "NEGATIVE") {
      negativeCount++;
    }
  });

  return {
    positiveCount: positiveCount,
    negativeCount: negativeCount,
  };
}
let totalEmotions=0;
function countEmotions(emotionData) {
  let emotionCounts = {};

  emotionData.forEach((result) => {
    let label = result.label;
    if (emotionCounts[label]) {
      emotionCounts[label]++;
    } else {
      emotionCounts[label] = 1;
    }
    totalEmotions++;
  });

  return emotionCounts;
}

export default function V4({ emotiondata, sentimentdata }) {
  let counte = countEmotions(emotiondata);
  let counts = countSentiment(sentimentdata);
  let convertedData = convertEmotionCountsToObject(counte);
  console.log(counts,counte,convertedData);
  return (
    <div className="p-5">
      <div className="bg-white/40 p-6 mx-10 w-full shadow-lg rounded-lg  mx-auto">
        <div className="flex  justify-between items-center mb-4">
          <h1 className="text-xl font-semibold">
            {/* SAMSUNG WIRED HEADSET EARPHONE FOR 3.5MM JACK - WHITE */}
          </h1>
          <span className="text-sm font-medium bg-gray-200 px-3 py-1 rounded-full">
            100 records
          </span>
        </div>
        <div className="flex gap-8">
          <div className="flex flex-col">
            <LabelledpieChart counts={counts} className="w-[500px] h-[300px]" />
            <div className="flex justify-around mt-4">
              <span className="text-sm text-red-500">
                Negative:{" "}
                {(
                  (counts.negativeCount /
                    (counts.positiveCount + counts.negativeCount)) *
                  100
                ).toFixed(2)}
                %
              </span>
              <span className="text-sm text-green-500">
                Positive:{" "}
                {(
                  (counts.positiveCount /
                    (counts.positiveCount + counts.negativeCount)) *
                  100
                ).toFixed(2)}
                %
              </span>
            </div>
          </div>
          <div className="flex w-1/3">
            <div className="flex flex-col pr-5  mb-4"></div>
            <div className="flex flex-col min-w-[200px] pr-5">
              <h2 className="text-lg font-semibold">Emotions</h2>
              <ul className="list-disc pl-5">
                {Object.entries(counte).map(([emotion, count]) => (
                  <li key={emotion}>
                    {emotion.toUpperCase()} : {(count/totalEmotions *100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
            <BarChart className="w-[350px] h-[300px]" convertedData={convertedData} />
          </div>
        </div>
      </div>
    </div>
  );
}

function BarChart({convertedData,className}) {
  return (
    <div className={className}>
      <ResponsiveBar
        data={convertedData}
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

function LabelledpieChart({ className, counts }) {
  return (
    <div className={className}>
      <ResponsivePie
        data={[
          { id: "Positive", value: counts.positiveCount },
          { id: "Negative", value: counts.negativeCount },
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
        colors={["#2563eb", "#f87171", "#d1d5db"]}
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
