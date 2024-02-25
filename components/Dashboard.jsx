import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Tabs } from "./ui/tabs";
import {
  CardTitle,
  CardDescription,
  CardHeader,
  CardContent,
  Card,
} from "./ui/card";
import { ResponsiveBar } from "@nivo/bar";
import { ScrollArea } from "./ui/scroll-area";
import { ResponsiveScatterPlot } from "@nivo/scatterplot";
import { Link } from "react-router-dom";
import { Progress } from "./ui/progress";
import V4 from "./Visuals/V4";
import V2 from "./Visuals/V2";
import Loader from "./Visuals/Loader";
import axios from "axios";
import { useState } from "react";
import { Input } from "./ui/input";
import {
  TableHead,
  TableRow,
  TableHeader,
  TableCell,
  TableBody,
  Table,
} from "./ui/table";
import { Rate } from "antd";
import { ResponsivePie } from "@nivo/pie";
import { ResponsiveLine } from "@nivo/line";
import ReviewCard from "./Visuals/ReviewCard";
import csvDownload from 'json-to-csv-export'



const dataToConvert = {
  data: [
    { label: "POSITIVE", score: 0.9985387325286865 },
    { label: "NEGATIVE", score: 0.9994971752166748 },
    { label: "NEGATIVE", score: 0.9986660480499268 },
    { label: "NEGATIVE", score: 0.9994903802871704 },
    { label: "NEGATIVE", score: 0.9995039701461792 },
    { label: "NEGATIVE", score: 0.9995081424713135 },
    { label: "NEGATIVE", score: 0.9994896650314331 },
    { label: "NEGATIVE", score: 0.9995051622390747 },
    { label: "NEGATIVE", score: 0.9995003938674927 },
    { label: "NEGATIVE", score: 0.9994901418685913 },
    { label: "POSITIVE", score: 0.9989123344421387 },
    { label: "POSITIVE", score: 0.9987847208976746 },
    { label: "POSITIVE", score: 0.9989033937454224 },
    { label: "POSITIVE", score: 0.9988586902618408 },
    { label: "NEGATIVE", score: 0.9994233846664429 },
    { label: "NEGATIVE", score: 0.9995104074478149 },
    { label: "NEGATIVE", score: 0.9995107650756836 },
    { label: "NEGATIVE", score: 0.9961118102073669 },
  ],
  filename: 'analysis_report',
  delimiter: ',',
  headers: ['Result', "Score"]
}

function smoothScrollTo(id) {
  const element = document.getElementById(id);
  if (element) {
    element.scrollIntoView({ behavior: "smooth" });
  }
}

export default function Dashboard() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [dataloaded, setDataloaded] = useState(true);
  const [keyworddata, setKeyworddata] = useState({
    keywords: [
      ["flossers", 0.5597],
      ["floss", 0.5461],
      ["flossing", 0.5218],
      ["toothpick", 0.4285],
      ["dental", 0.4054],
    ],
  });


  const [sentimentdata, setSentimentdata] = useState([
    { label: "POSITIVE", score: 0.9985387325286865 },
    { label: "NEGATIVE", score: 0.9994971752166748 },
    { label: "NEGATIVE", score: 0.9986660480499268 },
    { label: "NEGATIVE", score: 0.9994903802871704 },
    { label: "NEGATIVE", score: 0.9995039701461792 },
    { label: "NEGATIVE", score: 0.9995081424713135 },
    { label: "NEGATIVE", score: 0.9994896650314331 },
    { label: "NEGATIVE", score: 0.9995051622390747 },
    { label: "NEGATIVE", score: 0.9995003938674927 },
    { label: "NEGATIVE", score: 0.9994901418685913 },
    { label: "POSITIVE", score: 0.9989123344421387 },
    { label: "POSITIVE", score: 0.9987847208976746 },
    { label: "POSITIVE", score: 0.9989033937454224 },
    { label: "POSITIVE", score: 0.9988586902618408 },
    { label: "NEGATIVE", score: 0.9994233846664429 },
    { label: "NEGATIVE", score: 0.9995104074478149 },
    { label: "NEGATIVE", score: 0.9995107650756836 },
    { label: "NEGATIVE", score: 0.9961118102073669 },
  ]);
  const [emotiondata, setEmotiondata] = useState([
    { label: "love", score: 0.8744370341300964 },
    { label: "sadness", score: 0.7720382809638977 },
    { label: "sadness", score: 0.7553319334983826 },
    { label: "sadness", score: 0.9972155094146729 },
    { label: "sadness", score: 0.7809393405914307 },
    { label: "joy", score: 0.9743551015853882 },
    { label: "anger", score: 0.7660535573959351 },
    { label: "joy", score: 0.8289968371391296 },
    { label: "joy", score: 0.9988948702812195 },
    { label: "anger", score: 0.9456976652145386 },
    { label: "joy", score: 0.9955364465713501 },
    { label: "joy", score: 0.6670888662338257 },
    { label: "love", score: 0.9949658513069153 },
    { label: "joy", score: 0.5473855137825012 },
    { label: "sadness", score: 0.9839763641357422 },
    { label: "sadness", score: 0.9948090314865112 },
    { label: "sadness", score: 0.993830144405365 },
    { label: "joy", score: 0.9958131909370422 },
  ]);

  const [obj, setObj] = useState({
    avg_rating: 2.22,
    csat_rate: 22.22,
    rating_counts: [{ 1: 9 }, { 2: 4 }, { 3: 1 }, { 4: 0 }, { 5: 4 }],
    rating_trend: [
      ["2023-06-08", 5.0],
      ["2023-06-18", 2.0],
      ["2023-06-23", 5.0],
      ["2023-07-04", 2.0],
      ["2023-07-13", 1.0],
      ["2023-07-19", 5.0],
      ["2023-07-25", 1.0],
      ["2023-08-03", 1.0],
      ["2023-08-05", 2.0],
      ["2023-09-07", 2.0],
      ["2023-09-12", 1.0],
      ["2023-09-20", 1.0],
      ["2023-10-01", 1.0],
      ["2023-10-12", 5.0],
      ["2023-10-30", 1.0],
      ["2023-11-14", 3.0],
      ["2023-11-26", 1.0],
      ["2024-02-15", 1.0],
    ],
    total_reviews: 18,
    total_verified_count: 11,
  });
  const data = obj.rating_trend.reduce((acc, [date]) => {
    const month = new Date(date).getMonth(); // Extract month (0-indexed)
    acc[month] = (acc[month] || 0) + 1; // Initialize and increment count
    return acc;
  }, {});

  const monthNames = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  const formattedData2 = Object.entries(data).map(([monthIndex, count]) => ({
    x: monthNames[monthIndex],
    y: count,
  }));
  console.log(formattedData2);

  const fetchData = () => {
    setLoading(true);
    setDataloaded(false);

    let data = JSON.stringify({
      url: url,
    });

    let config = {
      method: "post",
      maxBodyLength: Infinity,
      url: "http://127.0.0.1:5000/scraping",
      headers: {
        "Content-Type": "application/json",
      },
      data: data,
    };

    axios
      .request(config)
      .then((response) => {
        console.log(response.data);
        setObj(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.log(error);
        setLoading(false);
      });
    fetchEmotion();
    fetchSentiment();
    fetchKeywords();
  };

  const fetchKeywords = () => {
    let data = JSON.stringify({
      url: "https://www.amazon.com/Dental-Floss-Packs-Smooth-Gentle/dp/B0C3WDM9PY",
    });

    let config = {
      method: "post",
      maxBodyLength: Infinity,
      url: "http://127.0.0.1:5000/extract_keywords",
      headers: {
        "Content-Type": "application/json",
      },
      data: data,
    };

    axios
      .request(config)
      .then((response) => {
        console.log(JSON.stringify(response.data));
      })
      .catch((error) => {
        console.log(error);
      });
  };

  const fetchEmotion = () => {
    let data = JSON.stringify({
      url: "https://www.amazon.com/Dental-Floss-Packs-Smooth-Gentle/dp/B0C3WDM9PY",
    });

    let config = {
      method: "post",
      maxBodyLength: Infinity,
      url: "http://127.0.0.1:5000/emotion-analysis",
      headers: {
        "Content-Type": "application/json",
      },
      data: data,
    };

    axios
      .request(config)
      .then((response) => {
        console.log(JSON.stringify(response.data));
      })
      .catch((error) => {
        console.log(error);
      });
  };

  const fetchSentiment = () => {
    let data = JSON.stringify({
      url: "https://www.amazon.com/Dental-Floss-Packs-Smooth-Gentle/dp/B0C3WDM9PY",
    });

    let config = {
      method: "post",
      maxBodyLength: Infinity,
      url: "http://127.0.0.1:5000/sentiment-analysis",
      headers: {
        "Content-Type": "application/json",
      },
      data: data,
    };

    axios
      .request(config)
      .then((response) => {
        console.log(JSON.stringify(response.data));
      })
      .catch((error) => {
        console.log(error);
      });
  };
  const formatted_keyword = keyworddata.keywords.map(item => ({ value: item[0], count: item[1] }));
  console.log(formatted_keyword);
  return (
    <div className="ml-[200px] pr-10 mb-[100px] scroll-smooth">
      <div className="flex flex-col">
        {/* <div className="flex flex-col  text-white justify-between p-5 m-5 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 ">
          <div className="flex items-center justify-between space-x-4  ">
            <div className="flex gap-3">
              <h1 className="text-xl font-semibold">Earbud Headphones Q5</h1>
              <Badge variant="secondary">Total ASIN: 18</Badge>
              <Badge variant="secondary">Including variations 0</Badge>
            </div>
            <div className="flex items-center space-x-2">
              <ShareIcon className="text-white" />
              <DownloadIcon className="text-white" />
              <BellIcon className="text-white" />
              <HelpCircleIcon className="text-white" />
            </div>
          </div>
          <div className="flex items-center justify-between p-4">
            <div>
              <div className="text-sm">Number of reviews analyzed</div>
              <div className="text-3xl font-bold">20,295</div>
            </div>
            <div className="text-sm">Last Updated: 2023.03.08</div>
          </div>
        </div> */}
        <div className="my-8">
          <h1 className="text-3xl font-bold mb-6">Analysis Report</h1>
          <div className="flex gap-4 mb-4">
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter product url..."
            />
            <Button onClick={fetchData}>Search</Button>
          </div>
          <div className="flex gap-4 mb-4">
            <Badge>Source: Amazon US</Badge>
            <Button
            onClick={() => csvDownload(dataToConvert)} className="flex items-center" variant="ghost">
              <DownloadIcon className="w-5 h-5 mr-2" />
              Download{"\n          "}
            </Button>
            {/* <Button className="flex items-center" variant="ghost">
              <UploadIcon className="w-5 h-5 mr-2" />
              Upload{"\n          "}
            </Button> */}
            <Button variant="ghost">Filters</Button>
            <Button variant="ghost">All Time</Button>
          </div>
        </div>
        {url === "" || dataloaded ? null : loading ? (
          <Loader />
        ) : (
          <>
            <div className="  my-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <Card className="w-full">
                  <CardHeader>
                    <CardTitle>Total Reviews</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{obj.total_reviews}</p>
                    <p className="text-sm text-gray-500">
                      Verified: {obj.total_verified_count}
                    </p>
                  </CardContent>
                </Card>
                <Card className="w-full">
                  <CardHeader>
                    <CardTitle>Average Review Rating</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{obj.avg_rating}</p>
                  </CardContent>
                </Card>
                <Card className="w-full">
                  <CardHeader>
                    <CardTitle>CSAT rating</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{obj.csat_rate}</p>
                  </CardContent>
                </Card>
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>PRODUCTS</TableHead>
                    <TableHead>CHANNEL</TableHead>
                    <TableHead>RECORDS</TableHead>
                    <TableHead>AVERAGE</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>
                      {/* Samsung Wired Headset Earphone for 3.5mm Jack - White */}
                    </TableCell>
                    <TableCell>Amazon US</TableCell>
                    <TableCell>{obj.total_reviews}</TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        {obj.avg_rating}
                        <Rate
                          className="ml-2"
                          allowHalf
                          defaultValue={obj.avg_rating}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
              <div className="flex p-3 m-3 gap-6">
                <div className="w-1/2 bg-white/30 rounded-lg p-3 flex flex-col  justify-center items-center shadow-lg ">
                  <h1 className="text-xl font-bold items-start w-full ml-9">
                    Ratings Distribution{" "}
                  </h1>
                  <LabelledpieChart1
                    className="w-[500px] h-[300px]"
                    obj={obj}
                  />
                </div>
                {/* <BarChart className="w-[600px] h-[300px]" /> */}
                <div className="w-1/2 bg-white/30 rounded-lg p-3 flex flex-col  justify-center items-center shadow-lg ">
                  <h1 className="text-xl font-bold items-start w-full ml-9">
                    Monthly Rating Distribution{" "}
                  </h1>
                  <LineChart1
                    className="w-[500px] h-[300px]"
                    formattedData2={formattedData2}
                  />
                </div>
              </div>
            </div>
            <V2 formatted_keyword={formatted_keyword} />
            <V4 emotiondata={emotiondata} sentimentdata={sentimentdata} />
          

            {/* <div className="flex flex-col px-5 space-y-4 ">
              <div className="flex space-x-2 justify-evenly p-4 sticky z-10 bg-white/90 backdrop-blur-sm top-0">
                <Button
                  onClick={() => smoothScrollTo("cust-profile")}
                  variant="ghost"
                >
                  Customer Profile
                </Button>
                <Button onClick={() => smoothScrollTo("usage")} variant="ghost">
                  Usage Scenario
                </Button>
                <Button
                  onClick={() => smoothScrollTo("cust-profile")}
                  variant="ghost"
                >
                  Rating Optimization
                </Button>
                <Button
                  onClick={() => smoothScrollTo("cust-profile")}
                  variant="ghost"
                >
                  Customer Sentiment
                </Button>
                <Button
                  onClick={() => smoothScrollTo("cust-profile")}
                  variant="ghost"
                >
                  Buyers Motivation
                </Button>
                <Button
                  onClick={() => smoothScrollTo("cust-profile")}
                  variant="ghost"
                >
                  Customer Expectations
                </Button>
              </div>
              
              
              
              <Card id="buyers">
                <CardHeader>
                  <CardTitle>Buyers Motivation</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-48 w-full">
                    <ul className="space-y-2">
                      <li className="flex justify-between">
                        <span>Feel for the Price</span>
                        <span>30.9%</span>
                      </li>
                    </ul>
                  </ScrollArea>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Customer Expectations</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-48 w-full">
                    <ul className="space-y-2">
                      <li className="flex justify-between">
                        <span>Durability</span>
                        <span>20.0%</span>
                      </li>
                    </ul>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div> */}
          </>
        )}
      </div>
    </div>
  );
}

function LineChart1({ className, formattedData2 }) {
  return (
    <div className={className}>
      <ResponsiveLine
        data={[
          {
            id: "US",
            color: "hsl(240, 100%, 50%)",
            data: formattedData2,
          },
        ]}
        margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
        xScale={{ type: "point" }}
        yScale={{
          type: "linear",
          min: "auto",
          max: "auto",
          stacked: true,
          reverse: false,
        }}
        yFormat=" >-.2f"
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: "transportation",
          legendOffset: 36,
          legendPosition: "middle",
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: "count",
          legendOffset: -40,
          legendPosition: "middle",
        }}
        pointSize={10}
        pointColor={{ theme: "background" }}
        pointBorderWidth={2}
        pointBorderColor={{ from: "serieColor" }}
        pointLabelYOffset={-12}
        useMesh={true}
        legends={[
          {
            anchor: "bottom-right",
            direction: "column",
            justify: false,
            translateX: 100,
            translateY: 0,
            itemsSpacing: 0,
            itemDirection: "left-to-right",
            itemWidth: 80,
            itemHeight: 20,
            itemOpacity: 0.75,
            symbolSize: 12,
            symbolShape: "circle",
            symbolBorderColor: "rgba(0, 0, 0, .5)",
            effects: [
              {
                on: "hover",
                style: {
                  itemBackground: "rgba(0, 0, 0, .03)",
                  itemOpacity: 1,
                },
              },
            ],
          },
        ]}
      />
    </div>
  );
}

function BarChart1(props) {
  return (
    <div {...props}>
      <ResponsiveBar
        data={formattedData}
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

function LabelledpieChart1({ className, obj }) {
  return (
    <div className={className}>
      <ResponsivePie
        data={obj.rating_counts.map((rating) => {
          const [key, value] = Object.entries(rating)[0]; // Destructure key and value
          return { id: key, value }; // Return object with id and value
        })}
        sortByValue
        margin={{ top: 30, right: 50, bottom: 30, left: 50 }}
        innerRadius={0.5}
        padAngle={1}
        cornerRadius={3}
        activeOuterRadiusOffset={2}
        borderWidth={1}
        arcLinkLabelsThickness={1}
        enableArcLabels={false}
        colors={["#2563eb", "#f87171", "#d1d5db", "#007171", "#febe10"]}
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

function DownloadIcon(props) {
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
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
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

function StarIcon(props) {
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
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function UploadIcon(props) {
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
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
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

function BellIcon(props) {
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
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
      <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    </svg>
  );
}

function DotChart(props) {
  return (
    <div {...props}>
      <ResponsiveScatterPlot
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
        xScale={{ type: "point" }}
        yScale={{ type: "linear" }}
        blendMode="multiply"
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

function HelpCircleIcon(props) {
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
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <path d="M12 17h.01" />
    </svg>
  );
}

function MenuIcon(props) {
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
      <line x1="4" x2="20" y1="12" y2="12" />
      <line x1="4" x2="20" y1="6" y2="6" />
      <line x1="4" x2="20" y1="18" y2="18" />
    </svg>
  );
}

function ShareIcon(props) {
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
      <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
      <polyline points="16 6 12 2 8 6" />
      <line x1="12" x2="12" y1="2" y2="15" />
    </svg>
  );
}
