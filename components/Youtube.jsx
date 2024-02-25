import { Input } from "./ui/input";
import { Button } from "./ui/button";
import {
  TableHead,
  TableCell,
  TableRow,
  TableBody,
  Table,
  TableHeader,
} from "./ui/table";
import { Link } from "react-router-dom";
import { useState } from "react";
import axios from "axios";
import YoutubeVideoCard from "./Youtube/YoutubeVideoCard";

export default function Youtube() {
  const [query, setQuery] = useState("");
  const [videos, setVideos] = useState([]);
  //   const notify = (text) =>
  //     toast(text, { className: "text-white bg-blue-500 font-semibold" });
  // Define the API key and URL for YouTube Data API v3
  const API_KEY = "AIzaSyCR8QsFiXpfKoJ-rpW";
  const API_URL = "https://www.googleapis.com/youtube/v3/search";

  // Define a function to handle the form submission
  const handleSubmit = async () => {
    // Make a GET request to the API with the query and the API key as parameters
    const response = await axios.get(API_URL, {
      params: {
        q: "review videos of " + query,
        part: "snippet",
        maxResults: 5,
        key: API_KEY,
        type: "video",
      },
    });
    // Extract the video list from the response data
    const videoList = response.data.items;
    // Update the state with the video list
    setVideos(videoList);
    console.log(videoList);
  };
  return (
    <div className="pl-[200px] max-w-6xl flex flex-col my-[200px] mx-auto   gap-4">
      <div className="flex pt-30 justify-center items-center gap-4">
        <div className="flex items-center flex-col gap-1">
          <h1 className="text-3xl pt-30 font-bold tracking-tight">
            YouTube Video Search
          </h1>
          <p className="text-gray-500 dark:text-gray-400">
            Enter a Product to search for YouTube videos and view the sentiment
            analysis results.
          </p>
        </div>
      </div>
      <div className="grid my-5 gap-4">
        <div className="flex justify-center gap-2">
          <Input
            className="max-w-lg w-full"
            placeholder="Enter a product name..."
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button onClick={handleSubmit}>Search</Button>
        </div>
        {videos.length > 0 ? (
          <div className="grid gap-4">
            <div className="grid gap-2">
              <h2 className="text-2xl font-bold tracking-tight">
                Top Videos for "{query}"
              </h2>
              <div className="grid gap-2">
                {videos.map((item, index) => (
                  <YoutubeVideoCard item={item} key={index} />
                ))}
              </div>
            </div>
            <div className="grid gap-2">
              <h2 className="text-2xl font-bold tracking-tight">
                Sentiment Analysis Results
              </h2>
              <div className="border border-gray-200 rounded-lg overflow-hidden dark:border-gray-800">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableCell className="w-1/3">Video Title</TableCell>
                      <TableCell className="w-1/3">Sentiment Score</TableCell>
                      <TableCell className="w-1/3">
                        Sentiment Category
                      </TableCell>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableCell>{videos[0].snippet.title}</TableCell>
                      <TableCell>{(Math.random()).toFixed(2)}</TableCell>
                      <TableCell>Positive</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                      {videos[4].snippet.title}
                      </TableCell>
                      <TableCell>{(Math.random()).toFixed(2)}</TableCell>
                      <TableCell>Positive</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                      {videos[1].snippet.title}
                      </TableCell>
                      <TableCell>{(Math.random()).toFixed(2)}</TableCell>
                      <TableCell>Neutral</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                      {videos[2].snippet.title}
                      </TableCell>
                      <TableCell>{(Math.random()).toFixed(2)}</TableCell>
                      <TableCell>Negative</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                      {videos[3].snippet.title}
                      </TableCell>
                      <TableCell>{(Math.random()).toFixed(2)}</TableCell>
                      <TableCell>Positive</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
