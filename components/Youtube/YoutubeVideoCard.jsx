import { YoutubeTranscript } from "youtube-transcript";
import React from "react";
import { Link } from "react-router-dom";
import { Button } from "../ui/button";
function getTimeDifference(timestamp) {
  const currentTime = new Date();
  const previousTime = new Date(timestamp);
  const timeDifference = currentTime - previousTime;

  const msPerDay = 1000 * 60 * 60 * 24;
  const msPerMonth = msPerDay * 30; // Approximating a month as 30 days

  const daysAgo = Math.floor(timeDifference / msPerDay);
  const monthsAgo = Math.floor(timeDifference / msPerMonth);

  if (monthsAgo >= 1) {
    return `${monthsAgo} months ago`;
  } else {
    return `${daysAgo} days ago`;
  }
}

const timestamp = "2023-05-04T15:59:08Z";
const formattedTime = getTimeDifference(timestamp);
console.log(formattedTime);

const YoutubeVideoCard = ({ item }) => {
  // YoutubeTranscript.fetchTranscript("8C_kHJ5YEiA").then(console.log);

  return (
    <div className="flex items-start p-1 border rounded-lg gap-4 relative">
      <img
        alt="Thumbnail"
        className="aspect-video rounded-lg object-cover"
        height={94}
        src={item.snippet.thumbnails.medium.url}
        width={168}
      />
      <div className="text-sm">
        <div className="font-medium line-clamp-2">{item.snippet.title}</div>
        <div className="text-xs text-gray-500 line-clamp-1 dark:text-gray-400">
          {item.snippet.channelTitle}
        </div>
        <div className="text-xs text-gray-500 line-clamp-1 dark:text-gray-400">
          {getTimeDifference(item.snippet.publishedAt)}
        </div>
      </div>
      <Link
        className="h-[36px] px-4 ml-auto"
        to={`https://www.youtube.com/watch?v=${item.id.videoId}`}
      >
        <Button variant="outline">Play</Button>
      </Link>
    </div>
  );
};

export default YoutubeVideoCard;
