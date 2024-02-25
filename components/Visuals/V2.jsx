import React from "react";
import Doublebar from "./Doublebar";
import Wordcloud from "./Wordcloud";

const V2 = ({formatted_keyword}) => {
  return (
    <div className="flex p-3 m-3 gap-6">
      <div className="w-1/2 bg-white/30 rounded-lg p-3 flex flex-col  justify-center items-center shadow-lg ">
        <h1 className="text-xl font-bold items-start w-full ml-9">
          Sentiment Analysis
        </h1>
        <Doublebar />
      </div>
      <div className="w-1/2 bg-white/30 rounded-lg p-3 flex flex-col  justify-center items-center shadow-lg ">
        <h1 className="text-xl font-bold items-start w-full ml-9">
          Word Cloud{" "}
        </h1>
        <Wordcloud formatted_keyword={formatted_keyword} />
      </div>
    </div>
  );
};

export default V2;
