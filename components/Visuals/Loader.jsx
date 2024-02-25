import Lottie from "lottie-react";
import Animation_fetch from "../../src/assets/animation2.json";
import React from "react";
import TextTransition, { presets } from "react-text-transition";
const Loader = () => {
  const TEXTS = [
    "Analyzing reviews... This might take a few moments as we gather insights from a vast amount of data.",
    "Sit back and relax! We're using advanced algorithms to uncover valuable information from customer reviews.",
    "Unveiling the voice of the customer... Please wait as we process thousands of opinions.",
    "Getting ready to deliver key takeaways from the reviews. Stay tuned!",
    "Patience is a virtue! The wait will be worth it as we reveal hidden trends and patterns.",
  ];

  const [index, setIndex] = React.useState(0);

  React.useEffect(() => {
    const intervalId = setInterval(
      () => setIndex((index) => index + 1),
      5000 // every 3 seconds
    );
    return () => clearTimeout(intervalId);
  }, []);

  return (
    <div className="p-10 bg-gray-100/10 border shadow-xl rounded-2xl mx-60">
      <Lottie
        animationData={Animation_fetch}
        className="h-[400px] bg-transparent"
      />
      <div className="flex justify-center text-center mt-3 text-xl font-bold">
        <TextTransition springConfig={presets.wobbly}>
          {TEXTS[index % TEXTS.length]}
        </TextTransition>
      </div>
    </div>
  );
};

export default Loader;
