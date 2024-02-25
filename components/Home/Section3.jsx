import { Button } from "../ui/button";
import Lottie from "lottie-react";
import animation2 from "../../src/assets/animation2.json";
export default function Section3() {
  return (
    <div className="bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 flex justify-between items-center">
        <div className="flex-shrink-0">
          {/* <img alt="Analytics illustration" className="h-64 w-auto" src="" /> */}
          <Lottie animationData={animation2} />
        </div>
        <div className="max-w-lg">
          <p className="text-sm font-semibold uppercase tracking-wide text-gray-500">
            TRANSPARENT AI
          </p>
          <h2 className="mt-2 text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl">
            AI-Driven Feedback Analysis
          </h2>
          <p className="mt-4 text-lg text-gray-500">
            Founded by CypherTech, leveraging years of experience in natural
            language processing and neural networks, Analyser.AI automatically
            categorizes the feedback into fine grained categories cut by
            sentiment.
          </p>
          <p className="mt-2 text-lg text-gray-500">
            Explanations given by Analyser.AI will help you internalize the
            feedback, and commit to the actions required to grow your business.{" "}
          </p>
          <Button className="mt-6" variant="default">
            Get Started
          </Button>
        </div>
      </div>
    </div>
  );
}
