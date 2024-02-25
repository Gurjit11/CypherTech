import { Button } from "../ui/button";
import Lottie from "lottie-react";
import animation from "../../src/assets/animation.json";
export default function Section2() {
  return (
    <div className="bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 flex justify-between items-center">
        <div className="max-w-lg">
          <p className="text-sm font-semibold uppercase tracking-wide text-gray-500">
            Custom Report Generation
          </p>
          <h2 className="mt-2 text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl">
            Uncover Hidden Insights
          </h2>
          <p className="mt-4 text-lg text-gray-500">
            Discover large themes that impact customer experience and drill down
            for data driven actionable insights on root causes with our dynamic
            analytics engine.
          </p>
          <p className="mt-2 text-lg text-gray-500">
            Spend less time trying to generate reports, and more time listening
            to your customers.
          </p>
          <Button className="mt-6" variant="default">
            Get Started
          </Button>
        </div>
        <div className="flex-0">
          {/* <img alt="Analytics illustration" className="h-63" src={analytics} /> */}
          <Lottie animationData={animation} />
        </div>
      </div>
    </div>
  );
}
