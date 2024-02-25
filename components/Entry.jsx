import { Button } from "./ui/button";
import { AvatarImage, AvatarFallback, Avatar } from "./ui/avatar";
import { Input } from "./ui/input";
import {
  CardTitle,
  CardHeader,
  CardDescription,
  CardContent,
  CardFooter,
  Card,
} from "./ui/card";
import { Badge } from "./ui/badge";
import {
  Award,
  CameraIcon,
  CupSoda,
  MedalIcon,
  PlusIcon,
  Trophy,
  VideoIcon,
} from "lucide-react";
import { Rate } from "antd";
import { Link } from "react-router-dom";

export default function Entry() {
  return (
    <div className="bg-white ml-[200px]">
      <div className="p-4">
        <div className="flex gap-4">
          {/* <div className="w-1/4">
            <nav className="flex flex-col space-y-2">
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                Consumer Insight
              </Button>
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                Market Insight
              </Button>
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                Tools
              </Button>
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                Add member
              </Button>
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                Language
              </Button>
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                Notifications
              </Button>
              <Button className="text-blue-600 bg-transparent hover:bg-blue-50">
                User Center
              </Button>
            </nav>
          </div> */}
          <div className="pt-10">
            <div className="mb-4">
              <div className="flex flex-col justify-between items-center">
                <h2 className="text-2xl font-bold">
                  Search any keywords/Asin to Get Consumer Insight
                </h2>
                <div className="flex items-center space-x-2 py-5">
                  <Input
                    className="border rounded-md w-[300px]"
                    placeholder="Powered Toothbrush"
                  />
                  <Button>Search</Button>
                </div>
              </div>
              <div className="flex justify-evenly mx-10 mt-2">
                <Button variant="outline">Customer Profile</Button>
                <Button variant="outline">Usage Scenario</Button>
                <Button variant="outline">Rating Optimization</Button>
                <Button variant="outline">Customer Sentiment</Button>
                <Button variant="outline">Customer Expectation</Button>
                <Button variant="outline">Purchase Motivations</Button>
              </div>
            </div>
            <div className="flex gap-4 mb-4">
              <Card className="bg-gradient-to-br from-white via-slate-300 to-gray-600">
                <CardHeader>
                  <CardTitle>
                  Analyzing Oral Care Reviews
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-black">
                  Extracting insights from customer feedback on oral hygiene products, spotting trends and opportunities for improvemen
                  </CardDescription>
                </CardContent>
                <CardFooter>
                  <Button variant="ghost">Get Started</Button>
                </CardFooter>
              </Card>
              <Card className="bg-gradient-to-br from-white via-slate-300 to-gray-600">
                <CardHeader>
                  <CardTitle>Evaluating Competitors in Oral Care
</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-black">
                  Digging into customer reviews to identify gaps in competitors' products, guiding optimization strategies
                  </CardDescription>
                </CardContent>
                <CardFooter>
                  <Button variant="ghost">Get Started</Button>
                </CardFooter>
              </Card>
              <Card className="bg-gradient-to-br from-white via-slate-300 to-gray-600">
                <CardHeader>
                  <CardTitle>Optimizing Oral Care Listings</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-black">
                  Leveraging customer review analysis to enhance product listings and connect with the oral care market
                  </CardDescription>
                </CardContent>
                <CardFooter>
                  <Button variant="ghost">Get Started</Button>
                </CardFooter>
              </Card>
            </div>
            <div className="flex my-5 justify-between">
              <h3 className="text-xl font-bold mb-2">Create Your Own Report</h3>
              <div className="">
                <Button className="mr-3">
                  {" "}
                  <PlusIcon className="text-sm mr-2" /> Create Report
                </Button>
                <Button variant="outline">
                  <VideoIcon className="mr-2 text-xs" /> Learn
                </Button>
              </div>
            </div>
            <div className="mb-4">
              <div className="grid grid-cols-4 gap-4">
                <Card className="bg-[url('https://m.media-amazon.com/images/W/MEDIAX_849526-T3/images/I/31MB4ZROXdL.jpg')] bg-cover h-[380px] ">
                  <div className="h-[250px]"></div>
                  <CardContent>
                    <h4 className="text-lg text-black font-semibold mt-2">
                      Tongue Cleaner
                    </h4>
                    <div className="flex text-black items-center mt-1">
                      <Rate allowHalf disabled defaultValue={4.8} />
                      <span className="font-bold">4.5</span>
                    </div>
                    <Button className="mt-2  w-full">
                      <Link to="/analytics">See Demo Report</Link>
                    </Button>
                  </CardContent>
                </Card>
                <Card className="bg-[url('https://m.media-amazon.com/images/W/MEDIAX_849526-T3/images/I/51lT0ymbceL.SX679.jpg')] bg-cover h-[380px]">
                  <div className="h-[250px]"></div>
                  <CardContent>
                    <h4 className="text-lg text-black font-semibold mt-2">
                      Electric Toothbrush
                    </h4>
                    <div className="flex text-black items-center mt-1">
                      <Rate allowHalf disabled defaultValue={4.8} />
                      <span className="font-bold">4.5</span>
                    </div>
                    <Button className="mt-2  w-full">See Demo Report</Button>
                  </CardContent>
                </Card>
                <Card className="bg-[url('https://m.media-amazon.com/images/W/MEDIAX_849526-T3/images/I/41MzcfeNyyL.jpg')] bg-cover h-[380px]">
                  <div className="h-[250px]"></div>
                  <CardContent>
                    <h4 className="text-lg  font-semibold mt-2">
                      Neem tooth brush
                    </h4>
                    <div className="flex text-black items-center mt-1">
                      <Rate allowHalf disabled defaultValue={4.5} />
                      <span className="font-bold">4.5</span>
                    </div>
                    <Button className="mt-2   w-full">See Demo Report</Button>
                  </CardContent>
                </Card>
                <Card className="bg-[url('https://m.media-amazon.com/images/W/MEDIAX_849526-T3/images/I/71jvmucUpRL.SX679.jpg')] bg-cover h-[380px]">
                  <div className="h-[250px]"></div>
                  <CardContent>
                    <h4 className="text-lg text-black font-semibold mt-2 ">
                      Tongue Cleaner
                    </h4>
                    <div className="flex text-black items-center mt-1 ">
                      <Rate allowHalf disabled defaultValue={4.8} />
                      <span className="font-bold">4.5</span>
                    </div>
                    <Button className="mt-2  w-full">See Demo Report</Button>
                  </CardContent>
                </Card>
              </div>
            </div>
            {/* <div className="flex my-10 flex-col space-y-2 justify-between items-center ">
              <p>Over 70,000+ users are currently using Analyser.ai</p>
              <div className="flex items-center  space-x-2">
                <Badge variant="secondary">
                  <Award size={15} /> Featured
                </Badge>
                <Badge variant="secondary">
                  <Trophy size={15} /> Excellent
                </Badge>
              </div>
            </div> */}
          </div>
        </div>
      </div>
    </div>
  );
}