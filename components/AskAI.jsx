import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { useState } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";
import ReviewAnalysisTable from "./ReviewAnalysisTable";
const genAI = new GoogleGenerativeAI("AIzaSyDmnLNJ6kmDogUz9jGDo8i9");
const model = genAI.getGenerativeModel({ model: "gemini-pro" });

export default function AskAI() {
  const [isLoading, setIsLoading] = useState(false);
  const [prompts, setPrompts] = useState({
    question: "",
    answer: "",
  });
  const [chatMessages, setChatMessages] = useState([]);

  const handleChatBox = async () => {
    setIsLoading(true);

    const prompt = `context: you are an ai agent that performs semantic analysis on user input give a detailed review in form of a json object your answer should include 
sample  output object : 
{
  "semanticAnalysis": {
    "overallSentiment": "positive",
    "sentimentOfSpecificAspects": {
      "fit": "extremelyComfortable",
      "soundQuality": "good",
      "batteryLife": "great",
      "functionalityWithiPhone": "seamless",
      "functionalityWithAndroid": "perfect"
    },
    "topicExtraction": ["AirPods", "BluetoothHeadset", "Podcasts", "Music"],
    "opinionMining": [
      "I love my AirPods",
      "They are the coolest piece of tech that you can buy for yourself",
      "Fit is extremely comfortable for my ears",
      "They are very lightweight and can be worn all day",
      "Sound quality is good",
      "Battery life is great",
      "They work seamlessly with the iPhone but also work with android"
    ],
    "emotionDetection": ["happiness", "excitement"],
    "keyphraseExtraction": [
      "AirPods",
      "BluetoothHeadset",
      "comfortable",
      "lightweight",
      "goodSoundQuality",
      "greatBatteryLife",
      "iPhone",
      "Android"
    ],
    "aspectBasedSentimentAnalysis": {
      "fit": "positive",
      "soundQuality": "positive",
      "batteryLife": "positive",
      "functionalityWithiPhone": "positive",
      "functionalityWithAndroid": "positive"
    },
    "entityRecognition": [
      {
        "entity": "RichardUnderwood",
        "type": "User"
      },
      {
        "entity": "AirPods",
        "type": "Product"
      },
      {
        "entity": "iPhone",
        "type": "Product"
      },
      {
        "entity": "Android",
        "type": "Product"
      }
    ],
    "additionalConsiderations": {
      "sarcasmAndNegationDetection": "noneDetected",
      "contextualUnderstanding": {
        "productCategory": "wirelessEarbuds",
        "userDemographics": "enthusiasticTechUser",
        "reviewPlatform": "Amazon"
      },
      "comparativeAnalysis": "notApplicable"
    }
  }
}
you need to very very very strictly follow this format for output.

User Input: ${prompts.question}`;
    await model
      .generateContent(prompt)
      .then((res) => {
        setPrompts((prev) => ({
          ...prev,
          answer: res?.response?.candidates[0]?.content?.parts[0]?.text,
        }));
        console.log(res);
        setChatMessages((prev) => [
          ...prev,
          {
            question: prompts.question,
            answer: res?.response.candidates[0]?.content?.parts[0]?.text,
          },
        ]);
      })
      .catch((err) => console.log(err))
      .finally(() => setIsLoading(false));
  };

  return (
    <div className="pt-[150px] pl-[200px] mx-auto p-6 flex flex-col h-full w-full justify-center items-center rounded-lg ">
      <h1 className="text-5xl font-bold text-center mb-6">Ask AI</h1>
      <p className="text-lg text-center mb-8">
        Get Sentiment Analysis of any text using advanced AI. Enter your text
        below and let our AI do the rest.
      </p>

      {chatMessages?.map((prompts, index) => (
        <div key={index} className="w-full">
          <div className="p-5">
            <div className="backdrop-blur-sm bg-white/40  shadow-md p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-4">You</h2>
              <p className="text-gray-800">
                <span className="flex-1">{prompts?.question}</span>
              </p>
            </div>
          </div>
          <div className="p-5">
            <div className="flex gap-3">
              <div className="backdrop-blur-sm bg-white/40 shadow-md w-full p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4">AI Output</h2>
                <p className="text-gray-800 w-full">
                  {/* <textarea
                    className="bg-transparent border-none  h-auto resize-y w-full "
                    value={prompts?.answer}
                    contentEditable={false}
                    readOnly
                    rows={30}
                    // cols={50}
                  ></textarea> */}
                  <ReviewAnalysisTable analysis={JSON.parse(prompts?.answer)} />
                </p>
              </div>
            </div>
          </div>
        </div>
      ))}
      {/* <ReviewAnalysisTable analysis={obj} /> */}
      <div className="flex justify-center p-5 sticky bottom-10 items-end rounded-xl shadow-md bg-gradient-to-t   backdrop-blur-2xl from-white/40 ">
        <div className="flex flex-col gap-4 w-full">
          <div className="flex justify-center mb-6">
            <Input
              onChange={(e) =>
                setPrompts((prev) => ({ ...prev, question: e.target.value }))
              }
              className="border p-3 w-[400px] rounded-l-lg mr-2"
              placeholder="Enter a user review..."
            />
            <Button className="rounded-r-lg" onClick={handleChatBox}>
              {isLoading ? (
                <div className="flex justify-center animate-spin">
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 20 20"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      opacity="0.2"
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M10 17C10.9193 17 11.8295 16.8189 12.6788 16.4672C13.5281 16.1154 14.2997 15.5998 14.9497 14.9497C15.5998 14.2997 16.1154 13.5281 16.4672 12.6788C16.8189 11.8295 17 10.9193 17 10C17 9.08075 16.8189 8.17049 16.4672 7.32122C16.1154 6.47194 15.5998 5.70026 14.9497 5.05025C14.2997 4.40024 13.5281 3.88463 12.6788 3.53284C11.8295 3.18106 10.9193 3 10 3C8.14348 3 6.36301 3.7375 5.05025 5.05025C3.7375 6.36301 3 8.14348 3 10C3 11.8565 3.7375 13.637 5.05025 14.9497C6.36301 16.2625 8.14348 17 10 17ZM10 20C15.523 20 20 15.523 20 10C20 4.477 15.523 0 10 0C4.477 0 0 4.477 0 10C0 15.523 4.477 20 10 20Z"
                      fill="white"
                    />
                    <path
                      d="M0 10C0 4.477 4.477 0 10 0V3C8.14348 3 6.36301 3.7375 5.05025 5.05025C3.7375 6.36301 3 8.14348 3 10H0Z"
                      fill="#ffffff"
                    />
                  </svg>
                </div>
              ) : (
                "Summarize"
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Semantic analysis of a review should include several key elements to gain a comprehensive understanding of the reviewer's sentiment and meaning:

// 1. Sentiment Analysis:

// Overall sentiment: Categorize the review as positive, negative, or neutral based on the reviewer's tone and language.
// Sentiment of specific aspects: Identify the sentiment towards different aspects of the product or service mentioned in the review, such as features, functionality, customer service, etc.
// 2. Topic Extraction:

// Identify the main topics and entities discussed in the review. This can include the product's features, functionalities, customer service experience, or any other relevant topics.
// 3. Opinion Mining:

// Extract the reviewer's opinions and beliefs about the product or service. This can involve identifying specific strengths, weaknesses, and suggestions mentioned in the review.
// 4. Emotion Detection:

// Analyze the review for emotions expressed by the reviewer, such as happiness, frustration, anger, or disappointment. This can provide deeper insights into the reviewer's overall experience.
// 5. Keyphrase Extraction:

// Identify the most important keywords and phrases used in the review. This can help understand the reviewer's focus and highlight key aspects of their experience.
// 6. Aspect-based Sentiment Analysis:

// Analyze the sentiment of the review towards specific aspects of the product or service, such as specific features, functionalities, or customer service interactions. This provides granular insights into individual aspects of the reviewer's experience.
// 7. Entity Recognition:

// Identify and categorize named entities mentioned in the review, such as brands, people, locations, or organizations. This can provide context and additional insights into the reviewer's experience.
// Additional Considerations:

// Sarcasm and Negation Detection: Implement techniques to identify and handle sarcasm and negation in the review text, as these can alter the intended meaning.
// Contextual Understanding: Consider the context of the review, such as the product category, user demographics, and review platform, to better interpret the meaning and sentiment.
// Comparative Analysis: Compare the sentiment and topics across multiple reviews to identify trends and patterns in customer feedback.
