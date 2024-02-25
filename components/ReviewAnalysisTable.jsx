import React from "react";

const ReviewAnalysisTable = ({ analysis }) => {
  console.log(analysis);
  const {
    overallSentiment,
    sentimentOfSpecificAspects,
    topicExtraction,
    opinionMining,
    emotionDetection,
    keyphraseExtraction,
    aspectBasedSentimentAnalysis,
    entityRecognition,
    additionalConsiderations,
  } = analysis?.semanticAnalysis;

  return (
    <table className="table-auto w-full border border-gray-200 shadow-md rounded-lg">
      <thead>
        <tr className="bg-gray-100 text-left text-xs font-medium">
          <th className="px-4 py-2">Category</th>
          <th className="px-4 py-2">Details</th>
        </tr>
      </thead>
      <tbody className="text-sm">
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Overall Sentiment</td>
          <td className="px-4 py-2">{overallSentiment || ""}</td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Sentiment of Specific Aspects</td>
          <td className="px-4 py-2">
            {sentimentOfSpecificAspects &&
              Object.entries(sentimentOfSpecificAspects)?.map(
                ([aspect, sentiment]) => (
                  <tr key={aspect} className="border-b border-gray-200">
                    <td className="px-4 py-2">{aspect}</td>
                    <td className="px-4 py-2">{sentiment}</td>
                  </tr>
                )
              )}
          </td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Topics</td>
          <td className="px-4 py-2">
            {topicExtraction && topicExtraction?.join(", ")}
          </td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Opinion Mining</td>
          <td className="px-4 py-2">
            {opinionMining && opinionMining?.join("\n")}
          </td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Emotion Detection</td>
          <td className="px-4 py-2">
            {emotionDetection && emotionDetection?.join(", ")}
          </td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Keyphrase Extraction</td>
          <td className="px-4 py-2">{keyphraseExtraction?.join(", ")}</td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Aspect-based Sentiment Analysis</td>
          <td className="px-4 py-2">
            {Object.entries(aspectBasedSentimentAnalysis)?.map(
              ([aspect, sentiment]) => (
                <tr key={aspect} className="border-b border-gray-200">
                  <td className="px-4 py-2">{aspect}</td>
                  <td className="px-4 py-2">{sentiment}</td>
                </tr>
              )
            )}
          </td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Entity Recognition</td>
          <td className="px-4 py-2">
            {entityRecognition?.map((entity) => (
              <tr key={entity.entity} className="border-b border-gray-200">
                <td className="px-4 py-2">{entity.entity}</td>
                <td className="px-4 py-2">{entity.type}</td>
              </tr>
            ))}
          </td>
        </tr>
        <tr className="border-b border-gray-200">
          <td className="px-4 py-2">Additional Considerations</td>
          <td className="px-4 py-2">
            <ul className="list-disc space-y-1">
              <li>
                Sarcasm and Negation Detection:{" "}
                {additionalConsiderations?.sarcasmAndNegationDetection}
              </li>
              <li>
                Contextual Understanding:
                <ul>
                  <li>
                    Product Category:{" "}
                    {
                      additionalConsiderations?.contextualUnderstanding
                        ?.productCategory
                    }
                  </li>
                  <li>
                    User Demographics:{" "}
                    {
                      additionalConsiderations?.contextualUnderstanding
                        ?.userDemographics
                    }
                  </li>
                  <li>
                    Review Platform:{" "}
                    {
                      additionalConsiderations?.contextualUnderstanding
                        ?.reviewPlatform
                    }
                  </li>
                </ul>
              </li>
              <li>
                Comparative Analysis:{" "}
                {additionalConsiderations?.comparativeAnalysis}
              </li>
            </ul>
          </td>
        </tr>
      </tbody>
    </table>
  );
};

export default ReviewAnalysisTable;
