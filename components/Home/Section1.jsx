export default function Section1() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h2 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">
          Everything you need
        </h2>
        <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
          Make Customer Feedback Actionable
        </p>
        <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
          Key insights from review, chat & survey data available in one place -
          to help you make better decisions.
        </p>
      </div>
      <div className="mt-10">
        <ul className="md:grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <li>
            <CheckIcon className="text-green-500" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Enterprise Grade Security
            </h3>
            <p className="mt-2 text-base text-gray-500">
              Data stored securely with the highest grade of security.
            </p>
          </li>
          <li>
            <CheckIcon className="text-green-500" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Multi-Language Support
            </h3>
            <p className="mt-2 text-base text-gray-500">
              Analyze feedback across markets and languages.
            </p>
          </li>
          <li>
            <CheckIcon className="text-green-500" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Custom Source Connectors
            </h3>
            <p className="mt-2 text-base text-gray-500">
              If we do not yet have a connector for your source, we will build
              one.
            </p>
          </li>
          <li>
            <CheckIcon className="text-green-500" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Intelligent Search
            </h3>
            <p className="mt-2 text-base text-gray-500">
              Intelligent Search to help navigate through feedback.
            </p>
          </li>
          <li>
            <CheckIcon className="text-green-500" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Category Insights
            </h3>
            <p className="mt-2 text-base text-gray-500">
              Discover interesting trends and keep an eye on competition.
            </p>
          </li>
          <li>
            <CheckIcon className="text-green-500" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Internet Knowledge Base
            </h3>
            <p className="mt-2 text-base text-gray-500">
              Leverage deep vertical specific knowledge base enriched from
              internet data.
            </p>
          </li>
        </ul>
      </div>
    </div>
  );
}

function CheckIcon(props) {
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
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}
