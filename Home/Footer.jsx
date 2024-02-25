export default function Footer() {
  return (
    <footer className="bg-black text-white py-12">
      <div className="max-w-6xl py-10 mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 py-10 md:grid-cols-6 gap-8">
          <div className="col-span-2 md:col-span-2">
            <div className="flex items-center mb-6">
              <AppleIcon className="h-8 w-8 text-white" />
              <span className="ml-3 text-xl font-semibold">analyser.ai</span>
            </div>
            <p className="text-white text-sm mb-4">
              Convert textual customer feedback from multiple sources - reviews,
              chats, tickets - into actionable insights and trends in minutes.
            </p>
            <p className="text-white text-sm">
              analyser.ai Pte. Ltd.
              <br />
              Subhash Lane
              <br />
              #02-01 Mumbai (049422)
            </p>
          </div>
          <div className="col-span-1 md:col-span-1">
            <h6 className="text-white text-sm font-semibold mb-4">COMPANY</h6>
            <ul className="text-white text-sm">
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  About
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Contact
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Blog
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  FAQs
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Privacy Hub
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a className="hover:underline" href="#">
                  Terms of Use
                </a>
              </li>
            </ul>
          </div>
          <div className="col-span-1 md:col-span-1">
            <h6 className="text-white text-sm font-semibold mb-4">CHANNELS</h6>
            <ul className="text-white text-sm">
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Zendesk
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  App Store
                </a>
              </li>
              <li>
                <a className="hover:underline" href="#">
                  Chat Transcripts
                </a>
              </li>
            </ul>
            <h6 className="text-white text-sm font-semibold mb-4 mt-6">
              USE CASES
            </h6>
            <ul className="text-white text-sm">
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Amazon Review Analysis
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  CSAT / NPS Survey
                </a>
              </li>
              <li>
                <a className="hover:underline" href="#">
                  Support Tickets
                </a>
              </li>
            </ul>
          </div>
          <div className="col-span-1 md:col-span-1">
            <h6 className="text-white text-sm font-semibold mb-4">VERTICALS</h6>
            <ul className="text-white text-sm">
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Electronics
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Hospitality
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Lifestyle
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Gaming
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  FMCG
                </a>
              </li>
              <li>
                <a className="hover:underline" href="#">
                  SaaS
                </a>
              </li>
            </ul>
          </div>
          <div className="col-span-1 md:col-span-1">
            <h6 className="text-white text-sm font-semibold mb-4">
              CAPABILITIES
            </h6>
            <ul className="text-white text-sm">
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Data Aggregation
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Feedback Analytics
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Topic Classification
                </a>
              </li>
              <li className="mb-2">
                <a className="hover:underline" href="#">
                  Sentiment Analysis
                </a>
              </li>
              <li>
                <a className="hover:underline" href="#">
                  Product Analysis
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-white mt-8 py-6 text-center text-white text-sm">
          © 2024 analyser.ai Pte. Ltd. All rights reserved.
        </div>
      </div>
    </footer>
  );
}

function AppleIcon(props) {
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
      <path d="M12 20.94c1.5 0 2.75 1.06 4 1.06 3 0 6-8 6-12.22A4.91 4.91 0 0 0 17 5c-2.22 0-4 1.44-5 2-1-.56-2.78-2-5-2a4.9 4.9 0 0 0-5 4.78C2 14 5 22 8 22c1.25 0 2.5-1.06 4-1.06Z" />
      <path d="M10 2c1 .5 2 2 2 5" />
    </svg>
  );
}
