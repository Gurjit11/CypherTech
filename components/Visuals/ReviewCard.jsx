import { CardContent, CardFooter, Card } from "../ui/card";
import { Button } from "../ui/button";
import { Rate } from "antd";

export default function ReviewCard() {
  return (
    <Card className="w-full max-w-lg rounded-lg border">
      <CardContent className="p-6 flex gap-6">
        <div className="flex items-start">
          <div className="ml-4">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold">Alice Johnson</h3>
              <div className="flex items-center gap-1">
                <Rate value={3.2} />
              </div>
            </div>
            <div className="text-sm text-muted-foreground-variant-1 dark:text-muted-foreground-variant-1">
              3.2 out of 5
            </div>
          </div>
        </div>
      </CardContent>
      <div className="border-t border-gray-100 dark:border-gray-800" />
      <CardContent className="p-6 flex-1 overflow-auto">
        <p className="text-sm/relaxed">
          The product is good, but the color is a bit different from the
          picture. Overall, it's a nice purchase.
        </p>
      </CardContent>
      <div className="border-t border-gray-100 dark:border-gray-800" />
      <CardContent className="p-6 flex items-center gap-4 text-xs border-t">
        <div className="flex items-center gap-2">
          <StarIcon className="w-4 h-4 fill-primary" />
          <StarIcon className="w-4 h-4 fill-primary" />
          <StarIcon className="w-4 h-4 fill-primary" />
          <StarIcon className="w-4 h-4 fill-muted stroke-muted-foreground" />
          <StarIcon className="w-4 h-4 fill-muted stroke-muted-foreground" />
        </div>
        <div className="flex items-center gap-2">
          <CalendarIcon className="w-4 h-4 fill-muted-variant-2 stroke-muted-variant-2" />
          <span className="text-sm text-muted-foreground-variant-1 dark:text-muted-foreground-variant-1">
            Reviewed on August 5, 2023
          </span>
        </div>
      </CardContent>
      {/* <CardFooter className="p-6 flex justify-between">
        <div className="flex gap-2">
          <Button size="xs" variant="outline">
            <ThumbsUpIcon className="w-4 h-4 mr-2 shrink-0" />
            Like
          </Button>
          <Button size="xs" variant="outline">
            <ShareIcon className="w-4 h-4 mr-2 shrink-0" />
            Share
          </Button>
        </div>
      </CardFooter> */}
    </Card>
  );
}

function CalendarIcon(props) {
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
      <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
      <line x1="16" x2="16" y1="2" y2="6" />
      <line x1="8" x2="8" y1="2" y2="6" />
      <line x1="3" x2="21" y1="10" y2="10" />
    </svg>
  );
}

function ShareIcon(props) {
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
      <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
      <polyline points="16 6 12 2 8 6" />
      <line x1="12" x2="12" y1="2" y2="15" />
    </svg>
  );
}

function StarIcon(props) {
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
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function ThumbsUpIcon(props) {
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
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z" />
    </svg>
  );
}
