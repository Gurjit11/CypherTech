import { TagCloud } from "react-tagcloud";



const Wordcloud = ({formatted_keyword}) => (
  <div className="w-[300px] flex flex-col justify-center items-center">
    <TagCloud
      minSize={12}
      maxSize={35}
      tags={formatted_keyword}
      shuffle
      className="text-center"
      onClick={(tag) => alert(`'${tag.value}' was selected!`)}
    />
  </div>
);

export default Wordcloud;
