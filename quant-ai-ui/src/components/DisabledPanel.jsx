export default function DisabledPanel({ title, description }) {
  return (
    <div className="mt-5 p-4 border border-dashed border-gray-600 rounded-lg opacity-70">
      <h4 className="text-gray-300 font-medium mb-2">{title}</h4>
      <p className="text-gray-500 text-sm m-0">
        {description || "Coming soon"}
      </p>
    </div>
  );
}
