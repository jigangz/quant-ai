export default function DisabledPanel({ title, description }) {
  return (
    <div
      style={{
        marginTop: 20,
        padding: 16,
        border: "1px dashed #bbb",
        borderRadius: 6,
        background: "#fafafa",
        opacity: 0.7,
      }}
    >
      <h4 style={{ marginBottom: 8 }}>{title}</h4>
      <p style={{ margin: 0, fontSize: 14 }}>
        {description || "Coming soon"}
      </p>
    </div>
  );
}
