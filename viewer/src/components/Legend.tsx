interface Props {
  legend: Array<[string, string]>;
}

export function Legend({ legend }: Props) {
  const filtered = legend.filter(([, color]) => color !== "#21262d");
  if (filtered.length === 0) return null;

  return (
    <div style={{
      display: "flex",
      flexWrap: "wrap",
      gap: 12,
      fontSize: 10,
      color: "#8b949e",
    }}>
      {filtered.slice(0, 14).map(([name, color]) => (
        <span key={name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{
            width: 8,
            height: 8,
            borderRadius: 2,
            background: color,
            display: "inline-block",
          }} />
          {name}
        </span>
      ))}
    </div>
  );
}
