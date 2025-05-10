const { useState } = React;

function App() {
  const [file, setFile]       = useState(null);
  const [status, setStatus]   = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async e => {
    e.preventDefault();
    if (!file) { setStatus("Please choose a file."); return; }
    setStatus("");
    setResults([]);
    setLoading(true);

    const data = new FormData();
    data.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: data
      });
      if (!res.ok) throw new Error(res.statusText || res.status);
      const json = await res.json();
      setResults(json.predictions);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // split into top‐hero + rest
  const top  = results[0];
  const rest = results.slice(1);

  return (
    <div>
      <h1>Audio Genre Classifier</h1>

      {/* 1) Hero appears above the form, pushing it up */}
      {top && (
        <div className="top-genre">
          <h2>{top.genre}</h2>
          <p>{(top.confidence * 100).toFixed(1)}%</p>
        </div>
      )}

      {/* 2) The form stays in place but moves up when hero mounts */}
      <form onSubmit={handleSubmit} style={{ marginBottom: "1rem" }}>
        <input
          type="file"
          accept=".wav,.mp3"
          onChange={e => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={!file}>
          Upload & Classify
        </button>
      </form>

      {loading && (
        <div className="loader">
          <span/><span/><span/>
          </div>
      )}

      {status && <div className="status">{status}</div>}

      {/* 3) Stagger the rest of the bars and delay their fill */}
      {rest.map((r, i) => {
        // baseStagger: how long after hero to start?
        const baseStagger = 0.6; // seconds
        const delayStep   = 0.2; // seconds per bar

        // delay for fadeInUp:
        const containerDelay = baseStagger + i * delayStep;
        // delay before filling width:
        const fillDelay = containerDelay + 0.4; 

        return (
          <div
            key={r.genre}
            className="bar-container"
            style={{ animationDelay: `${containerDelay}s` }}
          >
            <div
              className="bar-fill"
              style={{
                "--fill-delay": `${fillDelay}s`,
                "--target-width": `${(r.confidence * 100).toFixed(1)}%`
              }}
            ></div>
            <div className="bar-labels">
              <span>{r.genre}</span>
              <span>{(r.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
