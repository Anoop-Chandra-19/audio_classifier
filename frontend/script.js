const { useState, useRef, useEffect } = React;

function App() {
  const [file,    setFile]    = useState(null);
  const [status,  setStatus]  = useState("");
  const [results, setResults]= useState([]);
  const [loading, setLoading]= useState(false);

  // --- NEW: ref for measuring content height
  const contentRef = useRef(null);
  const [maxH, setMaxH] = useState("160px");
  const initialH = "160px";
  const collapsedMargin = `calc(50vh - ${initialH/2}px)`;
  const expandedMargin = "2rem";  

  // When results change, measure and set
  useEffect(() => {
    if (results.length > 0 && contentRef.current) {
      // scrollHeight is the full height of the content
      const fullHeight = contentRef.current.scrollHeight;
      setMaxH(fullHeight + "px");
    }
  }, [results]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setStatus("Please choose a file."); return; }
    setStatus(""); setResults([]); setLoading(true);

    const data = new FormData(); data.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST", body: data
      });
      if (!res.ok) throw new Error(res.statusText||res.status);
      const { predictions } = await res.json();
      setResults(predictions);
    } catch (err) {
      console.error(err);
      setStatus("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const top  = results[0];
  const rest = results.slice(1);

  return (
    // --- apply inline max-height + transition here
    <div
      className="container"
      style={{
        maxHeight: maxH,
        margin: `${results.length > 0 ? collapsedMargin : expandedMargin} auto`,
        transition: "max-height 0.6s ease-in-out, margin 0.6s ease-in-out",
      }}
    >
      <div ref={contentRef}>
        <h1>Audio Genre Classifier</h1>

        {top && (
          <div className="top-genre">
            <h2>{top.genre}</h2>
            <p>{(top.confidence*100).toFixed(1)}%</p>
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ marginBottom: "1rem" }}>
          <div className="file-upload">
            <input
              id="fileInput"
              type="file"
              accept=".wav,.mp3"
              onChange={e => setFile(e.target.files[0])}
            />
            <label htmlFor="fileInput" className="btn">Choose File</label>
            <span className="file-name">
              {file?.name || "No file selected"}
            </span>
            <button type="submit" className="btn" disabled={!file}>
              Upload &amp; Classify
            </button>
          </div>
        </form>

        {loading && (
          <div className="loader"><span/><span/><span/></div>
        )}

        {status && <div className="status">{status}</div>}

        {rest.map((r,i) => {
          const base = 0.6 + i*0.2,
                fillDelay = base + 0.4;
          return (
            <div
              key={r.genre}
              className="bar-container"
              style={{ animationDelay: `${base}s` }}
            >
              <div
                className="bar-fill"
                style={{
                  "--fill-delay": `${fillDelay}s`,
                  "--target-width": `${(r.confidence*100).toFixed(1)}%`
                }}
              />
              <div className="bar-labels">
                <span>{r.genre}</span>
                <span>{(r.confidence*100).toFixed(1)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
