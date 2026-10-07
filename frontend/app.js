const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const answerBox = document.getElementById("answer");
const sourcesBox = document.getElementById("sources");

async function askQuestion(question) {
  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();
  if (!response.ok) {
    // FastAPI returns a string detail for HTTPException, a list of errors for validation failures
    const detail = typeof data.detail === "string" ? data.detail : data.detail?.[0]?.msg;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

function renderAnswer(markdown) {
  // Answers are Markdown from an LLM: sanitize before inserting as HTML
  answerBox.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesBox.textContent = "";
    return;
  }
  const labels = sources.map((s) => (s.page ? `${s.file} (p. ${s.page})` : s.file));
  sourcesBox.textContent = "Sources: " + labels.join(", ");
}

function setLoading(isLoading) {
  askButton.disabled = isLoading;
  askButton.textContent = isLoading ? "Thinking..." : "Ask";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  setLoading(true);
  answerBox.hidden = false;
  answerBox.classList.remove("error");
  answerBox.textContent = "Searching the docs...";
  sourcesBox.textContent = "";

  try {
    const data = await askQuestion(question);
    renderAnswer(data.answer);
    renderSources(data.sources);
  } catch (err) {
    answerBox.classList.add("error");
    answerBox.textContent = "Error: " + err.message;
  } finally {
    setLoading(false);
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});
