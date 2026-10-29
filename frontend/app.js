const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const answerBox = document.getElementById("answer");
const sourcesBox = document.getElementById("sources");

// Streams the answer from /ask/stream: newline-delimited JSON events
// {"type":"sources"} once retrieval is done, {"type":"token"} as text is written, then {"type":"done"}.
async function askQuestionStreaming(question, { onSources, onText }) {
  const response = await fetch("/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    // Errors before streaming starts (rate limit, validation, startup) are normal JSON responses.
    // FastAPI returns a string detail for HTTPException, a list of errors for validation failures.
    const data = await response.json().catch(() => ({}));
    const detail = typeof data.detail === "string" ? data.detail : data.detail?.[0]?.msg;
    throw new Error(detail || `Request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // the last piece may be an incomplete line; keep it for the next chunk
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "sources") onSources(event.sources);
      else if (event.type === "token") onText((answer += event.text));
      else if (event.type === "error") throw new Error(event.detail);
      else if (event.type === "done") return answer;
    }
  }
  throw new Error("The connection closed before the answer was complete.");
}

// Re-render at most once per frame: tokens can arrive faster than parsing Markdown is worth
let pendingMarkdown = null;
function scheduleRender(markdown) {
  if (pendingMarkdown === null) {
    requestAnimationFrame(() => {
      // Cleared in the meantime (answer finished or failed)? Then don't overwrite what's shown
      if (pendingMarkdown !== null) renderAnswer(pendingMarkdown);
      pendingMarkdown = null;
    });
  }
  pendingMarkdown = markdown;
}

function renderAnswer(markdown) {
  // Answers are Markdown from an LLM: sanitize before inserting as HTML
  answerBox.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
  fixLinks(answerBox);
}

function fixLinks(container) {
  // Relative links from the docs (e.g. /docs/setup/) would resolve against this app and 404:
  // show them as plain text. Real links open in a new tab so the answer isn't lost.
  container.querySelectorAll("a").forEach((link) => {
    const href = link.getAttribute("href") || "";
    if (!/^https?:\/\//i.test(href)) {
      link.replaceWith(document.createTextNode(link.textContent));
      return;
    }
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
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
  answerBox.textContent = "Searching the docs..."; // retrieval takes a few seconds before the first words
  sourcesBox.textContent = "";

  try {
    const answer = await askQuestionStreaming(question, {
      onSources: renderSources,
      onText: scheduleRender,
    });
    pendingMarkdown = null;
    renderAnswer(answer); // final render, so nothing is left waiting for the next frame
  } catch (err) {
    pendingMarkdown = null;
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
