import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.grounding import unverified_code
from app.main import app
from app.rag_chain import ask_question, build_answer_chain, stream_answer

FENCE = "```"


def code(block: str, lang: str = "") -> str:
    return f"{FENCE}{lang}\n{block}\n{FENCE}"


# What the docs actually said for the restart question: grace period, --force, a URL-based apply
LIFECYCLE_DOCS = [Document(
    page_content=(
        "A Pod is granted a term to terminate gracefully, which defaults to 30 seconds.\n"
        "You can use the flag --force to terminate a Pod by force.\n"
        "kubectl apply -f https://k8s.io/examples/controllers/nginx-deployment.yaml\n"
        "restartPolicy: OnFailure\n"
        "    image: registry.k8s.io/busybox:1.27.2\n"
    ),
    metadata={"source": "k8s_pod_lifecycle.html"},
)]


# --- the real case ------------------------------------------------------------------------------

def test_flags_the_kubectl_command_the_model_added_from_memory():
    # The live answer to "How do I restart Kubernetes Pods?": the command appears in a bullet AND a code block
    answer = (
        "The docs don't cover how to restart Pods directly.\n\n"
        "- Use `kubectl delete pod <pod-name>` to delete a Pod; it will be recreated by its controller\n"
        "- Run with: `kubectl apply -f pod.yaml`\n"
        "- `restartPolicy: OnFailure` restarts only on failure\n\n"
        + code("kubectl delete pod <pod-name>")
    )
    # Reported once; the adapted apply command and the YAML key are fine
    assert unverified_code(answer, LIFECYCLE_DOCS) == ["kubectl delete pod <pod-name>"]


def test_fully_grounded_answer_has_nothing_flagged():
    answer = "Use `--force` sparingly.\n\n" + code("restartPolicy: OnFailure\nimage: registry.k8s.io/busybox:1.27.2", "yaml")
    assert unverified_code(answer, LIFECYCLE_DOCS) == []


# --- inline commands ----------------------------------------------------------------------------

def test_inline_command_with_adapted_arguments_is_accepted():
    # `kubectl apply` appears in the docs; only the file argument differs
    assert unverified_code("Apply it with `kubectl apply -f pod.yaml`.", LIFECYCLE_DOCS) == []


def test_inline_command_the_docs_never_mention_is_flagged():
    assert unverified_code("Or run `kubectl rollout restart deployment/web`.", LIFECYCLE_DOCS) == [
        "kubectl rollout restart deployment/web"
    ]


def test_inline_code_that_is_not_a_command_is_ignored():
    answer = "Set `restartPolicy: Always`, use `depends_on`, create a `.env` file, see `registry.k8s.io/busybox:1.27.2`."
    assert unverified_code(answer, LIFECYCLE_DOCS) == []


# --- code blocks (the rules the eval suite has always used) ------------------------------------

COMPOSE_DOCS = [Document(page_content="services:\n  redis:\n    image: redis:alpine\n$ docker run -p 127.0.0.1:8000:8000 test:latest",
                         metadata={"source": "compose.html"})]


def test_edited_config_value_is_flagged():
    assert unverified_code(code("services:\n  redis:\n    image: redis", "yaml"), COMPOSE_DOCS) == ["image: redis"]


def test_command_with_only_placeholder_values_changed_is_accepted():
    assert unverified_code(code("$ docker run -p 127.0.0.1:5000:5000 my-app:latest"), COMPOSE_DOCS) == []


def test_command_with_an_added_flag_is_flagged():
    line = "$ docker run -d -p 127.0.0.1:5000:5000 my-app:latest"
    assert unverified_code(code(line), COMPOSE_DOCS) == [line]


def test_elision_marker_is_not_flagged():
    assert unverified_code(code("services:\n  # ... web service config"), COMPOSE_DOCS) == []


# --- wired into the app ------------------------------------------------------------------------

def test_ask_question_returns_the_unverified_code():
    chain = MagicMock()
    chain.invoke.return_value = {"answer": "Run `kubectl delete pod web`.", "source_documents": LIFECYCLE_DOCS}
    with patch("app.rag_chain.get_qa_chain", return_value=chain):
        result = ask_question("How do I restart a Pod?")
    assert result["unverified_code"] == ["kubectl delete pod web"]


def test_stream_answer_ends_with_a_grounding_event_only_when_needed():
    def run(text):
        llm = GenericFakeChatModel(messages=iter([AIMessage(content=text)]))
        with patch("app.rag_chain.get_pipeline", return_value=(lambda q: LIFECYCLE_DOCS, build_answer_chain(llm))):
            return list(stream_answer("How do I restart a Pod?"))

    flagged = run("Delete it with `kubectl delete pod web` and wait.")
    assert flagged[-1] == ("grounding", ["kubectl delete pod web"])
    grounded = run("Pods get 30 seconds to terminate gracefully.")
    assert all(kind != "grounding" for kind, _ in grounded)


def test_stream_endpoint_forwards_the_grounding_event():
    def fake_stream(question):
        yield "sources", []
        yield "token", "Run `kubectl delete pod web`."
        yield "grounding", ["kubectl delete pod web"]

    with patch("app.main.ANTHROPIC_API_KEY", "test-key"), patch("app.main.get_qa_chain"), \
            patch("app.main.stream_answer", side_effect=fake_stream):
        with TestClient(app) as client:
            response = client.post("/ask/stream", json={"question": "How do I restart a Pod?"})

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert {"type": "grounding", "unverified_code": ["kubectl delete pod web"]} in events
    assert events[-1] == {"type": "done"}
