# Grounding check: which code in an answer is NOT backed by the retrieved docs?
# Used at run time (the app warns the user) and by the eval suite (such answers fail), so both agree.

import re

FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
# A command + its subcommand or flag, e.g. "kubectl delete", "docker compose", "pip install", "docker --version".
# Starts with a lowercase word and a space, so YAML ("restartPolicy: Always") and Dockerfile lines don't match.
COMMAND_HEAD_RE = re.compile(r"([a-z][\w.-]*) +([a-z][\w.-]*|--?[\w-]+)")
# A trailing "  # comment" after code. Needs whitespace before the "#", so "#!/bin/sh" and URL fragments stay.
TRAILING_COMMENT_RE = re.compile(r"\s+#.*$")


def unverified_code(answer: str, docs) -> list[str]:
    """Code in the answer that the docs the model was given don't contain - in the order it appears.

    The model is told to answer only from the docs, but when the obvious answer is missing it sometimes
    fills the gap from memory (e.g. `kubectl delete pod`, which none of the docs mention). Such code
    may even be right, but nothing backs it.
    """
    context_text = "\n".join(d.page_content for d in docs)
    found = invented_code_lines(answer, context_text) + unverified_inline_commands(answer, context_text)
    return list(dict.fromkeys(found))


def invented_code_lines(answer: str, context_text: str) -> list[str]:
    """Lines inside the answer's code blocks that don't appear in the docs.

    Catches merged or edited examples (e.g. an invented `image: redis`), which an LLM judge
    tends to wave through. Indentation is ignored. Config lines must match a whole docs line
    (so `image: redis` doesn't pass because `image: redis:alpine` exists); shell commands only need
    to appear somewhere, since docs often mention them inline without a "$ " prompt. Lines that are
    only an elision marker ("# ...", "...") are allowed. A shell command may also differ from a docs
    command in placeholder values only (image name/tag, port numbers) - see command_shape().
    Trailing comments are ignored on both sides: the docs' `- name: try-once-container  # runs once`
    copied without its comment is still the docs' line.
    """
    context_lines = {line.strip() for line in context_text.splitlines()}
    context_lines |= {TRAILING_COMMENT_RE.sub("", line) for line in context_lines}
    context_shapes = {command_shape(line.removeprefix("$ ")) for line in context_lines if line}
    invented = []
    for block in FENCED_BLOCK_RE.findall(answer):
        for line in (l.strip() for l in block.splitlines()):
            if not line or line in context_lines or TRAILING_COMMENT_RE.sub("", line) in context_lines:
                continue
            if re.fullmatch(r"(#|//)?\s*\.\.\..*", line):  # an explicit "# ..." elision isn't invented config
                continue
            # Shell commands, with or without a "$ " prompt: the docs often write "$ docker compose up"
            # (or inline in a sentence, without a prompt). "Command-like" = starts with a lowercase word
            # and a space, so YAML keys ("image: x") and Dockerfile lines ("EXPOSE 80") never match here.
            command = line.removeprefix("$ ")
            is_command = line.startswith("$ ") or re.match(r"[a-z][\w.-]* ", command)
            if f"$ {command}" in context_lines or (is_command and command in context_text):
                continue
            if is_command and command_shape(command) in context_shapes:
                continue
            invented.append(line)
    return invented


def unverified_inline_commands(answer: str, context_text: str) -> list[str]:
    """Inline `commands` in the prose whose command + subcommand never appears in the docs.

    Looser than the code-block check on purpose: prose often adapts arguments (`kubectl apply -f pod.yaml`
    when the docs use a URL), which is fine - but a whole command the docs never mention
    (`kubectl delete pod <name>`) is not.
    """
    prose = FENCED_BLOCK_RE.sub("", answer)
    flagged = []
    for span in INLINE_CODE_RE.findall(prose):
        command = span.strip().removeprefix("$ ")
        head = COMMAND_HEAD_RE.match(command)
        if head and f"{head.group(1)} {head.group(2)}" not in context_text:
            flagged.append(command)
    return flagged


def command_shape(command: str) -> str:
    """A command with its placeholder values blanked: `docker run -p 127.0.0.1:5000:5000 my-app:latest`
    and `docker run -p 127.0.0.1:8000:8000 test:latest` have the same shape. Commands, subcommands and
    flags are kept, so adding a flag or a different command still counts as invented."""
    command = re.sub(r"(-t|--tag)\s+\S+", r"\1 IMG", command.strip())
    command = re.sub(r"\d+", "N", command)
    command = re.sub(r"\b[\w./-]+:[\w.-]+\b", "IMG", command)  # image:tag (also N.N.N.N:N:N port maps)
    return re.sub(r"\s+", " ", command)
