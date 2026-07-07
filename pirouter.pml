#! vim: ft=paimel

# pirouter — OpenAI-compatible HTTP router backed by `pi --mode rpc`.

import argparse
import builtins as py
import collections
import hashlib
import http.server
import json refer {JSONDecodeError}
import logging
import paimel.json refer {dumps, loads}
import pathlib refer {Path}
import subprocess
import sys
import threading
import time
import uuid

let log = logging.getLogger "pirouter"


# ---------- constants ----------

let REASONING_EFFORT_MAP = {
  none: "off",
  minimal: "minimal",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
}

let IGNORED_WITH_WARNING =
  ["temperature", "stop", "max_tokens", "max_completion_tokens"]


# ---------- entry point ----------

# CLI entry point: parse args and run the HTTP server.
let main () =
  let p = argparse.ArgumentParser prog:"pirouter"
  p.add_argument "--port" type:int default:8742
  p.add_argument "--host" default:"127.0.0.1"
  p.add_argument "--sessions" type:int default:4
  p.add_argument "--pi-binary" default:"pi"
  let args = p.parse_args ()
  let cache = SessionCache args.sessions
  let handlerCls = makeHandler args.pi_binary cache
  let addr = py.tuple [args.host, args.port]
  let server = http.server.ThreadingHTTPServer addr handlerCls
  logging.basicConfig
    level:logging.INFO
    format:"%(levelname)s %(name)s: %(message)s"
    stream:sys.stderr
  log.info "pirouter listening on http://%s:%d" args.host args.port
  try server.serve_forever ()
  except KeyboardInterrupt do nil
  finally
    server.server_close ()
    cache.closeAll ()


# ---------- HTTP layer ----------

let makeHandler piBinary cache =
  """Build a BaseHTTPRequestHandler subclass closing over piBinary+cache."""
  class Handler [http.server.BaseHTTPRequestHandler] =
    let log_message self fmt *args =
      log.info "%s - %s" (self.address_string ()) (fmt mod args)

    let do_POST self =
      if self.path != "/chat/completions"
      then self._error 404 "not found" "invalid_request_error"
      else
        try
          let length = int $ self.headers.get "Content-Length" "0"
          let raw = self.rfile.read length
          let body = loads $ raw.decode "utf-8"
          let parsed = parseRequest body
          let response = handleCompletion piBinary cache parsed
          self._json 200 response
        except JSONDecodeError as e do
          self._error 400 "invalid JSON: ${str e}" "invalid_request_error"
        except ValueError as e do
          self._error 400 (str e) "invalid_request_error"
        except Exception as e do
          log.exception "request failed"
          self._error 500 (str e) "server_error"

    let _json self status payload =
      let body = dumps payload |. encode "utf-8"
      self.send_response status
      self.send_header "Content-Type" "application/json"
      self.send_header "Content-Length" (str $ len body)
      self.end_headers ()
      self.wfile.write body

    let _error self status message etype =
      self._json status
        {error: {message, type: etype, code: status}}
  Handler


# ---------- request pipeline ----------

let handleCompletion piBinary cache parsed =
  """Run one chat-completion turn against the session cache."""
  let prefixKey = keyFor parsed parsed.prefixMsgs
  let sessionBox = atom $ cache.pop prefixKey
  try
    let [session, promptText] =
      prepareSession piBinary parsed (sessionBox.deref ())
    sessionBox.swap (fun _ -> session)
    let endEvent = runTurn session promptText
    let [text, reasoning, stop, usage, errorMessage] =
      extractAssistantTextAndMeta endEvent
    let newMsgs =
      concat parsed.prefixMsgs
        [parsed.lastMsg, {role: "assistant", content: text}]
    # Only cache sessions that ended cleanly. On error/aborted the pi
    # agent may still be processing, which would reject the next prompt.
    when stop != "error" && stop != "aborted" do
      cache.insert (keyFor parsed newMsgs) session
      sessionBox.swap (fun _ -> nil)  # ownership handed to cache
    buildCompletion parsed.model text reasoning stop usage errorMessage
  finally
    let s = sessionBox.deref ()
    when some? s do s.close ()


let parseRequest body =
  """Validate and normalize the body. Raises ValueError on bad input."""
  let messages = get body "messages"
  when not (vector? messages || list? messages) || empty? messages do
    raise $ ValueError "messages must be a non-empty array"
  for! f in IGNORED_WITH_WARNING do
    when contains? body f do
      log.warning "ignoring unsupported field: %s" f
  let model = get body "model"
  let [provider, modelId] = parseModel model
  let thinking = lookupThinking $ get body "reasoning_effort"
  let [systemPrompt, conv] = splitMessages messages
  when empty? conv do raise $ ValueError "no user/assistant messages"
  let lastIdx = len conv - 1
  let lastMsg = conv.(lastIdx)
  when lastMsg.role != "user" do
    raise $ ValueError "last message must have role=user"
  Parsed
    model provider modelId thinking
    systemPrompt
    (vec $ take lastIdx conv)
    lastMsg


let prepareSession piBinary parsed existing =
  """Return [session, promptText]. Spawns a new session on cache miss."""
  if some? existing then [existing, parsed.lastMsg.content]
  else
    let session =
      spawnPi piBinary parsed.provider parsed.modelId parsed.systemPrompt
    when some? parsed.thinking do
      session.send {type: "set_thinking_level", level: parsed.thinking}
      let resp = session.awaitResponse "set_thinking_level"
      when not resp.success do
        log.warning "set_thinking_level rejected: %s" resp.error
    [session, composeInitialPrompt parsed.prefixMsgs parsed.lastMsg.content]


let runTurn session promptText =
  """Send a prompt and return the agent_end event."""
  session.send {type: "prompt", message: promptText}
  let resp = session.awaitResponse "prompt"
  when not resp.success do
    raise $ RuntimeError "pi rejected prompt: ${str resp.error}"
  session.awaitAgentEnd ()


let extractAssistantTextAndMeta agentEnd =
  """Pull [text, reasoning, stopReason, usage] from the last assistant message."""
  let assistants =
    get agentEnd "messages" []
      |> filter (fun m -> (get m "role") == "assistant")
      |> vec
  when empty? assistants do
    raise $ RuntimeError "pi agent_end contained no assistant message"
  let last = assistants.(len assistants - 1)
  let content = get last "content" []
  let textParts =
    content
      |> filter (fun c -> map? c && (get c "type") == "text")
      |> map (fun c -> get c "text" "")
      |> filter string?
  let thinkingParts =
    content
      |> filter (fun c -> map? c && (get c "type") == "thinking")
      |> map (fun c -> get c "thinking" "")
      |> filter string?
  [
    "" |. join textParts,
    "\n\n" |. join thinkingParts,
    get last "stopReason" "stop",
    get last "usage" {},
    get last "errorMessage" nil,
  ]


let buildCompletion model assistantText reasoningText stopReason usage errorMessage =
  """Format the OpenAI ChatCompletion response body."""
  let finishReason = if stopReason == "length" then "length" else "stop"
  let promptTokens = int $ get usage "input" 0
  let completionTokens = int $ get usage "output" 0
  let message =
    merge
      {role: "assistant", content: assistantText} $
      when notEmpty? reasoningText do {reasoning_content: reasoningText}
  when stopReason == "error" || stopReason == "aborted" do
    log.warning "assistant stopReason=%s: %s"
      stopReason $ errorMessage || "(no detail)"
  {
    id: "chatcmpl-${uuid.uuid4 () |. hex}",
    object: "chat.completion",
    created: int $ time.time (),
    model: model || "",
    choices: [{
      index: 0,
      message: message,
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  }


# ---------- Pi subprocess ----------

let spawnPi piBinary provider modelId systemPrompt =
  """Spawn a `pi --mode rpc` subprocess locked down for text completion."""
  let baseArgs = [
    piBinary, "--mode", "rpc",
    "--no-session", "--no-tools", "--no-extensions",
    "--no-skills", "--no-prompt-templates", "--no-context-files",
    "--extension", Path __file__ |. parent |. joinpath "system-prompt.ts" |> str,
  ]
  let providerArgs = if provider then ["--provider", provider] else []
  let modelArgs = if modelId then ["--model", modelId] else []
  let promptArgs =
    if systemPrompt != "" then ["--system-prompt", systemPrompt] else []
  let args = vec $ concat baseArgs providerArgs modelArgs promptArgs
  log.info "spawning: %s" $ " " |. join args
  PiSession $ subprocess.Popen
    args
    stdin:subprocess.PIPE
    stdout:subprocess.PIPE
    stderr:sys.stderr


# PiSession wraps one `pi --mode rpc` subprocess.
class PiSession =
  let __init__ self proc = set! self.proc proc

  let send self cmd =
    let data = dumps cmd + "\n" |. encode "utf-8"
    self.proc.stdin.write data
    self.proc.stdin.flush ()

  let _recv self =
    let line = self.proc.stdout.readline ()
    if len line == 0
    then raise $ RuntimeError "pi subprocess closed stdout unexpectedly"
    else loads $ line.rstrip () |. decode "utf-8"

  let awaitResponse self command =
    loop _ = nil in
      let msg = self._recv ()
      if msg.type == "response" && msg.command == command
      then msg
      else
        when msg.type == "extension_ui_request" do
          log.warning
            "ignoring extension_ui_request method=%s"
            msg.method
        recur nil

  let awaitAgentEnd self =
    loop _ = nil in
      let msg = self._recv ()
      if msg.type == "agent_end" then msg
      else
        when msg.type == "extension_ui_request" do
          log.warning
            "ignoring extension_ui_request method=%s"
            msg.method
        recur nil

  let close self =
    try
      when self.proc.stdin do self.proc.stdin.close ()
    except Exception do nil
    try
      self.proc.wait timeout:2.0
      nil
    except subprocess.TimeoutExpired do
      try
        self.proc.terminate ()
        self.proc.wait timeout:2.0
        nil
      except Exception do
        try
          self.proc.kill ()
          nil
        except Exception do nil


# ---------- LRU session cache ----------

# SessionCache: LRU cache of `PiSession`s keyed by prefix hash.
class SessionCache =
  let __init__ self maxsize =
    set! self.maxsize maxsize
    set! self.entries (collections.OrderedDict ())
    set! self.lock (threading.Lock ())

  let pop self key =
    with _ = self.lock do self.entries.pop key nil

  let insert self key session =
    let toClose =
      with _ = self.lock do
        let evicted =
          if self.entries.__contains__ key
          then [self.entries.pop key]
          else []
        self.entries.__setitem__ key session
        self.entries.move_to_end key
        if len self.entries > self.maxsize
        then conj evicted (second $ self.entries.popitem last:false)
        else evicted
    for! s in toClose do s.close ()

  let closeAll self =
    let sessions =
      with _ = self.lock do
        let s = vec $ self.entries.values ()
        self.entries.clear ()
        s
    for! s in sessions do s.close ()


# ---------- helpers ----------

class Parsed model provider modelId thinking systemPrompt prefixMsgs lastMsg


let keyFor parsed messages =
  """Hash messages with the parsed request's model/thinking/sys."""
  hashKey
    messages parsed.provider parsed.modelId
    parsed.thinking parsed.systemPrompt


let hashKey messages provider modelId thinking systemPrompt =
  """Compute the SHA-256 cache key for messages + model/thinking/sys."""
  let canonical = dumps messages
  let suffix = "|" + (provider || "") + "|" + (modelId || "") + "|"
    + (thinking || "default") + "|" + systemPrompt
  let h = hashlib.sha256 $ canonical + suffix |. encode "utf-8"
  h.hexdigest ()


let lookupThinking effort =
  """Map an OpenAI reasoning_effort string to Pi's thinking-level token."""
  when some? effort do
    let t = get REASONING_EFFORT_MAP effort
    when nil? t do
      log.warning "unknown reasoning_effort: %r (ignored)" effort
    t


let parseModel m =
  """Split an OpenAI `model` string into (provider, modelId)."""
  let nilIfEmpty s = when notEmpty? s do s
  if empty? m then [nil, nil]
  else
    let parts = m.split "/" 1
    if len parts == 2
    then [nilIfEmpty parts.(0), nilIfEmpty parts.(1)]
    else [nil, m]


let splitMessages messages =
  """Return [systemPrompt, normalized user/assistant messages]."""
  let isSys m =
    (get m "role") == "system" || (get m "role") == "developer"
  let isConv m =
    (get m "role") == "user" || (get m "role") == "assistant"
  let textOf m = extractText $ get m "content"
  let sysParts =
    messages
      |> filter isSys
      |> map textOf
      |> filter notEmpty?
      |> vec
  let rest =
    messages
      |> filter isConv
      |> map (fun m -> {role: m.role, content: textOf m})
      |> vec
  ["\n\n" |. join sysParts, rest]


let extractText content =
  """OpenAI content (string or array of parts) -> plain text."""
  case
    nil? content -> ""
    string? content -> content
    vector? content || list? content ->
      let parts =
        content
          |> filter (fun p -> map? p && (get p "type") == "text")
          |> map (fun p -> get p "text" "")
          |> filter string?
      "" |. join parts
    _ -> ""


let composeInitialPrompt history lastText =
  """Compose the first prompt sent to a fresh Pi session."""
  if empty? history then lastText
  else
    let formatMsg m =
      "<message role=\"${m.role}\">${m.content}</message>"
    let lines =
      concat
        ["<conversation>"]
        (map history formatMsg)
        ["</conversation>"]
    ("\n".join lines) + "\n\n" + lastText
