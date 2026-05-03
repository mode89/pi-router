#! vim: ft=paimel

# pirouter — OpenAI-compatible HTTP router backed by `pi --mode rpc`.

import argparse
import builtins
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

def log = logging.getLogger "pirouter"


# ---------- constants ----------

def REASONING_EFFORT_MAP = {
  none: "off",
  minimal: "minimal",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
}

def IGNORED_WITH_WARNING =
  ["temperature", "stop", "max_tokens", "max_completion_tokens"]


# ---------- entry point ----------

# CLI entry point: parse args and run the HTTP server.
def main () =
  let p = argparse.ArgumentParser prog:"pirouter" in (
    p.add_argument "--port" type:builtins.int default:8742;
    p.add_argument "--host" default:"127.0.0.1";
    p.add_argument "--sessions" type:builtins.int default:4;
    p.add_argument "--pi-binary" default:"pi";
    let args = p.parse_args () in
    let cache = SessionCache args.sessions in
    let handlerCls = makeHandler args.pi_binary cache in
    let addr = builtins.tuple [args.host, args.port] in
    let server = http.server.ThreadingHTTPServer addr handlerCls in (
      logging.basicConfig
        level:logging.INFO
        format:"%(levelname)s %(name)s: %(message)s"
        stream:sys.stderr;
      log.info "pirouter listening on http://%s:%d" args.host args.port;
      try server.serve_forever ()
      except KeyboardInterrupt do nil
      finally (server.server_close (); cache.closeAll ())
    )
  )


# ---------- HTTP layer ----------

def makeHandler piBinary cache =
  """Build a BaseHTTPRequestHandler subclass closing over piBinary+cache."""
  let class Handler [http.server.BaseHTTPRequestHandler] = {
    def log_message self fmt *args =
      log.info "%s - %s" (self.address_string ()) (fmt mod args)

    def do_POST self =
      if self.path != "/chat/completions"
      then self._error 404 "not found" "invalid_request_error"
      else
        try
          let length = builtins.int $ self.headers.get "Content-Length" "0" in
          let raw = self.rfile.read length in
          let body = loads $ raw.decode "utf-8" in
          let parsed = parseRequest body in
          let response = handleCompletion piBinary cache parsed
          in self._json 200 response
        except JSONDecodeError as e do
          self._error 400 "invalid JSON: ${str e}" "invalid_request_error"
        except ValueError as e do
          self._error 400 (str e) "invalid_request_error"
        except Exception as e do (
          log.exception "request failed";
          self._error 500 (str e) "server_error"
        )

    def _json self status payload =
      let body = dumps payload |. encode "utf-8" in (
        self.send_response status;
        self.send_header "Content-Type" "application/json";
        self.send_header "Content-Length" (str $ len body);
        self.end_headers ();
        self.wfile.write body
      )

    def _error self status message etype =
      self._json status
        {error: {message, type: etype, code: status}}
  } in Handler


# ---------- request pipeline ----------

def handleCompletion piBinary cache parsed =
  """Run one chat-completion turn against the session cache."""
  let prefixKey = keyFor parsed parsed.prefixMsgs in
  let sessionBox = atom $ cache.pop prefixKey in
  try
    let [session, promptText] =
      prepareSession piBinary parsed (sessionBox.deref ()) in (
      sessionBox.swap (fun _ -> session);
      let endEvent = runTurn session promptText in
      let [text, stop, usage] = extractAssistantTextAndMeta endEvent in
      let newMsgs =
        concat parsed.prefixMsgs
          [parsed.lastMsg, {role: "assistant", content: text}]
      in (
        cache.insert (keyFor parsed newMsgs) session;
        sessionBox.swap (fun _ -> nil);  # ownership handed to cache
        buildCompletion parsed.model text stop usage
      )
    )
  finally
    let s = sessionBox.deref ()
    in when some? s do s.close ()


def parseRequest body =
  """Validate and normalize the body. Raises ValueError on bad input."""
  let messages = get body "messages" in (
    when not (vector? messages || list? messages) || empty? messages do
      raise $ ValueError "messages must be a non-empty array";
    for! f in IGNORED_WITH_WARNING do
      when contains? body f do
        log.warning "ignoring unsupported field: %s" f;
    let model = get body "model" in
    let [provider, modelId] = parseModel model in
    let thinking = lookupThinking $ get body "reasoning_effort" in
    let [systemPrompt, conv] = splitMessages messages in (
      when empty? conv do raise $ ValueError "no user/assistant messages";
      let lastIdx = len conv - 1 in
      let lastMsg = conv.(lastIdx) in (
        when lastMsg.role != "user" do
          raise $ ValueError "last message must have role=user";
        Parsed
          model provider modelId thinking
          systemPrompt
          (vec $ take lastIdx conv)
          lastMsg
      )
    )
  )


def prepareSession piBinary parsed existing =
  """Return [session, promptText]. Spawns a new session on cache miss."""
  if some? existing then [existing, parsed.lastMsg.content]
  else
    let session =
      spawnPi piBinary parsed.provider parsed.modelId parsed.systemPrompt
    in (
      when some? parsed.thinking do (
        session.send {type: "set_thinking_level", level: parsed.thinking};
        let resp = session.awaitResponse "set_thinking_level"
        in when not (get resp "success") do
          log.warning "set_thinking_level rejected: %s" (get resp "error")
      );
      [session, composeInitialPrompt parsed.prefixMsgs parsed.lastMsg.content]
    )


def runTurn session promptText =
  """Send a prompt and return the agent_end event."""
  (
    session.send {type: "prompt", message: promptText};
    let resp = session.awaitResponse "prompt" in (
      when not (get resp "success") do
        raise $ RuntimeError "pi rejected prompt: ${str (get resp \"error\")}";
      session.awaitAgentEnd ()
    )
  )


def extractAssistantTextAndMeta agentEnd =
  """Pull [text, stopReason, usage] from the last assistant message."""
  let assistants =
    get agentEnd "messages" []
    |>> filter (fun m -> (get m "role") == "assistant")
    |> vec
  in (
    when empty? assistants do
      raise $ RuntimeError "pi agent_end contained no assistant message";
    let last = assistants.(len assistants - 1) in
    let parts =
      get last "content" []
      |>> filter (fun c -> map? c && (get c "type") == "text")
      |>> map (fun c -> get c "text" "")
      |>> filter string?
    in [
      "" |. join parts,
      get last "stopReason" "stop",
      get last "usage" {},
    ]
  )


def buildCompletion model assistantText stopReason usage =
  """Format the OpenAI ChatCompletion response body."""
  let finishReason = if stopReason == "length" then "length" else "stop" in
  let promptTokens = builtins.int $ get usage "input" 0 in
  let completionTokens = builtins.int $ get usage "output" 0
  in (
    when stopReason == "error" || stopReason == "aborted" do
      log.warning "assistant stopReason=%s" stopReason;
    {
      id: "chatcmpl-${uuid.uuid4 () |. hex}",
      object: "chat.completion",
      created: builtins.int $ time.time (),
      model: model || "",
      choices: [{
        index: 0,
        message: {role: "assistant", content: assistantText},
        finish_reason: finishReason,
      }],
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
      },
    }
  )


# ---------- Pi subprocess ----------

def spawnPi piBinary provider modelId systemPrompt =
  """Spawn a `pi --mode rpc` subprocess locked down for text completion."""
  let baseArgs = [
    piBinary, "--mode", "rpc",
    "--no-session", "--no-tools", "--no-extensions",
    "--no-skills", "--no-prompt-templates", "--no-context-files",
    "--extension", Path __file__ |. parent |. joinpath "system-prompt.ts" |> str,
  ] in
  let providerArgs = if provider then ["--provider", provider] else [] in
  let modelArgs = if modelId then ["--model", modelId] else [] in
  let promptArgs =
    if systemPrompt != "" then ["--system-prompt", systemPrompt] else [] in
  let args = vec $ concat baseArgs providerArgs modelArgs promptArgs
  in (
    log.info "spawning: %s" $ " " |. join args;
    PiSession $ subprocess.Popen
      args
      stdin:subprocess.PIPE
      stdout:subprocess.PIPE
      stderr:sys.stderr
  )


# PiSession wraps one `pi --mode rpc` subprocess.
class PiSession = {
  def __init__ self proc = set! self.proc proc

  def send self cmd =
    let data = dumps cmd + "\n" |. encode "utf-8"
    in (self.proc.stdin.write data; self.proc.stdin.flush ())

  def _recv self =
    let line = self.proc.stdout.readline ()
    in if len line == 0
       then raise $ RuntimeError "pi subprocess closed stdout unexpectedly"
       else loads $ line.rstrip () |. decode "utf-8"

  def awaitResponse self command =
    loop _ = nil in
      let msg = self._recv () in
      if (get msg "type") == "response" && (get msg "command") == command
      then msg
      else (
        when (get msg "type") == "extension_ui_request" do
          log.warning
            "ignoring extension_ui_request method=%s"
            (get msg "method");
        recur nil
      )

  def awaitAgentEnd self =
    loop _ = nil in
      let msg = self._recv () in
      let t = get msg "type" in
      if t == "agent_end" then msg
      else (
        when t == "extension_ui_request" do
          log.warning
            "ignoring extension_ui_request method=%s"
            (get msg "method");
        recur nil
      )

  def close self = (
    try (when self.proc.stdin do self.proc.stdin.close ())
    except Exception do nil;
    try (self.proc.wait timeout:2.0; nil)
    except subprocess.TimeoutExpired do
      try (self.proc.terminate (); self.proc.wait timeout:2.0; nil)
      except Exception do
        try (self.proc.kill (); nil)
        except Exception do nil
  )
}


# ---------- LRU session cache ----------

# SessionCache: LRU cache of `PiSession`s keyed by prefix hash.
class SessionCache = {
  def __init__ self maxsize = (
    set! self.maxsize maxsize;
    set! self.entries (collections.OrderedDict ());
    set! self.lock (threading.Lock ())
  )

  def pop self key =
    with _ = self.lock do self.entries.pop key nil

  def insert self key session =
    let toClose =
      with _ = self.lock do
        let evicted =
          if self.entries.__contains__ key
          then [self.entries.pop key]
          else []
        in (
          self.entries.__setitem__ key session;
          self.entries.move_to_end key;
          if len self.entries > self.maxsize
          then conj evicted (second $ self.entries.popitem last:false)
          else evicted
        )
    in for! s in toClose do s.close ()

  def closeAll self =
    let sessions =
      with _ = self.lock do
        let s = vec $ self.entries.values ()
        in (self.entries.clear (); s)
    in for! s in sessions do s.close ()
}


# ---------- helpers ----------

class Parsed
  model provider modelId thinking systemPrompt prefixMsgs lastMsg


def keyFor parsed messages =
  """Hash messages with the parsed request's model/thinking/sys."""
  hashKey
    messages parsed.provider parsed.modelId
    parsed.thinking parsed.systemPrompt


def hashKey messages provider modelId thinking systemPrompt =
  """Compute the SHA-256 cache key for messages + model/thinking/sys."""
  let canonical = dumps messages in
  let suffix =
    "|" + (provider || "") + "|" + (modelId || "") +
    "|" + (thinking || "default") + "|" + systemPrompt in
  let h = hashlib.sha256 $ canonical + suffix |. encode "utf-8"
  in h.hexdigest ()


def lookupThinking effort =
  """Map an OpenAI reasoning_effort string to Pi's thinking-level token."""
  when some? effort do
    let t = get REASONING_EFFORT_MAP effort in (
      when nil? t do
        log.warning "unknown reasoning_effort: %r (ignored)" effort;
      t
    )


def parseModel m =
  """Split an OpenAI `model` string into (provider, modelId)."""
  let nilIfEmpty s = when notEmpty? s do s in
  if empty? m then [nil, nil]
  else
    let parts = m.split "/" 1 in
    if len parts == 2
    then [nilIfEmpty parts.(0), nilIfEmpty parts.(1)]
    else [nil, m]


def splitMessages messages =
  """Return [systemPrompt, normalized user/assistant messages]."""
  let isSys m =
    (get m "role") == "system" || (get m "role") == "developer" in
  let isConv m =
    (get m "role") == "user" || (get m "role") == "assistant" in
  let textOf m = extractText $ get m "content" in
  let sysParts =
    messages
    |>> filter isSys
    |>> map textOf
    |>> filter notEmpty?
    |> vec in
  let rest =
    messages
    |>> filter isConv
    |>> map (fun m -> {role: get m "role", content: textOf m})
    |> vec
  in ["\n\n" |. join sysParts, rest]


def extractText content =
  """OpenAI content (string or array of parts) -> plain text."""
  case
  | nil? content -> ""
  | string? content -> content
  | vector? content || list? content ->
    let parts =
      content
      |>> filter (fun p -> map? p && (get p "type") == "text")
      |>> map (fun p -> get p "text" "")
      |>> filter string?
    in "" |. join parts
  | _ -> ""


def composeInitialPrompt history lastText =
  """Compose the first prompt sent to a fresh Pi session."""
  if empty? history then lastText
  else
    let formatMsg m =
      "<message role=\"${m.role}\">${m.content}</message>" in
    let lines =
      concat
        ["<conversation>"]
        (map formatMsg history)
        ["</conversation>"]
    in ("\n".join lines) + "\n\n" + lastText
