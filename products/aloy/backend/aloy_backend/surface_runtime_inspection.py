"""Browser execution gate for locally built generated Surfaces.

Static validation and compilation cannot detect React render failures caused by
real Event data. Local development therefore mounts the exact host-owned
runtime document in headless Chrome, injects the exact scoped Surface context
through the production MessageChannel protocol, and fails preview on any
uncaught exception, missing bridge acknowledgement, or empty React root.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from .surface_manifest import (
    SurfaceInteractionCheck,
    SurfaceManifest,
    validate_intent_payload,
)
from .surface_runtime import SurfaceRuntimeDocument

INSPECTION_TIMEOUT_SECONDS = 8.0
SETTLE_SECONDS = 1.5


def _browser_executable() -> str | None:
    configured = os.getenv("ALOY_SURFACE_BROWSER", "").strip()
    candidates = [
        configured,
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
        shutil.which("chromium-browser") or "",
        shutil.which("msedge") or "",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((item for item in candidates if item and Path(item).is_file()), None)


def _diagnostic(code: str, message: str) -> dict[str, Any]:
    return {
        "stage": "runtime",
        "code": code,
        "severity": "error",
        "message": message[:4000],
        "path": None,
        "line": None,
    }


def inspect_surface_runtime(
    document: SurfaceRuntimeDocument,
    context: dict[str, Any],
    *,
    manifest: SurfaceManifest | None = None,
) -> list[dict[str, Any]]:
    """Execute the exact runtime and every manifest-declared UI interaction.

    The checks use accessible roles and names rather than model-controlled CSS
    selectors. Each declared intent must be reachable through a visible user
    path and produce the expected typed SDK request before publication.
    """
    browser = _browser_executable()
    if browser is None:
        return [
            _diagnostic(
                "runtime_inspector_unavailable",
                "A headless Chrome or Edge runtime is required to inspect Surfaces",
            )
        ]
    try:
        from websockets.sync.client import connect
    except ImportError:
        return [
            _diagnostic(
                "runtime_inspector_unavailable",
                "The browser inspection transport is unavailable",
            )
        ]

    with tempfile.TemporaryDirectory(prefix="aloy-surface-inspect-") as temp:
        root = Path(temp)
        runtime_path = root / "surface.html"
        runtime_path.write_text(document.html, encoding="utf-8")
        profile = root / "chrome-profile"
        profile.mkdir()
        process = subprocess.Popen(
            [
                browser,
                "--headless=new",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-extensions",
                "--disable-gpu",
                "--disable-sync",
                "--metrics-recording-only",
                "--no-first-run",
                "--remote-allow-origins=*",
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                runtime_path.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            active_port = profile / "DevToolsActivePort"
            startup_deadline = time.monotonic() + INSPECTION_TIMEOUT_SECONDS
            while not active_port.is_file() and time.monotonic() < startup_deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if not active_port.is_file():
                return [
                    _diagnostic(
                        "runtime_inspector_failed",
                        "The isolated browser did not become ready",
                    )
                ]
            port = active_port.read_text(encoding="utf-8").splitlines()[0]
            with urllib.request.urlopen(  # noqa: S310 -- fixed loopback CDP endpoint
                f"http://127.0.0.1:{port}/json/list",
                timeout=2,
            ) as response:
                targets = json.load(response)
            target = next(
                (item for item in targets if item.get("type") == "page"),
                None,
            )
            if target is None:
                return [
                    _diagnostic("runtime_inspector_failed", "No browser page exists")
                ]
            socket = connect(str(target["webSocketDebuggerUrl"]), open_timeout=2)
            try:
                message_id = 0

                def send(method: str, params: dict[str, Any] | None = None) -> int:
                    nonlocal message_id
                    message_id += 1
                    socket.send(
                        json.dumps(
                            {"id": message_id, "method": method, "params": params or {}}
                        )
                    )
                    return message_id

                send("Runtime.enable")
                ready, load_exceptions = _wait_for_runtime_document(
                    socket,
                    send=send,
                    deadline=time.monotonic() + INSPECTION_TIMEOUT_SECONDS,
                )
                if load_exceptions:
                    return [
                        _diagnostic("runtime_exception", message)
                        for message in load_exceptions[:20]
                    ]
                if not ready:
                    return [
                        _diagnostic(
                            "runtime_inspector_failed",
                            "The Surface runtime document did not become ready",
                        )
                    ]
                smoke_context = json.dumps(context, ensure_ascii=True, default=str)
                smoke_commands = json.dumps(
                    {
                        name: {
                            "effect": (
                                "state"
                                if declaration.interaction_class
                                in {"state", "durable_selection"}
                                else declaration.interaction_class
                            ),
                            "status": (
                                "committed"
                                if declaration.interaction_class
                                in {"state", "durable_selection"}
                                else "queued"
                            ),
                            "write": (
                                declaration.write.model_dump(mode="json")
                                if declaration.write is not None
                                else None
                            ),
                        }
                        for name, declaration in (
                            manifest.intents.items() if manifest is not None else []
                        )
                    },
                    ensure_ascii=True,
                )
                expression = (
                    "(() => {"
                    "const channel = new MessageChannel();"
                    "let currentContext=" + smoke_context + ";"
                    "const commands=" + smoke_commands + ";"
                    "window.__aloySmokeMessages = [];"
                    "window.__aloySmokePort = channel.port1;"
                    "const applyState=(base,outcome,params,nextRevision)=>{"
                    "const write=outcome.write;if(!write)return base;"
                    "const payload=params?.payload&&typeof params.payload==='object'?params.payload:{};"
                    "const key=write.key||payload[write.key_field];"
                    "if(typeof key!=='string'||!key)return base;"
                    "const namespace=write.namespace;"
                    "const surface={...((base.data||{}).surface||{})};"
                    "const records=[...(surface[namespace]||[])];"
                    "const index=records.findIndex(item=>item?.key===key);"
                    "if(write.operation==='delete'){if(index>=0)records.splice(index,1);}"
                    "else{const current=index>=0?records[index]:null;"
                    "const data=write.operation==='merge'?{...(current?.data||{}),...payload}:{...payload};"
                    "const now=new Date().toISOString();"
                    "const record={id:current?.id||`smoke:${namespace}:${key}`,namespace,key,data,"
                    "revision:nextRevision,posture:write.posture||'user_reported',"
                    "actor_id:'runtime-inspector',provenance:{command_name:params.name},"
                    "evidence_refs:[],created_at:current?.created_at||now,updated_at:now};"
                    "if(index>=0)records[index]=record;else records.push(record);}"
                    "surface[namespace]=records;return {...base,data:{...(base.data||{}),surface}};};"
                    "channel.port1.onmessage = event => {"
                    "const message=event.data;window.__aloySmokeMessages.push(message);"
                    "if(message?.protocol==='1'&&message?.type==='request'){"
                    "const params=message.params||{};"
                    "const commandName=params.name||params.action?.name||'surface.command';"
                    "const outcome=commands[commandName]||{effect:'intent',status:'committed'};"
                    "const nextRevision=Number(currentContext.data_revision||0)+(outcome.effect==='state'?1:0);"
                    "const interaction={id:`interaction-${message.requestId}`,event_id:currentContext.event_id,"
                    "build_id:currentContext.build_id,code_revision_id:currentContext.code_revision_id,"
                    "name:commandName,interaction_class:outcome.effect,"
                    "component_id:params.componentId||'surface',status:outcome.status,"
                    "handling_run_id:null,proposal_id:null,request_message_id:null,"
                    "outcome_message_id:null,result:{},error:null,created_at:new Date().toISOString(),"
                    "updated_at:new Date().toISOString()};"
                    "const attempt={id:`attempt-${message.requestId}`,event_id:currentContext.event_id,"
                    "build_id:currentContext.build_id,code_revision_id:currentContext.code_revision_id,"
                    "interaction_id:interaction.id,method:message.method,name:commandName,"
                    "interaction_class:outcome.effect,component_id:interaction.component_id,"
                    "base_data_revision:Number(currentContext.data_revision||0),"
                    "observed_data_revision:nextRevision,status:outcome.status,error_code:null,error:null,"
                    "http_status:200,retryable:false,created_at:new Date().toISOString()};"
                    "const projected=applyState(currentContext,outcome,params,nextRevision);"
                    "currentContext={...projected,data_revision:nextRevision,"
                    "data:{...(projected.data||{}),"
                    "interactions:[...((projected.data||{}).interactions||[]),interaction],"
                    "command_attempts:[...((projected.data||{}).command_attempts||[]),attempt]}};"
                    "channel.port1.postMessage({protocol:'1',type:'context',"
                    "sessionId:'runtime-smoke',context:currentContext});"
                    "channel.port1.postMessage({protocol:'1',type:'response',"
                    "sessionId:'runtime-smoke',requestId:message.requestId,ok:true,"
                    "result:{...interaction,event_id:"
                    + json.dumps(str(context.get("event_id") or "event-smoke"))
                    + ",build_id:"
                    + json.dumps(str(context.get("build_id") or "build-smoke"))
                    + ",code_revision_id:"
                    + json.dumps(
                        str(context.get("code_revision_id") or "revision-smoke")
                    )
                    + ",status:outcome.status,data_revision:currentContext.data_revision,"
                    "proposal_id:null,handling_run_id:null,replayed:false}});}};"
                    "channel.port1.start();"
                    "window.postMessage({protocol:'1',type:'aloy.surface.connect',"
                    "sessionId:'runtime-smoke',context:"
                    + json.dumps(context, ensure_ascii=True, default=str)
                    + "},'*',[channel.port2]);return true;})()"
                )
                send(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "awaitPromise": True,
                        "returnByValue": True,
                    },
                )
                exceptions: list[str] = []
                settle_deadline = time.monotonic() + SETTLE_SECONDS
                while time.monotonic() < settle_deadline:
                    try:
                        message = json.loads(socket.recv(timeout=0.2))
                    except TimeoutError:
                        continue
                    if message.get("method") != "Runtime.exceptionThrown":
                        continue
                    details = dict(
                        message.get("params", {}).get("exceptionDetails") or {}
                    )
                    exception = dict(details.get("exception") or {})
                    exceptions.append(
                        str(
                            exception.get("description")
                            or details.get("text")
                            or "Uncaught runtime exception"
                        )
                    )
                if exceptions:
                    return [
                        _diagnostic("runtime_exception", message)
                        for message in exceptions[:20]
                    ]

                result_id = send(
                    "Runtime.evaluate",
                    {
                        "expression": (
                            "({rootChildren:document.getElementById('root')?.childElementCount||0,"
                            "runtimeError:window.__aloyRuntimeError||null,"
                            "bridgeReady:(window.__aloySmokeMessages||[]).some(message=>"
                            "message?.protocol==='1'&&message?.type==='ready')})"
                        ),
                        "returnByValue": True,
                    },
                )
                result: dict[str, Any] = {}
                result_deadline = time.monotonic() + INSPECTION_TIMEOUT_SECONDS
                while time.monotonic() < result_deadline:
                    try:
                        message = json.loads(socket.recv(timeout=0.5))
                    except TimeoutError:
                        continue
                    if message.get("id") == result_id:
                        result = dict(
                            message.get("result", {}).get("result", {}).get("value", {})
                        )
                        break
                diagnostics: list[dict[str, Any]] = []
                if result.get("runtimeError"):
                    diagnostics.append(
                        _diagnostic(
                            "runtime_exception",
                            str(result["runtimeError"]),
                        )
                    )
                if not result.get("bridgeReady"):
                    diagnostics.append(
                        _diagnostic(
                            "runtime_bridge_failed",
                            "The Surface did not acknowledge Aloy's secure bridge",
                        )
                    )
                if int(result.get("rootChildren") or 0) < 1:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_empty_root",
                            "The Surface mounted no visible React root",
                        )
                    )
                if diagnostics or manifest is None:
                    return diagnostics
                for check in manifest.interaction_checks:
                    diagnostics.extend(
                        _execute_interaction_check(
                            socket,
                            send=send,
                            check=check,
                            manifest=manifest,
                            deadline=time.monotonic() + INSPECTION_TIMEOUT_SECONDS,
                        )
                    )
                return diagnostics
            finally:
                socket.close()
        except Exception as exc:
            return [
                _diagnostic(
                    "runtime_inspector_failed",
                    f"Surface runtime inspection failed: {exc}",
                )
            ]
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def _receive_evaluation(
    socket,
    *,
    result_id: int,
    deadline: float,
) -> tuple[Any, list[str]]:
    result: Any = None
    exceptions: list[str] = []
    while time.monotonic() < deadline:
        try:
            message = json.loads(socket.recv(timeout=0.25))
        except TimeoutError:
            continue
        if message.get("method") == "Runtime.exceptionThrown":
            details = dict(message.get("params", {}).get("exceptionDetails") or {})
            exception = dict(details.get("exception") or {})
            exceptions.append(
                str(
                    exception.get("description")
                    or details.get("text")
                    or "Uncaught runtime exception"
                )
            )
            continue
        if message.get("id") != result_id:
            continue
        remote = dict(message.get("result", {}).get("result", {}) or {})
        result = remote.get("value")
        break
    return result, exceptions


def _wait_for_runtime_document(
    socket,
    *,
    send,
    deadline: float,
) -> tuple[bool, list[str]]:
    """Wait until navigation and the host-owned runtime script have settled.

    Chrome exposes a page target before its file navigation has necessarily
    completed. Injecting the MessageChannel into that early target can post to
    the initial document before the Surface SDK installs its listener, or even
    into a document that navigation then replaces. Readiness is therefore an
    explicit inspection stage rather than a timing assumption.
    """
    exceptions: list[str] = []
    expression = (
        "({ready:document.readyState==='complete',"
        "hasRoot:Boolean(document.getElementById('root'))})"
    )
    while time.monotonic() < deadline:
        result_id = send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
            },
        )
        result, observed = _receive_evaluation(
            socket,
            result_id=result_id,
            deadline=min(deadline, time.monotonic() + 0.5),
        )
        exceptions.extend(observed)
        if exceptions:
            return False, exceptions
        if (
            isinstance(result, dict)
            and result.get("ready") is True
            and result.get("hasRoot") is True
        ):
            return True, []
        time.sleep(0.05)
    return False, exceptions


def _interaction_expression(step: dict[str, Any]) -> str:
    payload = json.dumps(step, ensure_ascii=True)
    return (
        "(() => {const step="
        + payload
        + ";const normalize=value=>String(value||'').replace(/\\s+/g,' ').trim().toLowerCase();"
        "const accessibleName=element=>{const direct=element.getAttribute('aria-label');"
        "if(direct)return direct;const labelledBy=element.getAttribute('aria-labelledby');"
        "if(labelledBy){const value=labelledBy.split(/\\s+/).map(id=>document.getElementById(id)?.textContent||'').join(' ').trim();if(value)return value;}"
        'if(element.id){const label=document.querySelector(`label[for="${CSS.escape(element.id)}"]`);if(label?.textContent)return label.textContent;}'
        "const parent=element.closest('label');if(parent?.textContent){const clone=parent.cloneNode(true);clone.querySelectorAll('input,select,textarea,button').forEach(node=>node.remove());const value=clone.textContent?.trim();if(value)return value;}"
        "return element.getAttribute('placeholder')||element.textContent||'';};"
        "const selector=step.role==='button'?'button,[role=button]':step.role==='combobox'?'select,[role=combobox]':'input:not([type=button]):not([type=submit]):not([type=hidden]),textarea,[role=textbox]';"
        "const candidates=[...document.querySelectorAll(selector)].filter(element=>{const style=getComputedStyle(element);const rect=element.getBoundingClientRect();return style.visibility!=='hidden'&&style.display!=='none'&&rect.width>0&&rect.height>0;});"
        "const element=candidates.find(item=>normalize(accessibleName(item))===normalize(step.name));"
        "if(!element)return {ok:false,error:`No visible ${step.role} named ${step.name}`,available:candidates.map(item=>accessibleName(item)).slice(0,30)};"
        "if(element.disabled||element.getAttribute('aria-disabled')==='true')return {ok:false,error:`${step.name} is disabled`};"
        "if(step.action==='click'){element.click();return {ok:true};}"
        "if(step.action==='fill'){const prototype=element instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;const setter=Object.getOwnPropertyDescriptor(prototype,'value')?.set;if(!setter)return {ok:false,error:'Input value setter is unavailable'};setter.call(element,step.value);element.dispatchEvent(new Event('input',{bubbles:true}));element.dispatchEvent(new Event('change',{bubbles:true}));return {ok:true};}"
        "if(step.action==='select'){const setter=Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value')?.set;if(!setter)return {ok:false,error:'Select value setter is unavailable'};setter.call(element,step.value);element.dispatchEvent(new Event('input',{bubbles:true}));element.dispatchEvent(new Event('change',{bubbles:true}));return {ok:true};}"
        "return {ok:false,error:'Unsupported interaction action'};})()"
    )


def _execute_interaction_check(
    socket,
    *,
    send,
    check: SurfaceInteractionCheck,
    manifest: SurfaceManifest,
    deadline: float,
) -> list[dict[str, Any]]:
    before_id = send(
        "Runtime.evaluate",
        {
            "expression": "(window.__aloySmokeMessages||[]).length",
            "returnByValue": True,
        },
    )
    before, exceptions = _receive_evaluation(
        socket, result_id=before_id, deadline=deadline
    )
    if exceptions:
        return [_diagnostic("runtime_exception", item) for item in exceptions[:20]]
    start = int(before or 0)
    for step in check.steps:
        result_id = send(
            "Runtime.evaluate",
            {
                "expression": _interaction_expression(step.model_dump(mode="json")),
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        result, exceptions = _receive_evaluation(
            socket,
            result_id=result_id,
            deadline=min(deadline, time.monotonic() + 2.0),
        )
        if exceptions:
            return [_diagnostic("runtime_exception", item) for item in exceptions[:20]]
        if not isinstance(result, dict) or not result.get("ok"):
            message = (
                str(result.get("error"))
                if isinstance(result, dict)
                else "The interaction step did not complete"
            )
            if isinstance(result, dict) and result.get("available"):
                message += f"; available controls: {result['available']}"
            return [
                _diagnostic(
                    "runtime_interaction_step_failed",
                    f"Interaction check {check.name!r} failed: {message}",
                )
            ]
        time.sleep(0.1)
    messages_id = send(
        "Runtime.evaluate",
        {
            "expression": (
                "(window.__aloySmokeMessages||[]).slice("
                + str(start)
                + ").filter(message=>message?.type==='request')"
            ),
            "returnByValue": True,
        },
    )
    messages, exceptions = _receive_evaluation(
        socket,
        result_id=messages_id,
        deadline=min(deadline, time.monotonic() + 2.0),
    )
    if exceptions:
        return [_diagnostic("runtime_exception", item) for item in exceptions[:20]]
    requests = messages if isinstance(messages, list) else []
    expected = check.expect
    matching = next(
        (
            item
            for item in requests
            if isinstance(item, dict)
            and item.get("method") == expected.method
            and isinstance(item.get("params"), dict)
            and (
                item["params"].get("name") == expected.name
                if expected.method in {"command", "dispatch"}
                else (
                    expected.name == "aloy.ask"
                    if expected.method == "askAloy"
                    else isinstance(item["params"].get("action"), dict)
                    and item["params"]["action"].get("name") == expected.name
                )
            )
        ),
        None,
    )
    if matching is None:
        observed = [
            {
                "method": item.get("method"),
                "name": (item.get("params") or {}).get("name"),
            }
            for item in requests
            if isinstance(item, dict)
        ]
        return [
            _diagnostic(
                "runtime_interaction_missing",
                f"Interaction check {check.name!r} did not send "
                f"{expected.method} {expected.name!r}; observed {observed}",
            )
        ]
    if expected.method in {"command", "dispatch", "requestAction"}:
        declaration = manifest.intents[expected.name]
        params = dict(matching.get("params") or {})
        payload = (
            params.get("payload")
            if expected.method in {"command", "dispatch"}
            else dict(params.get("action") or {}).get("payload")
        )
        try:
            validate_intent_payload(declaration.schema_, payload)
        except ValueError as exc:
            return [
                _diagnostic(
                    "runtime_interaction_invalid_payload",
                    f"Interaction check {check.name!r} sent an invalid payload: {exc}",
                )
            ]
        if expected.method == "command":
            expected_status = (
                "committed" if declaration.interaction_class == "state" else "accepted"
            )
            feedback_expression = (
                "(() => {"
                f"const name={json.dumps(expected.name, ensure_ascii=True)};"
                f"const status={json.dumps(expected_status)};"
                "const candidates=[...document.querySelectorAll('[data-aloy-command-name]')];"
                "const named=candidates.filter(item=>item.getAttribute('data-aloy-command-name')===name);"
                "if(!named.length)return {ok:false,error:`No command feedback for ${name}`};"
                "const settled=named.filter(item=>item.getAttribute('data-aloy-command-status')===status);"
                "if(!settled.length)return {ok:false,error:`Command feedback stayed ${named.map(item=>item.getAttribute('data-aloy-command-status')||'unknown').join(', ')}`};"
                "const element=settled.find(item=>{const style=getComputedStyle(item);"
                "const rect=item.getBoundingClientRect();const text=String(item.textContent||'').replace(/\\s+/g,' ').trim();"
                "return style.visibility!=='hidden'&&style.display!=='none'&&rect.width>0&&rect.height>0&&Boolean(text);});"
                "if(!element)"
                "return {ok:false,error:'Command feedback is not visibly rendered'};"
                "return {ok:true};})()"
            )
            feedback: Any = None
            feedback_exceptions: list[str] = []
            feedback_deadline = min(deadline, time.monotonic() + 1.5)
            while time.monotonic() < feedback_deadline:
                feedback_id = send(
                    "Runtime.evaluate",
                    {
                        "expression": feedback_expression,
                        "returnByValue": True,
                    },
                )
                feedback, feedback_exceptions = _receive_evaluation(
                    socket,
                    result_id=feedback_id,
                    deadline=min(feedback_deadline, time.monotonic() + 0.4),
                )
                if feedback_exceptions:
                    return [
                        _diagnostic("runtime_exception", item)
                        for item in feedback_exceptions[:20]
                    ]
                if isinstance(feedback, dict) and feedback.get("ok"):
                    break
                time.sleep(0.05)
            if not isinstance(feedback, dict) or not feedback.get("ok"):
                message = (
                    str(feedback.get("error"))
                    if isinstance(feedback, dict)
                    else "Command feedback did not render"
                )
                return [
                    _diagnostic(
                        "runtime_command_feedback_missing",
                        f"Interaction check {check.name!r} failed: {message}",
                    )
                ]
    return []


__all__ = ["inspect_surface_runtime"]
