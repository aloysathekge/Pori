"""Browser execution gate for locally built generated Surfaces.

Static validation and compilation cannot detect React render failures caused by
real Event data. Local development therefore mounts the exact host-owned
runtime document in headless Chrome, injects the exact scoped Surface context
through the production MessageChannel protocol, and fails preview on any
uncaught exception, missing bridge acknowledgement, or empty React root.
"""

from __future__ import annotations

import base64
import hashlib
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
    SurfacePrimaryJob,
    validate_intent_payload,
)
from .surface_quality import (
    REQUIRED_SURFACE_STATE_VIEWPORTS,
    REQUIRED_SURFACE_VIEWPORTS,
)
from .surface_resource_states import (
    REQUIRED_SURFACE_STATE_FIXTURES,
    SURFACE_STATE_POLICY_VERSION,
    surface_fixture_applicable,
    surface_state_fixture_context,
)
from .surface_runtime import SurfaceRuntimeDocument

INSPECTION_TIMEOUT_SECONDS = 8.0
SETTLE_SECONDS = 1.5
MAX_SURFACE_CAPTURE_BYTES = 4 * 1024 * 1024

_VIEWPORT_AUDIT_EXPRESSION = r"""
new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => {
  const visible = element => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden'
      && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
  };
  const accessibleName = element => {
    const direct = element.getAttribute('aria-label');
    if (direct && direct.trim()) return direct.trim();
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) {
      const text = labelledBy.split(/\s+/)
        .map(id => document.getElementById(id)?.textContent || '')
        .join(' ').trim();
      if (text) return text;
    }
    if ('labels' in element && element.labels?.length) {
      const text = [...element.labels].map(label => label.textContent || '').join(' ').trim();
      if (text) return text;
    }
    const title = element.getAttribute('title');
    if (title && title.trim()) return title.trim();
    return (element.textContent || '').replace(/\s+/g, ' ').trim();
  };
  const sample = element => ({
    tag: element.tagName.toLowerCase(),
    role: element.getAttribute('role'),
    name: accessibleName(element).slice(0, 120),
    id: (element.id || '').slice(0, 120),
  });
  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const viewportHeight = document.documentElement.clientHeight || window.innerHeight;
  const pageWidth = Math.max(
    document.documentElement.scrollWidth,
    document.body?.scrollWidth || 0,
  );
  const controls = [...document.querySelectorAll(
    'button,a[href],input:not([type=hidden]),select,textarea,[role=button],[role=link],[role=checkbox],[role=radio],[role=combobox],[role=textbox]'
  )].filter(visible).filter(element => !element.disabled && element.getAttribute('aria-disabled') !== 'true');
  const clippedControls = controls.filter(element => {
    const rect = element.getBoundingClientRect();
    return rect.left < -1 || rect.right > viewportWidth + 1;
  });
  const unnamedControls = controls.filter(element => !accessibleName(element));
  const imagesMissingAlt = [...document.querySelectorAll('img:not([alt])')].filter(visible);
  const keyboardUnreachable = controls.filter(element => {
    const role = element.getAttribute('role');
    const native = ['BUTTON','A','INPUT','SELECT','TEXTAREA'].includes(element.tagName);
    return element.tabIndex < 0 && (native
      || ['button','link','checkbox','radio','combobox','textbox'].includes(role || ''));
  });
  const ids = [...document.querySelectorAll('[id]')].map(element => element.id).filter(Boolean);
  const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
  const compact = window.innerWidth <= 768;
  const smallTouchTargets = compact
    ? controls.filter(element => {
        const rect = element.getBoundingClientRect();
        return ['BUTTON','A'].includes(element.tagName) || element.getAttribute('role') === 'button'
          ? rect.width < 44 || rect.height < 44
          : false;
      })
    : [];
  const resourceStateRegions = [...document.querySelectorAll(
    '[data-aloy-resource][data-aloy-resource-state]'
  )].filter(visible).map(element => ({
    resource: element.getAttribute('data-aloy-resource'),
    state: element.getAttribute('data-aloy-resource-state'),
  }));
  const approvalStateRegions = [...document.querySelectorAll(
    '[data-aloy-approval-state]'
  )].filter(visible).map(element => ({
    state: element.getAttribute('data-aloy-approval-state'),
  }));
  const root = document.getElementById('root');
  return resolve({
    viewport: { width: viewportWidth, height: viewportHeight },
    layout: {
      page_width: pageWidth,
      horizontal_overflow_px: Math.max(0, pageWidth - viewportWidth),
      root_children: root?.childElementCount || 0,
      clipped_controls: clippedControls.length,
      clipped_control_samples: clippedControls.slice(0, 10).map(sample),
    },
    accessibility: {
      main_landmarks: document.querySelectorAll('main,[role=main]').length,
      controls: controls.length,
      unnamed_controls: unnamedControls.length,
      unnamed_control_samples: unnamedControls.slice(0, 10).map(sample),
      images_missing_alt: imagesMissingAlt.length,
      image_samples: imagesMissingAlt.slice(0, 10).map(sample),
      keyboard_unreachable: keyboardUnreachable.length,
      keyboard_unreachable_samples: keyboardUnreachable.slice(0, 10).map(sample),
      duplicate_ids: duplicateIds.slice(0, 20),
      small_touch_targets: smallTouchTargets.length,
      small_touch_target_samples: smallTouchTargets.slice(0, 10).map(sample),
      resource_state_regions: resourceStateRegions.slice(0, 20),
      approval_state_regions: approvalStateRegions.slice(0, 20),
    },
  });
})))
""".strip()

_CONTRAST_AUDIT_EXPRESSION = r"""
(() => {
  const visible = element => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden'
      && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
  };
  const parseColor = value => {
    if (!value || value === 'transparent') return [0, 0, 0, 0];
    const match = value.match(/^rgba?\((.*)\)$/i);
    if (!match) return null;
    const parts = match[1].trim().split(/[\s,\/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.some(Number.isNaN)) return null;
    return [parts[0], parts[1], parts[2], parts.length > 3 ? parts[3] : 1];
  };
  const blend = (front, back) => {
    const alpha = front[3] + back[3] * (1 - front[3]);
    if (alpha <= 0) return [255, 255, 255, 1];
    return [
      (front[0] * front[3] + back[0] * back[3] * (1 - front[3])) / alpha,
      (front[1] * front[3] + back[1] * back[3] * (1 - front[3])) / alpha,
      (front[2] * front[3] + back[2] * back[3] * (1 - front[3])) / alpha,
      alpha,
    ];
  };
  const background = element => {
    const layers = [];
    let node = element;
    while (node instanceof Element) {
      const style = getComputedStyle(node);
      if (style.backgroundImage && style.backgroundImage !== 'none') {
        return { color: null, reason: 'background_image' };
      }
      const color = parseColor(style.backgroundColor);
      if (!color) return { color: null, reason: 'unparsed_background' };
      layers.push(color);
      node = node.parentElement;
    }
    let result = [255, 255, 255, 1];
    for (const layer of layers.reverse()) result = blend(layer, result);
    return { color: result, reason: null };
  };
  const luminance = color => {
    const channels = color.slice(0, 3).map(value => {
      const normalized = value / 255;
      return normalized <= 0.04045
        ? normalized / 12.92
        : Math.pow((normalized + 0.055) / 1.055, 2.4);
    });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  const ratio = (first, second) => {
    const a = luminance(first);
    const b = luminance(second);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const elements = [];
  const seen = new Set();
  let node;
  while ((node = walker.nextNode())) {
    if (!(node.textContent || '').trim()) continue;
    const element = node.parentElement;
    if (!element || seen.has(element) || !visible(element)) continue;
    if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(element.tagName)) continue;
    if (element.closest('[aria-hidden=true],[hidden],[disabled],[aria-disabled=true]')) continue;
    seen.add(element);
    elements.push(element);
  }
  const samples = [];
  let measured = 0;
  let failures = 0;
  let unmeasurable = 0;
  let minimum = null;
  for (const element of elements.slice(0, 1000)) {
    const style = getComputedStyle(element);
    const foreground = parseColor(style.color);
    const backdrop = background(element);
    const text = (element.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 120);
    if (!foreground || !backdrop.color) {
      unmeasurable += 1;
      if (samples.length < 20) samples.push({ text, outcome: 'unmeasurable', reason: backdrop.reason });
      continue;
    }
    const renderedForeground = blend(foreground, backdrop.color);
    const contrast = ratio(renderedForeground, backdrop.color);
    const fontSize = Number.parseFloat(style.fontSize) || 0;
    const fontWeight = Number.parseInt(style.fontWeight, 10) || 400;
    const large = fontSize >= 24 || (fontSize >= 18.66 && fontWeight >= 700);
    const required = large ? 3 : 4.5;
    const passed = contrast + 0.001 >= required;
    measured += 1;
    if (!passed) failures += 1;
    minimum = minimum === null ? contrast : Math.min(minimum, contrast);
    if ((!passed || samples.length < 5) && samples.length < 20) {
      samples.push({
        text,
        outcome: passed ? 'passed' : 'failed',
        ratio: Number(contrast.toFixed(3)),
        required,
        font_size_px: fontSize,
        font_weight: fontWeight,
        foreground: style.color,
        background: backdrop.color.map(value => Number(value.toFixed(2))),
      });
    }
  }
  return {
    policy_version: 'aloy-surface-contrast@1',
    passed: failures === 0 && unmeasurable === 0,
    text_nodes: elements.length,
    measured,
    failures,
    unmeasurable,
    minimum_ratio: minimum === null ? null : Number(minimum.toFixed(3)),
    samples,
  };
})()
""".strip()

_FOCUS_SETUP_EXPRESSION = r"""
(() => {
  const visible = element => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden'
      && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
  };
  const candidates = [...document.querySelectorAll(
    'button,a[href],input:not([type=hidden]),select,textarea,[contenteditable=true],[tabindex]'
  )].filter(visible).filter(element => !element.disabled
    && element.getAttribute('aria-disabled') !== 'true' && element.tabIndex >= 0);
  const radioGroups = new Map();
  for (const element of candidates) {
    if (!(element instanceof HTMLInputElement) || element.type !== 'radio' || !element.name) continue;
    const group = radioGroups.get(element.name) || [];
    group.push(element);
    radioGroups.set(element.name, group);
  }
  const controls = candidates.filter(element => {
    if (!(element instanceof HTMLInputElement) || element.type !== 'radio' || !element.name) return true;
    const group = radioGroups.get(element.name) || [];
    const checked = group.find(item => item.checked);
    return element === (checked || group[0]);
  }).slice(0, 200);
  const style = element => {
    const value = getComputedStyle(element);
    return {
      outline_style: value.outlineStyle,
      outline_width: Number.parseFloat(value.outlineWidth) || 0,
      outline_color: value.outlineColor,
      box_shadow: value.boxShadow,
      border_color: value.borderColor,
      border_width: value.borderWidth,
      background_color: value.backgroundColor,
      color: value.color,
    };
  };
  window.__aloyFocusBaseline = {};
  controls.forEach((element, index) => {
    const id = String(index);
    element.setAttribute('data-aloy-focus-index', id);
    window.__aloyFocusBaseline[id] = style(element);
  });
  document.activeElement?.blur();
  document.body.setAttribute('tabindex', '-1');
  document.body.focus();
  return { count: controls.length };
})()
""".strip()

_FOCUS_OBSERVATION_EXPRESSION = r"""
(() => {
  const element = document.activeElement;
  if (!(element instanceof Element)) return { index: null, visible_indicator: false };
  const index = element.getAttribute('data-aloy-focus-index');
  const value = getComputedStyle(element);
  const baseline = (window.__aloyFocusBaseline || {})[index] || {};
  const outlineWidth = Number.parseFloat(value.outlineWidth) || 0;
  const outlineVisible = !['none', 'hidden'].includes(value.outlineStyle)
    && outlineWidth >= 1 && value.outlineColor !== 'transparent';
  const parseColor = color => {
    if (!color || color === 'transparent') return null;
    const match = color.match(/^rgba?\((.*)\)$/i);
    if (!match) return null;
    const parts = match[1].trim().split(/[\s,\/]+/).filter(Boolean).map(Number);
    return parts.length >= 3 && !parts.some(Number.isNaN)
      && (parts.length < 4 || parts[3] > 0) ? parts.slice(0, 3) : null;
  };
  const luminance = color => {
    const channels = color.map(item => {
      const normalized = item / 255;
      return normalized <= 0.04045
        ? normalized / 12.92
        : Math.pow((normalized + 0.055) / 1.055, 2.4);
    });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  let backgroundElement = element.parentElement;
  let backgroundColor = null;
  while (backgroundElement && !backgroundColor) {
    backgroundColor = parseColor(getComputedStyle(backgroundElement).backgroundColor);
    backgroundElement = backgroundElement.parentElement;
  }
  backgroundColor = backgroundColor || [255, 255, 255];
  const outlineColor = parseColor(value.outlineColor);
  const outlineContrast = outlineColor
    ? (Math.max(luminance(outlineColor), luminance(backgroundColor)) + 0.05)
      / (Math.min(luminance(outlineColor), luminance(backgroundColor)) + 0.05)
    : null;
  const shadowChanged = value.boxShadow !== baseline.box_shadow && value.boxShadow !== 'none';
  const borderChanged = value.borderColor !== baseline.border_color
    || value.borderWidth !== baseline.border_width;
  const fillChanged = value.backgroundColor !== baseline.background_color
    || value.color !== baseline.color;
  return {
    index,
    tag: element.tagName.toLowerCase(),
    name: (element.getAttribute('aria-label') || element.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 120),
    visible_indicator: outlineVisible || shadowChanged || borderChanged || fillChanged,
    strong_outline: outlineVisible && outlineWidth >= 2
      && outlineContrast !== null && outlineContrast >= 3,
    signals: {
      outline: outlineVisible,
      outline_width: outlineWidth,
      outline_style: value.outlineStyle,
      outline_color: value.outlineColor,
      outline_contrast: outlineContrast === null ? null : Number(outlineContrast.toFixed(3)),
      shadow_changed: shadowChanged,
      border_changed: borderChanged,
      fill_changed: fillChanged,
    },
  };
})()
""".strip()


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
    evidence_sink: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute the exact runtime and every manifest-declared UI interaction.

    The checks use accessible roles and names rather than model-controlled CSS
    selectors. Each declared intent must be reachable through a visible user
    path and produce the expected typed SDK request before publication.
    """
    inspection_started = time.monotonic()
    timings: dict[str, Any] = {"policy_version": "aloy-surface-timings@2"}
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
            socket = connect(
                str(target["webSocketDebuggerUrl"]),
                open_timeout=2,
                max_size=16 * 1024 * 1024,
            )
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
                send("Page.enable")
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
                    "window.__aloySmokeContext=currentContext;"
                    "window.__aloySetSmokeContext=next=>{currentContext=next;window.__aloySmokeContext=currentContext;"
                    "channel.port1.postMessage({protocol:'1',type:'context',"
                    "sessionId:'runtime-smoke',context:currentContext});};"
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
                    "command_attempts:[...((projected.data||{}).command_attempts||[]),attempt]}};window.__aloySmokeContext=currentContext;"
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
                if diagnostics:
                    return diagnostics
                runtime_ready_at = time.monotonic()
                timings["runtime_bootstrap_ms"] = round(
                    (runtime_ready_at - inspection_started) * 1000,
                    3,
                )
                viewport_started = time.monotonic()
                viewport_diagnostics, viewport_evidence, captures = (
                    _inspect_viewport_matrix(socket, send=send)
                )
                timings["viewport_matrix_ms"] = round(
                    (time.monotonic() - viewport_started) * 1000,
                    3,
                )
                if evidence_sink is not None:
                    evidence_sink["viewport_matrix"] = viewport_evidence
                    evidence_sink["_capture_blobs"] = captures
                diagnostics.extend(viewport_diagnostics)
                if diagnostics:
                    timings["state_matrix_ms"] = 0.0
                    timings["interaction_checks_ms"] = 0.0
                    timings["primary_jobs_ms"] = 0.0
                    timings["total_ms"] = round(
                        (time.monotonic() - inspection_started) * 1000,
                        3,
                    )
                    if evidence_sink is not None:
                        evidence_sink["timings"] = timings
                    return diagnostics
                state_started = time.monotonic()
                state_diagnostics, state_evidence = _inspect_state_matrix(
                    socket,
                    send=send,
                    context=context,
                    manifest=manifest,
                )
                timings["state_matrix_ms"] = round(
                    (time.monotonic() - state_started) * 1000,
                    3,
                )
                if evidence_sink is not None:
                    evidence_sink["state_matrix"] = state_evidence
                diagnostics.extend(state_diagnostics)
                if diagnostics or manifest is None:
                    timings["interaction_checks_ms"] = 0.0
                    timings["primary_jobs_ms"] = 0.0
                    timings["total_ms"] = round(
                        (time.monotonic() - inspection_started) * 1000,
                        3,
                    )
                    if evidence_sink is not None:
                        evidence_sink["timings"] = timings
                    return diagnostics
                # Interaction paths execute at a stable wide composition after
                # every required responsive composition has been inspected.
                wide = REQUIRED_SURFACE_VIEWPORTS[0]
                send(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": wide["width"],
                        "height": wide["height"],
                        "deviceScaleFactor": 1,
                        "mobile": False,
                    },
                )
                primary_job_started = time.monotonic()
                primary_job_evidence: list[dict[str, Any]] = []
                for job in manifest.primary_jobs:
                    job_diagnostics, job_evidence = _execute_primary_job(
                        socket,
                        send=send,
                        job=job,
                        manifest=manifest,
                        context=context,
                        deadline=time.monotonic() + INSPECTION_TIMEOUT_SECONDS,
                    )
                    diagnostics.extend(job_diagnostics)
                    primary_job_evidence.append(job_evidence)
                timings["primary_jobs_ms"] = round(
                    (time.monotonic() - primary_job_started) * 1000,
                    3,
                )
                if evidence_sink is not None:
                    evidence_sink["primary_jobs"] = {
                        "policy_version": "aloy-surface-primary-jobs@1",
                        "required": [job.id for job in manifest.primary_jobs],
                        "passed": not any(
                            item.get("passed") is not True
                            for item in primary_job_evidence
                        ),
                        "jobs": primary_job_evidence,
                    }
                interaction_started = time.monotonic()
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
                timings["interaction_checks_ms"] = round(
                    (time.monotonic() - interaction_started) * 1000,
                    3,
                )
                timings["total_ms"] = round(
                    (time.monotonic() - inspection_started) * 1000,
                    3,
                )
                if evidence_sink is not None:
                    evidence_sink["timings"] = timings
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


def _receive_command_result(
    socket,
    *,
    result_id: int,
    deadline: float,
) -> dict[str, Any] | None:
    while time.monotonic() < deadline:
        try:
            message = json.loads(socket.recv(timeout=0.25))
        except TimeoutError:
            continue
        if message.get("id") != result_id:
            continue
        if message.get("error"):
            return None
        result = message.get("result")
        return dict(result) if isinstance(result, dict) else None
    return None


def _viewport_diagnostic(
    code: str,
    message: str,
    *,
    viewport: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = _diagnostic(code, message)
    diagnostic["viewport"] = str(viewport["id"])
    return diagnostic


def _inspect_contrast(
    socket,
    *,
    send,
    viewport: dict[str, Any],
    state: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_id = send(
        "Runtime.evaluate",
        {
            "expression": _CONTRAST_AUDIT_EXPRESSION,
            "returnByValue": True,
        },
    )
    result, exceptions = _receive_evaluation(
        socket,
        result_id=result_id,
        deadline=time.monotonic() + 3.0,
    )
    diagnostics: list[dict[str, Any]] = []
    if exceptions or not isinstance(result, dict):
        for message in exceptions[:10] or [
            "No deterministic contrast evidence returned"
        ]:
            diagnostic = _viewport_diagnostic(
                "contrast_audit_failed",
                message,
                viewport=viewport,
            )
            if state is not None:
                diagnostic["state"] = state
            diagnostics.append(diagnostic)
        return diagnostics, {}
    if result.get("passed") is not True:
        failures = int(result.get("failures") or 0)
        unmeasurable = int(result.get("unmeasurable") or 0)
        diagnostic = _viewport_diagnostic(
            "contrast_text_failed",
            (
                f"{failures} text region(s) miss the WCAG contrast threshold and "
                f"{unmeasurable} region(s) have no deterministic solid backdrop"
            ),
            viewport=viewport,
        )
        if state is not None:
            diagnostic["state"] = state
        diagnostics.append(diagnostic)
    return diagnostics, result


def _inspect_keyboard_focus(
    socket,
    *,
    send,
    viewport: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    setup_id = send(
        "Runtime.evaluate",
        {"expression": _FOCUS_SETUP_EXPRESSION, "returnByValue": True},
    )
    setup, exceptions = _receive_evaluation(
        socket,
        result_id=setup_id,
        deadline=time.monotonic() + 3.0,
    )
    if exceptions or not isinstance(setup, dict):
        return [
            _viewport_diagnostic(
                "focus_audit_failed",
                (exceptions[0] if exceptions else "Focus setup returned no evidence"),
                viewport=viewport,
            )
        ], {}
    expected = int(setup.get("count") or 0)
    visited: list[dict[str, Any]] = []
    visited_indexes: set[str] = set()
    repeated_before_complete = False
    for _ in range(expected + 2):
        key_down = send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyDown",
                "key": "Tab",
                "code": "Tab",
                "windowsVirtualKeyCode": 9,
                "nativeVirtualKeyCode": 9,
            },
        )
        _receive_command_result(
            socket,
            result_id=key_down,
            deadline=time.monotonic() + 1.0,
        )
        key_up = send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyUp",
                "key": "Tab",
                "code": "Tab",
                "windowsVirtualKeyCode": 9,
                "nativeVirtualKeyCode": 9,
            },
        )
        _receive_command_result(
            socket,
            result_id=key_up,
            deadline=time.monotonic() + 1.0,
        )
        observation_id = send(
            "Runtime.evaluate",
            {"expression": _FOCUS_OBSERVATION_EXPRESSION, "returnByValue": True},
        )
        observation, focus_exceptions = _receive_evaluation(
            socket,
            result_id=observation_id,
            deadline=time.monotonic() + 2.0,
        )
        if focus_exceptions or not isinstance(observation, dict):
            break
        index = observation.get("index")
        if index is None:
            if len(visited_indexes) >= expected:
                break
            continue
        index = str(index)
        if index in visited_indexes:
            if len(visited_indexes) < expected:
                repeated_before_complete = True
            break
        visited_indexes.add(index)
        visited.append(observation)
        if len(visited_indexes) >= expected:
            break

    cleanup_id = send(
        "Runtime.evaluate",
        {
            "expression": (
                "document.activeElement?.blur();"
                "document.body.removeAttribute('tabindex');"
                "document.querySelectorAll('[data-aloy-focus-index]').forEach("
                "element=>element.removeAttribute('data-aloy-focus-index'));true"
            ),
            "returnByValue": True,
        },
    )
    _receive_evaluation(
        socket,
        result_id=cleanup_id,
        deadline=time.monotonic() + 2.0,
    )
    expected_indexes = {str(index) for index in range(expected)}
    missing = sorted(expected_indexes - visited_indexes, key=int)
    missing_indicators = [
        item for item in visited if item.get("visible_indicator") is not True
    ]
    diagnostics: list[dict[str, Any]] = []
    if missing:
        diagnostics.append(
            _viewport_diagnostic(
                "focus_order_unreachable",
                f"Keyboard traversal did not reach {len(missing)} visible control(s)",
                viewport=viewport,
            )
        )
    if repeated_before_complete:
        diagnostics.append(
            _viewport_diagnostic(
                "focus_keyboard_trap",
                "Keyboard traversal repeated before reaching every visible control",
                viewport=viewport,
            )
        )
    if missing_indicators:
        diagnostics.append(
            _viewport_diagnostic(
                "focus_indicator_missing",
                (
                    f"{len(missing_indicators)} keyboard-focusable control(s) expose "
                    "no visible focus indicator"
                ),
                viewport=viewport,
            )
        )
    evidence = {
        "policy_version": "aloy-surface-focus@1",
        "passed": not diagnostics,
        "controls": expected,
        "visited": len(visited_indexes),
        "missing_indexes": missing,
        "trap_detected": repeated_before_complete,
        "visible_indicators": len(visited) - len(missing_indicators),
        "strong_outline_indicators": sum(
            1 for item in visited if item.get("strong_outline") is True
        ),
        "observations": visited,
    }
    return diagnostics, evidence


def _inspect_viewport_matrix(
    socket,
    *,
    send,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    for viewport in REQUIRED_SURFACE_VIEWPORTS:
        command_id = send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": viewport["width"],
                "height": viewport["height"],
                "deviceScaleFactor": 1,
                "mobile": bool(viewport["compact"]),
            },
        )
        if (
            _receive_command_result(
                socket,
                result_id=command_id,
                deadline=time.monotonic() + 2.0,
            )
            is None
        ):
            diagnostics.append(
                _viewport_diagnostic(
                    "viewport_emulation_failed",
                    "The trusted browser could not apply a required viewport",
                    viewport=viewport,
                )
            )
            continue
        result_id = send(
            "Runtime.evaluate",
            {
                "expression": _VIEWPORT_AUDIT_EXPRESSION,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        result, exceptions = _receive_evaluation(
            socket,
            result_id=result_id,
            deadline=time.monotonic() + 3.0,
        )
        if exceptions or not isinstance(result, dict):
            diagnostics.extend(
                _viewport_diagnostic(
                    "viewport_audit_failed",
                    message,
                    viewport=viewport,
                )
                for message in (
                    exceptions[:10]
                    or ["The required viewport returned no audit evidence"]
                )
            )
            continue
        observation = {
            "id": viewport["id"],
            "requested_width": viewport["width"],
            "requested_height": viewport["height"],
            "compact": viewport["compact"],
            **result,
        }
        layout = dict(result.get("layout") or {})
        accessibility = dict(result.get("accessibility") or {})
        if int(layout.get("horizontal_overflow_px") or 0) > 1:
            diagnostics.append(
                _viewport_diagnostic(
                    "viewport_page_overflow",
                    "The Surface creates page-level horizontal overflow",
                    viewport=viewport,
                )
            )
        if int(layout.get("clipped_controls") or 0) > 0:
            diagnostics.append(
                _viewport_diagnostic(
                    "viewport_clipped_control",
                    "A visible interactive control is clipped horizontally",
                    viewport=viewport,
                )
            )
        if int(accessibility.get("main_landmarks") or 0) != 1:
            diagnostics.append(
                _viewport_diagnostic(
                    "accessibility_main_landmark",
                    "The Surface must expose exactly one visible main landmark",
                    viewport=viewport,
                )
            )
        if int(accessibility.get("unnamed_controls") or 0) > 0:
            diagnostics.append(
                _viewport_diagnostic(
                    "accessibility_unnamed_control",
                    "A visible interactive control has no accessible name",
                    viewport=viewport,
                )
            )
        if int(accessibility.get("images_missing_alt") or 0) > 0:
            diagnostics.append(
                _viewport_diagnostic(
                    "accessibility_image_alt",
                    "A visible image has no alt attribute",
                    viewport=viewport,
                )
            )
        if int(accessibility.get("keyboard_unreachable") or 0) > 0:
            diagnostics.append(
                _viewport_diagnostic(
                    "accessibility_keyboard_unreachable",
                    "A custom interactive control is not keyboard reachable",
                    viewport=viewport,
                )
            )
        if accessibility.get("duplicate_ids"):
            diagnostics.append(
                _viewport_diagnostic(
                    "accessibility_duplicate_id",
                    "The Surface contains duplicate DOM ids",
                    viewport=viewport,
                )
            )
        contrast_diagnostics, contrast_evidence = _inspect_contrast(
            socket,
            send=send,
            viewport=viewport,
        )
        diagnostics.extend(contrast_diagnostics)
        observation["contrast"] = contrast_evidence
        screenshot_id = send(
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
            },
        )
        screenshot = _receive_command_result(
            socket,
            result_id=screenshot_id,
            deadline=time.monotonic() + 3.0,
        )
        encoded = str((screenshot or {}).get("data") or "")
        try:
            png = base64.b64decode(encoded, validate=True)
        except ValueError:
            png = b""
        if not png or len(png) > MAX_SURFACE_CAPTURE_BYTES:
            diagnostics.append(
                _viewport_diagnostic(
                    "viewport_capture_failed",
                    (
                        "The trusted browser produced no viewport capture"
                        if not png
                        else "The trusted viewport capture exceeded the safe size limit"
                    ),
                    viewport=viewport,
                )
            )
        else:
            checksum = hashlib.sha256(png).hexdigest()
            observation["capture"] = {
                "sha256": checksum,
                "size_bytes": len(png),
                "content_type": "image/png",
            }
            captures.append(
                {
                    "name": f"{viewport['id']}.png",
                    "content_type": "image/png",
                    "sha256": checksum,
                    "data": png,
                }
            )
        focus_diagnostics, focus_evidence = _inspect_keyboard_focus(
            socket,
            send=send,
            viewport=viewport,
        )
        diagnostics.extend(focus_diagnostics)
        observation["focus"] = focus_evidence
        observations.append(observation)
    evidence = {
        "policy_version": "aloy-surface-viewports@1",
        "required": [str(item["id"]) for item in REQUIRED_SURFACE_VIEWPORTS],
        "passed": not any(
            str(item.get("code") or "").startswith(("viewport_", "accessibility_"))
            for item in diagnostics
        )
        and len(observations) == len(REQUIRED_SURFACE_VIEWPORTS),
        "viewports": observations,
    }
    return diagnostics, evidence, captures


def _inspect_state_matrix(
    socket,
    *,
    send,
    context: dict[str, Any],
    manifest: SurfaceManifest | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exercise public non-ready states at trusted desktop and mobile sizes."""
    diagnostics: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    viewports = {
        str(item["id"]): item
        for item in REQUIRED_SURFACE_VIEWPORTS
        if str(item["id"]) in REQUIRED_SURFACE_STATE_VIEWPORTS
    }
    approval_contract_applicable = surface_fixture_applicable(
        manifest,
        "approval_required",
        list(context.get("capabilities") or []),
    )
    for state in REQUIRED_SURFACE_STATE_FIXTURES:
        fixture = surface_state_fixture_context(context, state, manifest=manifest)
        applicable = surface_fixture_applicable(
            manifest,
            state,
            list(context.get("capabilities") or []),
        )
        for viewport_id in REQUIRED_SURFACE_STATE_VIEWPORTS:
            viewport = viewports[viewport_id]
            command_id = send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": viewport["width"],
                    "height": viewport["height"],
                    "deviceScaleFactor": 1,
                    "mobile": bool(viewport["compact"]),
                },
            )
            if (
                _receive_command_result(
                    socket,
                    result_id=command_id,
                    deadline=time.monotonic() + 2.0,
                )
                is None
            ):
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_viewport_emulation_failed",
                        f"The trusted browser could not apply the {state} fixture viewport",
                        viewport=viewport,
                    )
                )
                continue
            set_id = send(
                "Runtime.evaluate",
                {
                    "expression": (
                        "new Promise(resolve=>{window.__aloySetSmokeContext("
                        + json.dumps(fixture, ensure_ascii=True, default=str)
                        + ");requestAnimationFrame(()=>requestAnimationFrame("
                        "()=>setTimeout(()=>resolve(true),50)));})"
                    ),
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            _, exceptions = _receive_evaluation(
                socket,
                result_id=set_id,
                deadline=time.monotonic() + 3.0,
            )
            if exceptions:
                diagnostics.extend(
                    _viewport_diagnostic(
                        "state_runtime_exception",
                        f"The {state} state failed: {message}",
                        viewport=viewport,
                    )
                    for message in exceptions[:10]
                )
                continue
            audit_id = send(
                "Runtime.evaluate",
                {
                    "expression": _VIEWPORT_AUDIT_EXPRESSION,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            result, exceptions = _receive_evaluation(
                socket,
                result_id=audit_id,
                deadline=time.monotonic() + 3.0,
            )
            if exceptions or not isinstance(result, dict):
                diagnostics.extend(
                    _viewport_diagnostic(
                        "state_audit_failed",
                        f"The {state} state returned no valid audit evidence: {message}",
                        viewport=viewport,
                    )
                    for message in (
                        exceptions[:10] or ["no browser observation was returned"]
                    )
                )
                continue
            layout = dict(result.get("layout") or {})
            accessibility = dict(result.get("accessibility") or {})
            expected_resource_states = dict(fixture.get("resource_states") or {})
            expected_resources = set(expected_resource_states)
            state_regions = list(accessibility.get("resource_state_regions") or [])
            matching_regions = [
                item
                for item in state_regions
                if isinstance(item, dict)
                and item.get("resource") in expected_resources
                and item.get("state")
                == dict(expected_resource_states[str(item.get("resource"))]).get(
                    "status"
                )
            ]
            approval_regions = list(accessibility.get("approval_state_regions") or [])
            if int(layout.get("root_children") or 0) < 1:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_empty_root",
                        f"The {state} state mounted no visible React root",
                        viewport=viewport,
                    )
                )
            if int(layout.get("horizontal_overflow_px") or 0) > 1:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_viewport_overflow",
                        f"The {state} state creates page-level horizontal overflow",
                        viewport=viewport,
                    )
                )
            if int(layout.get("clipped_controls") or 0) > 0:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_clipped_control",
                        f"The {state} state clips a visible interactive control",
                        viewport=viewport,
                    )
                )
            if int(accessibility.get("main_landmarks") or 0) != 1:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_main_landmark",
                        f"The {state} state must expose exactly one main landmark",
                        viewport=viewport,
                    )
                )
            if int(accessibility.get("unnamed_controls") or 0) > 0:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_unnamed_control",
                        f"The {state} state contains an unnamed control",
                        viewport=viewport,
                    )
                )
            if applicable and expected_resources and not matching_regions:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_region_missing",
                        (
                            f"The {state} state exposes no visible SDK-bound "
                            "resource region"
                        ),
                        viewport=viewport,
                    )
                )
            expected_approval_state = (
                "required" if state == "approval_required" else "clear"
            )
            matching_approval_regions = sum(
                1
                for item in approval_regions
                if isinstance(item, dict)
                and item.get("state") == expected_approval_state
            )
            if approval_contract_applicable and matching_approval_regions < 1:
                diagnostics.append(
                    _viewport_diagnostic(
                        "state_approval_region_missing",
                        (
                            "The Surface exposes no visible SDK-bound approval "
                            f"summary in the expected {expected_approval_state} state"
                        ),
                        viewport=viewport,
                    )
                )
            contrast_diagnostics, contrast_evidence = _inspect_contrast(
                socket,
                send=send,
                viewport=viewport,
                state=state,
            )
            diagnostics.extend(contrast_diagnostics)
            observation = {
                "state": state,
                "viewport_id": viewport_id,
                "requested_width": viewport["width"],
                "requested_height": viewport["height"],
                "applicable": applicable,
                "fixture_bytes": len(
                    json.dumps(
                        fixture,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        default=str,
                    ).encode("utf-8")
                ),
                "layout": layout,
                "accessibility": accessibility,
                "matching_resource_regions": len(matching_regions),
                "expected_approval_state": (
                    expected_approval_state if approval_contract_applicable else None
                ),
                "matching_approval_regions": matching_approval_regions,
                "contrast": contrast_evidence,
            }
            observation["fingerprint"] = hashlib.sha256(
                json.dumps(
                    observation,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            observations.append(observation)

    # Interaction checks must always start from the canonical ready context.
    reset_id = send(
        "Runtime.evaluate",
        {
            "expression": (
                "window.__aloySetSmokeContext("
                + json.dumps(context, ensure_ascii=True, default=str)
                + ");true"
            ),
            "returnByValue": True,
        },
    )
    _receive_evaluation(
        socket,
        result_id=reset_id,
        deadline=time.monotonic() + 2.0,
    )
    evidence = {
        "policy_version": SURFACE_STATE_POLICY_VERSION,
        "required_states": list(REQUIRED_SURFACE_STATE_FIXTURES),
        "required_viewports": list(REQUIRED_SURFACE_STATE_VIEWPORTS),
        "passed": not any(
            str(item.get("code") or "").startswith("state_") for item in diagnostics
        )
        and len(observations)
        == len(REQUIRED_SURFACE_STATE_FIXTURES) * len(REQUIRED_SURFACE_STATE_VIEWPORTS),
        "observations": observations,
    }
    return diagnostics, evidence


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


def _visible_assertion_expression(assertion: dict[str, Any]) -> str:
    payload = json.dumps(assertion, ensure_ascii=True)
    return (
        "(() => {const assertion="
        + payload
        + ";const normalize=value=>String(value||'').replace(/\\s+/g,' ').trim().toLowerCase();"
        "const name=element=>{const direct=element.getAttribute('aria-label');if(direct)return direct;"
        "const labelledBy=element.getAttribute('aria-labelledby');if(labelledBy){const value=labelledBy.split(/\\s+/).map(id=>document.getElementById(id)?.textContent||'').join(' ').trim();if(value)return value;}"
        "return element.textContent||'';};"
        "const selectors={button:'button,[role=button]',textbox:'input:not([type=hidden]),textarea,[role=textbox]',combobox:'select,[role=combobox]',heading:'h1,h2,h3,h4,h5,h6,[role=heading]',region:'main,section,[role=region]',status:'[role=status]'};"
        "const candidates=[...document.querySelectorAll(selectors[assertion.role]||'')].filter(element=>{const style=getComputedStyle(element);const rect=element.getBoundingClientRect();return style.visibility!=='hidden'&&style.display!=='none'&&rect.width>0&&rect.height>0;});"
        "const match=candidates.find(element=>normalize(name(element))===normalize(assertion.name));"
        "return match?{ok:true}:{ok:false,error:`No visible ${assertion.role} named ${assertion.name}`,available:candidates.map(name).slice(0,30)};})()"
    )


def _request_name(request: dict[str, Any]) -> str | None:
    params = request.get("params")
    if not isinstance(params, dict):
        return None
    if request.get("method") in {"command", "dispatch"}:
        return str(params.get("name") or "")
    if request.get("method") == "askAloy":
        return "aloy.ask"
    action = params.get("action")
    return str(action.get("name") or "") if isinstance(action, dict) else None


def _execute_primary_job(
    socket,
    *,
    send,
    job: SurfacePrimaryJob,
    manifest: SurfaceManifest,
    context: dict[str, Any],
    deadline: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.monotonic()
    diagnostics: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    reset_id = send(
        "Runtime.evaluate",
        {
            "expression": (
                "new Promise(resolve=>{window.__aloySetSmokeContext("
                + json.dumps(context, ensure_ascii=True, default=str)
                + ");requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(true)));})"
            ),
            "awaitPromise": True,
            "returnByValue": True,
        },
    )
    _, exceptions = _receive_evaluation(socket, result_id=reset_id, deadline=deadline)
    if exceptions:
        diagnostics.extend(_diagnostic("runtime_exception", item) for item in exceptions[:20])
    before_id = send(
        "Runtime.evaluate",
        {
            "expression": "(window.__aloySmokeMessages||[]).length",
            "returnByValue": True,
        },
    )
    before, exceptions = _receive_evaluation(socket, result_id=before_id, deadline=deadline)
    if exceptions:
        diagnostics.extend(_diagnostic("runtime_exception", item) for item in exceptions[:20])
    start = int(before or 0)
    if not diagnostics:
        for step in job.steps:
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
                diagnostics.extend(
                    _diagnostic("runtime_exception", item) for item in exceptions[:20]
                )
                break
            if not isinstance(result, dict) or result.get("ok") is not True:
                diagnostics.append(
                    _diagnostic(
                        "primary_job_step_failed",
                        f"Primary job {job.description!r} failed: "
                        + (str(result.get("error")) if isinstance(result, dict) else "step failed"),
                    )
                )
                break
            time.sleep(0.1)

    snapshot_id = send(
        "Runtime.evaluate",
        {
            "expression": (
                "({requests:(window.__aloySmokeMessages||[]).slice("
                + str(start)
                + ").filter(message=>message?.type==='request'),context:window.__aloySmokeContext||null})"
            ),
            "returnByValue": True,
        },
    )
    snapshot, exceptions = _receive_evaluation(
        socket, result_id=snapshot_id, deadline=min(deadline, time.monotonic() + 2.0)
    )
    if exceptions:
        diagnostics.extend(_diagnostic("runtime_exception", item) for item in exceptions[:20])
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    request_values = snapshot.get("requests")
    requests: list[Any] = request_values if isinstance(request_values, list) else []
    context_value = snapshot.get("context")
    current_context: dict[str, Any] = (
        context_value if isinstance(context_value, dict) else {}
    )

    if not diagnostics:
        for assertion in job.assertions:
            outcome: dict[str, Any] = {"kind": assertion.kind, "passed": False}
            if assertion.kind == "visible":
                result_id = send(
                    "Runtime.evaluate",
                    {
                        "expression": _visible_assertion_expression(assertion.model_dump(mode="json")),
                        "returnByValue": True,
                    },
                )
                result, exceptions = _receive_evaluation(
                    socket,
                    result_id=result_id,
                    deadline=min(deadline, time.monotonic() + 2.0),
                )
                outcome["passed"] = not exceptions and isinstance(result, dict) and result.get("ok") is True
                if not outcome["passed"]:
                    outcome["error"] = str(result.get("error") if isinstance(result, dict) else "visible outcome missing")
            elif assertion.kind == "request":
                matches = [
                    item
                    for item in requests
                    if isinstance(item, dict)
                    and item.get("method") == assertion.method
                    and _request_name(item) == assertion.name
                ]
                outcome.update({"count": len(matches), "passed": len(matches) == 1})
                if len(matches) == 1 and assertion.method in {"command", "dispatch", "requestAction"}:
                    params = dict(matches[0].get("params") or {})
                    payload = params.get("payload") if assertion.method in {"command", "dispatch"} else dict(params.get("action") or {}).get("payload")
                    try:
                        validate_intent_payload(manifest.intents[str(assertion.name)].schema_, payload)
                    except ValueError as exc:
                        outcome.update({"passed": False, "error": str(exc)})
            elif assertion.kind == "state":
                surface = dict(dict(current_context.get("data") or {}).get("surface") or {})
                records = surface.get(assertion.namespace or "")
                records = records if isinstance(records, list) else []
                record = next(
                    (item for item in records if isinstance(item, dict) and item.get("key") == assertion.key),
                    None,
                )
                actual = None
                found = record is not None
                if record is not None:
                    actual = dict(record.get("data") or {}).get(assertion.field) if assertion.field else record.get("data")
                outcome.update({"found": found, "actual": actual, "passed": found and actual == assertion.equals})
            else:
                expression = (
                    "(() => {const expected="
                    + json.dumps(assertion.status)
                    + ";const visible=element=>{const style=getComputedStyle(element);const rect=element.getBoundingClientRect();return style.visibility!=='hidden'&&style.display!=='none'&&rect.width>0&&rect.height>0;};return [...document.querySelectorAll('[data-aloy-approval-state]')].filter(visible).some(element=>element.getAttribute('data-aloy-approval-state')===expected);})()"
                )
                result_id = send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
                result, exceptions = _receive_evaluation(
                    socket,
                    result_id=result_id,
                    deadline=min(deadline, time.monotonic() + 2.0),
                )
                outcome["passed"] = not exceptions and result is True
            observations.append(outcome)
            if not outcome["passed"]:
                diagnostics.append(
                    _diagnostic(
                        "primary_job_assertion_failed",
                        f"Primary job {job.description!r} did not satisfy its {assertion.kind} assertion: {outcome}",
                    )
                )

    evidence: dict[str, Any] = {
        "id": job.id,
        "description": job.description,
        "passed": not diagnostics,
        "steps": len(job.steps),
        "assertions": observations,
        "duration_ms": round((time.monotonic() - started_at) * 1000, 3),
    }
    fingerprint_value = {key: value for key, value in evidence.items() if key != "duration_ms"}
    evidence["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return diagnostics, evidence


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
