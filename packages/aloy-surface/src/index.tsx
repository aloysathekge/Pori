import {
  cloneElement,
  isValidElement,
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import type {
  ButtonHTMLAttributes,
  CSSProperties,
  HTMLAttributes,
  ReactElement,
  ReactNode,
} from 'react';

/** Stable visual values for generated Surfaces. */
export const surfaceTokens = {
  color: {
    ink: '#172022',
    muted: '#5f6b6d',
    line: '#d7dfdd',
    surface: '#ffffff',
    canvas: '#f6f9f8',
    accent: '#0f8571',
    accentContrast: '#ffffff',
    danger: '#b42318',
    dangerSurface: '#fff1f0',
    warning: '#9a6700',
    warningSurface: '#fff8e6',
    success: '#147a52',
    successSurface: '#edf9f2',
  },
  space: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '0.75rem',
    lg: '1rem',
    xl: '1.5rem',
    '2xl': '2rem',
  },
  radius: {
    sm: '0.5rem',
    md: '0.75rem',
    lg: '1rem',
    pill: '999px',
  },
  type: {
    body: 'Inter, ui-sans-serif, system-ui, sans-serif',
    display: 'Inter, ui-sans-serif, system-ui, sans-serif',
    mono: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  breakpoint: {
    mobile: '30rem',
    tablet: '48rem',
    desktop: '64rem',
  },
} as const;

export type SurfaceTokens = typeof surfaceTokens;

type SurfaceStyleProps = {
  className?: string;
  style?: CSSProperties;
};

function mergeClassNames(...values: Array<string | undefined>): string | undefined {
  const value = values.filter(Boolean).join(' ').trim();
  return value || undefined;
}

/** The semantic root for a generated Surface application. */
export function SurfaceRoot({
  children,
  className,
  style,
  ...props
}: HTMLAttributes<HTMLElement> & SurfaceStyleProps): ReactElement {
  return (
    <main
      {...props}
      data-aloy-surface-root="true"
      className={mergeClassNames('aloy-surface-root', className)}
      style={{
        boxSizing: 'border-box',
        width: '100%',
        minHeight: '100%',
        margin: 0,
        padding: surfaceTokens.space.xl,
        overflowX: 'hidden',
        color: surfaceTokens.color.ink,
        background: surfaceTokens.color.canvas,
        fontFamily: surfaceTokens.type.body,
        lineHeight: 1.5,
        ...style,
      }}
    >
      {children}
    </main>
  );
}

export interface SurfaceStackProps extends HTMLAttributes<HTMLDivElement>, SurfaceStyleProps {
  children?: ReactNode;
  direction?: 'row' | 'column';
  gap?: keyof SurfaceTokens['space'];
  align?: CSSProperties['alignItems'];
  justify?: CSSProperties['justifyContent'];
  wrap?: boolean;
}

/** A small flex primitive with safe wrapping for narrow Surface panes. */
export function SurfaceStack({
  children,
  direction = 'column',
  gap = 'lg',
  align,
  justify,
  wrap = false,
  className,
  style,
  ...props
}: SurfaceStackProps): ReactElement {
  return (
    <div
      {...props}
      className={mergeClassNames('aloy-surface-stack', className)}
      style={{
        display: 'flex',
        flexDirection: direction,
        gap: surfaceTokens.space[gap],
        alignItems: align,
        justifyContent: justify,
        flexWrap: wrap ? 'wrap' : undefined,
        minWidth: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export interface SurfaceGridProps extends HTMLAttributes<HTMLDivElement>, SurfaceStyleProps {
  children?: ReactNode;
  gap?: keyof SurfaceTokens['space'];
  minColumnWidth?: string;
}

/** A responsive grid that collapses naturally instead of creating overflow. */
export function SurfaceGrid({
  children,
  gap = 'lg',
  minColumnWidth = '16rem',
  className,
  style,
  ...props
}: SurfaceGridProps): ReactElement {
  return (
    <div
      {...props}
      className={mergeClassNames('aloy-surface-grid', className)}
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${minColumnWidth}), 1fr))`,
        gap: surfaceTokens.space[gap],
        minWidth: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export interface SurfacePanelProps extends Omit<HTMLAttributes<HTMLElement>, 'title'>, SurfaceStyleProps {
  children?: ReactNode;
  title?: ReactNode;
  description?: ReactNode;
}

/** A labelled section that gives generated information hierarchy a stable baseline. */
export function SurfacePanel({
  children,
  title,
  description,
  className,
  style,
  ...props
}: SurfacePanelProps): ReactElement {
  const titleId = useId();
  return (
    <section
      {...props}
      aria-labelledby={title ? titleId : props['aria-labelledby']}
      className={mergeClassNames('aloy-surface-panel', className)}
      style={{
        boxSizing: 'border-box',
        minWidth: 0,
        padding: surfaceTokens.space.xl,
        border: `1px solid ${surfaceTokens.color.line}`,
        borderRadius: surfaceTokens.radius.lg,
        background: surfaceTokens.color.surface,
        boxShadow: '0 1px 2px rgb(23 32 34 / 6%)',
        ...style,
      }}
    >
      {title && (
        <div style={{ marginBottom: description ? surfaceTokens.space.sm : surfaceTokens.space.lg }}>
          <h2 id={titleId} style={{ margin: 0, color: surfaceTokens.color.ink, fontSize: '1.05rem', lineHeight: 1.25 }}>
            {title}
          </h2>
          {description && (
            <p style={{ margin: `${surfaceTokens.space.xs} 0 0`, color: surfaceTokens.color.muted, fontSize: '0.9rem' }}>
              {description}
            </p>
          )}
        </div>
      )}
      {children}
    </section>
  );
}

export interface SurfaceFieldProps extends HTMLAttributes<HTMLDivElement>, SurfaceStyleProps {
  label: ReactNode;
  children: ReactElement;
  hint?: ReactNode;
  error?: ReactNode;
  required?: boolean;
}

/** Label, hint, and error wiring for one form control. */
export function SurfaceField({
  label,
  children,
  hint,
  error,
  required = false,
  className,
  style,
  ...props
}: SurfaceFieldProps): ReactElement {
  const fieldId = `aloy-field-${useId().replaceAll(':', '')}`;
  const childProps = children.props as {
    id?: string;
    'aria-describedby'?: string;
    'aria-invalid'?: boolean | 'false' | 'true';
    'aria-required'?: boolean | 'false' | 'true';
  };
  const hintId = hint ? `${fieldId}-hint` : undefined;
  const errorId = error ? `${fieldId}-error` : undefined;
  const describedBy = [childProps['aria-describedby'], hintId, errorId]
    .filter(Boolean)
    .join(' ') || undefined;
  const controlId = childProps.id ?? fieldId;
  const control = isValidElement(children)
    ? cloneElement(children as ReactElement<Record<string, unknown>>, {
        id: controlId,
        'aria-describedby': describedBy,
        'aria-invalid': error ? true : childProps['aria-invalid'],
        'aria-required': required ? true : childProps['aria-required'],
      })
    : children;
  return (
    <div
      {...props}
      data-aloy-field="true"
      className={mergeClassNames('aloy-surface-field', className)}
      style={{ display: 'grid', gap: surfaceTokens.space.xs, minWidth: 0, ...style }}
    >
      <label htmlFor={controlId} style={{ color: surfaceTokens.color.ink, fontSize: '0.9rem', fontWeight: 600 }}>
        {label}{required && <span aria-hidden="true"> *</span>}
      </label>
      {control}
      {hint && <div id={hintId} style={{ color: surfaceTokens.color.muted, fontSize: '0.8rem' }}>{hint}</div>}
      {error && <div id={errorId} role="alert" style={{ color: surfaceTokens.color.danger, fontSize: '0.8rem' }}>{error}</div>}
    </div>
  );
}

export type SurfaceStatusKind = 'loading' | 'ready' | 'empty' | 'stale' | 'error' | 'permission_denied' | 'pending' | 'indeterminate';

export interface SurfaceStatusProps extends HTMLAttributes<HTMLDivElement>, SurfaceStyleProps {
  status: SurfaceStatusKind;
  message?: ReactNode;
  retry?: () => void;
  retryLabel?: string;
}

/** A visible, live region for every host resource state. */
export function SurfaceStatus({
  status,
  message,
  retry,
  retryLabel = 'Try again',
  className,
  style,
  ...props
}: SurfaceStatusProps): ReactElement | null {
  if (status === 'ready' && !message) return null;
  const defaults: Record<SurfaceStatusKind, string> = {
    loading: 'Loading...',
    ready: 'Ready',
    empty: 'Nothing here yet.',
    stale: 'This view may be out of date.',
    error: 'This view could not be loaded.',
    permission_denied: 'You do not have access to this view.',
    pending: 'Aloy is working on this.',
    indeterminate: 'Aloy is checking the result.',
  };
  const isError = status === 'error' || status === 'permission_denied';
  const tone = isError
    ? { background: surfaceTokens.color.dangerSurface, color: surfaceTokens.color.danger }
    : status === 'pending' || status === 'stale' || status === 'indeterminate'
      ? { background: surfaceTokens.color.warningSurface, color: surfaceTokens.color.warning }
      : { background: surfaceTokens.color.successSurface, color: surfaceTokens.color.success };
  return (
    <div
      {...props}
      className={mergeClassNames('aloy-surface-status', className)}
      role={isError ? 'alert' : 'status'}
      aria-live={isError ? 'assertive' : 'polite'}
      aria-atomic="true"
      data-aloy-resource-state={status}
      style={{ display: 'flex', alignItems: 'center', gap: surfaceTokens.space.md, padding: `${surfaceTokens.space.sm} ${surfaceTokens.space.md}`, borderRadius: surfaceTokens.radius.md, ...tone, ...style }}
    >
      <span style={{ minWidth: 0, flex: 1 }}>{message ?? defaults[status]}</span>
      {retry && (status === 'error' || status === 'stale' || status === 'indeterminate') && (
        <ActionButton type="button" variant="outline" onClick={retry}>{retryLabel}</ActionButton>
      )}
    </div>
  );
}

export type SurfaceActionButtonVariant = 'primary' | 'secondary' | 'danger' | 'outline' | 'quiet';

export interface SurfaceActionButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'>, SurfaceStyleProps {
  children: ReactNode;
  busy?: boolean;
  busyLabel?: string;
  variant?: SurfaceActionButtonVariant;
}

/** A 44px, keyboard-visible action control suitable for primary-job checks. */
export function ActionButton({
  children,
  busy = false,
  busyLabel = 'Working...',
  variant = 'primary',
  className,
  style,
  disabled,
  onFocus,
  onBlur,
  ...props
}: SurfaceActionButtonProps): ReactElement {
  const [focused, setFocused] = useState(false);
  const palette: Record<SurfaceActionButtonVariant, CSSProperties> = {
    primary: { background: surfaceTokens.color.accent, color: surfaceTokens.color.accentContrast, borderColor: surfaceTokens.color.accent },
    secondary: { background: surfaceTokens.color.surface, color: surfaceTokens.color.ink, borderColor: surfaceTokens.color.line },
    danger: { background: surfaceTokens.color.danger, color: surfaceTokens.color.accentContrast, borderColor: surfaceTokens.color.danger },
    outline: { background: 'transparent', color: surfaceTokens.color.accent, borderColor: surfaceTokens.color.accent },
    quiet: { background: 'transparent', color: surfaceTokens.color.ink, borderColor: 'transparent' },
  };
  return (
    <button
      {...props}
      type={props.type ?? 'button'}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      data-aloy-action="true"
      className={mergeClassNames('aloy-surface-action', className)}
      onFocus={(event) => { setFocused(true); onFocus?.(event); }}
      onBlur={(event) => { setFocused(false); onBlur?.(event); }}
      style={{
        boxSizing: 'border-box',
        minHeight: '2.75rem',
        minWidth: '2.75rem',
        padding: `${surfaceTokens.space.sm} ${surfaceTokens.space.md}`,
        border: '1px solid',
        borderRadius: surfaceTokens.radius.md,
        font: 'inherit',
        fontWeight: 600,
        lineHeight: 1.2,
        cursor: disabled || busy ? 'wait' : 'pointer',
        transition: 'filter 120ms ease, box-shadow 120ms ease',
        ...palette[variant],
        ...(focused ? { outline: `2px solid ${surfaceTokens.color.accent}`, outlineOffset: '2px' } : {}),
        ...style,
      }}
    >
      {busy ? busyLabel : children}
    </button>
  );
}

const PROTOCOL = '1' as const;
const REQUEST_TIMEOUT_MS = 20_000;
const MAX_ATTEMPTS = 2;

export interface SurfaceDataRecord<T = Record<string, unknown>> {
  id: string;
  namespace: string;
  key: string;
  data: T;
  revision: number;
  posture:
    | 'official'
    | 'user_reported'
    | 'inferred'
    | 'estimated'
    | 'pending'
    | 'committed'
    | 'failed'
    | 'indeterminate';
  provenance: Record<string, unknown>;
  evidence_refs: Array<Record<string, unknown>>;
}

export interface EventRecord<T = Record<string, unknown>> {
  id: string;
  namespace: string;
  key: string;
  title: string;
  summary: string;
  data: T;
  posture: 'observed' | 'inferred' | 'unverified';
  confidence: number;
  revision: number;
  evidence_refs: Array<{
    evidence_id: string;
    url: string;
    title: string;
    retrieved_at: string;
  }>;
  created_at: string;
  updated_at: string;
}

export type SurfaceInteractionStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'waiting_approval'
  | 'approved'
  | 'executing'
  | 'committed'
  | 'completed'
  | 'rejected'
  | 'failed'
  | 'cancelled'
  | 'indeterminate';

export interface SurfaceInteraction {
  id: string;
  event_id: string;
  build_id: string;
  code_revision_id: string;
  name: string;
  interaction_class: string;
  component_id: string;
  status: SurfaceInteractionStatus;
  handling_run_id: string | null;
  proposal_id: string | null;
  request_message_id: string | null;
  outcome_message_id: string | null;
  result: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface SurfaceProposal {
  id: string;
  event_id: string;
  tool: string;
  args: Record<string, unknown>;
  reason: string;
  impact: string;
  risk: string;
  routing: string;
  status: string;
  expires_at: string | null;
  decided_at: string | null;
  provider_operation_id: string | null;
  receipt: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface SurfaceReceipt {
  proposal_id: string;
  tool: string;
  receipt: Record<string, unknown>;
  status: string;
  updated_at: string;
}

export interface SurfaceTrailEntry {
  id: string;
  kind: string;
  summary: string;
  actor_id: string | null;
  run_id: string | null;
  proposal_id: string | null;
  task_id: string | null;
  evidence_refs: Array<Record<string, unknown>>;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface SurfaceCommandAttempt {
  id: string;
  event_id: string;
  build_id: string;
  code_revision_id: string;
  interaction_id: string | null;
  method: string;
  name: string;
  interaction_class: string;
  component_id: string;
  base_data_revision: number;
  observed_data_revision: number;
  status: string;
  error_code: string | null;
  error: string | null;
  http_status: number;
  retryable: boolean;
  created_at: string;
}

export interface SurfaceContext {
  protocol_version: '1';
  command_contract_version: '1';
  sdk_version: '1';
  resource_state_version?: '1';
  event_id: string;
  project_id: string;
  build_id: string;
  code_revision_id: string;
  data_revision: number;
  capabilities: string[];
  widgets: string[];
  resource_states?: Record<string, SurfaceResourceSnapshot>;
  data: {
    event?: Record<string, unknown>;
    tasks?: Array<Record<string, unknown>>;
    files?: Array<Record<string, unknown>>;
    proposals?: SurfaceProposal[];
    receipts?: SurfaceReceipt[];
    trail?: SurfaceTrailEntry[];
    interactions: SurfaceInteraction[];
    command_attempts?: SurfaceCommandAttempt[];
    surface?: Record<string, Array<SurfaceDataRecord>>;
    records?: Record<string, Array<EventRecord>>;
  };
}

export type SurfaceResourceStatus =
  | 'loading'
  | 'ready'
  | 'empty'
  | 'stale'
  | 'error'
  | 'permission_denied'
  | 'pending'
  | 'indeterminate';

export interface SurfaceResourceSnapshot {
  status: SurfaceResourceStatus;
  message?: string;
  retryable: boolean;
}

export interface SurfaceResourceState extends SurfaceResourceSnapshot {
  resource: string;
  feedbackProps: {
    'data-aloy-resource': string;
    'data-aloy-resource-state': SurfaceResourceStatus;
    'aria-busy': boolean;
  };
}

export type SurfaceApprovalStatus = 'loading' | 'clear' | 'required';

export interface SurfaceApprovalState {
  status: SurfaceApprovalStatus;
  required: boolean;
  proposals: SurfaceProposal[];
  interactions: SurfaceInteraction[];
  feedbackProps: {
    'data-aloy-approval-state': SurfaceApprovalStatus;
    role: 'status';
    'aria-live': 'polite';
    'aria-atomic': true;
  };
}

export interface SurfaceAction {
  name: string;
  payload: Record<string, unknown>;
  reason?: string;
}

export interface SurfaceCommandOptions {
  componentId?: string;
  idempotencyKey?: string;
}

export type SurfaceRequestErrorCode =
  | 'conflict'
  | 'permission_denied'
  | 'invalid'
  | 'rate_limited'
  | 'unavailable'
  | 'timeout'
  | 'disconnected'
  | 'failed';

export type SurfaceCommandStatus =
  | 'idle'
  | 'pending'
  | 'committed'
  | 'accepted'
  | 'conflict'
  | 'failed';

export interface SurfaceCommandReceipt {
  id: string;
  status: string;
  name: string;
  interaction_class: string;
  data_revision: number | null;
  handling_run_id: string | null;
  proposal_id: string | null;
  result: Record<string, unknown>;
  replayed: boolean;
}

export interface SurfaceCommandFeedbackProps {
  role: 'status' | 'alert';
  'aria-live': 'polite' | 'assertive';
  'aria-atomic': true;
  'data-aloy-command-name': string;
  'data-aloy-command-status': SurfaceCommandStatus;
  'data-aloy-interaction-status': SurfaceInteractionStatus | '';
}

export interface SurfaceCommandController<
  TInput extends Record<string, unknown>,
  TResult,
> {
  status: SurfaceCommandStatus;
  pending: boolean;
  result: TResult | null;
  error: SurfaceRequestError | null;
  /** Durable host lifecycle after the command request is accepted. */
  interaction: SurfaceInteraction | null;
  lifecycleStatus: SurfaceInteractionStatus | null;
  execute: (input: TInput) => Promise<TResult>;
  retry: () => Promise<TResult>;
  reset: () => void;
  feedbackProps: SurfaceCommandFeedbackProps;
}

export type SurfaceRuntimeStatus = 'disconnected' | 'healthy' | 'degraded';

export interface SurfaceRuntimeState {
  status: SurfaceRuntimeStatus;
  message?: string;
}

interface BridgeResponse {
  protocol: '1';
  type: 'response';
  sessionId: string;
  requestId: string;
  ok: boolean;
  result?: unknown;
  error?: string;
  errorCode?: SurfaceRequestErrorCode;
  serverCode?: string;
  attemptId?: string;
  statusCode?: number;
  retryable?: boolean;
}

interface PendingRequest {
  method: string;
  params: Record<string, unknown>;
  attempts: number;
  requestId: string;
  timeout: ReturnType<typeof setTimeout> | null;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

export class SurfaceRequestError extends Error {
  readonly method: string;
  readonly code: SurfaceRequestErrorCode;
  readonly retryable: boolean;
  readonly statusCode?: number;
  readonly idempotencyKey?: string;
  readonly serverCode?: string;
  readonly attemptId?: string;

  constructor(
    message: string,
    options: {
      method: string;
      code?: SurfaceRequestErrorCode;
      retryable?: boolean;
      statusCode?: number;
      idempotencyKey?: string;
      serverCode?: string;
      attemptId?: string;
    },
  ) {
    super(message);
    this.name = 'SurfaceRequestError';
    this.method = options.method;
    this.code = options.code ?? 'failed';
    this.retryable = options.retryable ?? false;
    this.statusCode = options.statusCode;
    this.idempotencyKey = options.idempotencyKey;
    this.serverCode = options.serverCode;
    this.attemptId = options.attemptId;
  }
}

export class SurfaceRequestTimeoutError extends SurfaceRequestError {
  readonly method: string;
  readonly idempotencyKey?: string;

  constructor(method: string, idempotencyKey?: string) {
    super(`Aloy Surface ${method} request timed out`, {
      method,
      code: 'timeout',
      retryable: true,
      idempotencyKey,
    });
    this.name = 'SurfaceRequestTimeoutError';
    this.method = method;
    this.idempotencyKey = idempotencyKey;
  }
}

let port: MessagePort | null = null;
let sessionId: string | null = null;
let context: SurfaceContext | null = null;
const disconnectedRuntime: SurfaceRuntimeState = { status: 'disconnected' };
let runtime: SurfaceRuntimeState = disconnectedRuntime;
const contextSubscribers = new Set<() => void>();
const runtimeSubscribers = new Set<() => void>();
const pending = new Map<string, PendingRequest>();

function publishContext(next: SurfaceContext) {
  context = next;
  for (const subscriber of contextSubscribers) subscriber();
}

function publishRuntime(next: SurfaceRuntimeState) {
  runtime = next;
  for (const subscriber of runtimeSubscribers) subscriber();
}

function clearPendingTimeout(request: PendingRequest) {
  if (request.timeout) clearTimeout(request.timeout);
  request.timeout = null;
}

function rejectPending(request: PendingRequest, error: Error) {
  clearPendingTimeout(request);
  pending.delete(request.requestId);
  request.reject(error);
}

function send(request: PendingRequest) {
  if (!port || !sessionId) {
    rejectPending(
      request,
      new SurfaceRequestError('Aloy Surface bridge is not ready', {
        method: request.method,
        code: 'disconnected',
        retryable: true,
        idempotencyKey:
          typeof request.params.idempotencyKey === 'string'
            ? request.params.idempotencyKey
            : undefined,
      }),
    );
    return;
  }
  clearPendingTimeout(request);
  pending.delete(request.requestId);
  request.requestId = crypto.randomUUID();
  request.attempts += 1;
  pending.set(request.requestId, request);
  request.timeout = setTimeout(() => {
    if (!pending.has(request.requestId)) return;
    if (request.attempts < MAX_ATTEMPTS && port && sessionId) {
      send(request);
      return;
    }
    rejectPending(
      request,
      new SurfaceRequestTimeoutError(
        request.method,
        typeof request.params.idempotencyKey === 'string'
          ? request.params.idempotencyKey
          : undefined,
      ),
    );
  }, REQUEST_TIMEOUT_MS);
  try {
    port.postMessage({
      protocol: PROTOCOL,
      type: 'request',
      sessionId,
      requestId: request.requestId,
      method: request.method,
      params: request.params,
    });
  } catch {
    port.close();
    port = null;
    sessionId = null;
    publishRuntime({
      status: 'degraded',
      message: 'The Aloy Surface bridge disconnected',
    });
    // Keep the bounded request pending so an imminent host reconnect can
    // replay it with the same idempotency key.
  }
}

function replayPending() {
  for (const request of [...pending.values()]) {
    if (request.attempts < MAX_ATTEMPTS) send(request);
    else {
      rejectPending(
        request,
        new SurfaceRequestError(
          'Aloy Surface bridge reconnected after request retry',
          {
            method: request.method,
            code: 'unavailable',
            retryable: true,
            idempotencyKey:
              typeof request.params.idempotencyKey === 'string'
                ? request.params.idempotencyKey
                : undefined,
          },
        ),
      );
    }
  }
}

function onPortMessage(event: MessageEvent<unknown>) {
  const value = event.data as {
    protocol?: string;
    type?: string;
    sessionId?: string;
    nonce?: string;
    context?: SurfaceContext;
    status?: string;
    message?: string;
  } | null;
  if (
    !value
    || value.protocol !== PROTOCOL
    || value.sessionId !== sessionId
    || !value.type
  ) return;
  if (value.type === 'ping' && typeof value.nonce === 'string') {
    port?.postMessage({
      protocol: PROTOCOL,
      type: 'pong',
      sessionId,
      nonce: value.nonce,
    });
    return;
  }
  if (value.type === 'context' && value.context) {
    publishContext(value.context);
    return;
  }
  if (value.type === 'runtime' && value.status === 'degraded') {
    publishRuntime({ status: 'degraded', message: value.message });
    port?.close();
    port = null;
    sessionId = null;
    return;
  }
  if (value.type !== 'response' || !('requestId' in value)) return;
  const response = value as BridgeResponse;
  if (typeof response.requestId !== 'string') return;
  const request = pending.get(response.requestId);
  if (!request) return;
  clearPendingTimeout(request);
  pending.delete(response.requestId);
  if (response.ok) {
    request.resolve(response.result);
    return;
  }
  if (response.retryable && request.attempts < MAX_ATTEMPTS && port && sessionId) {
    send(request);
    return;
  }
  request.reject(
    new SurfaceRequestError(
      response.error || 'Aloy Surface request failed',
      {
        method: request.method,
        code: response.errorCode,
        retryable: response.retryable,
        statusCode: response.statusCode,
        serverCode: response.serverCode,
        attemptId: response.attemptId,
        idempotencyKey:
          typeof request.params.idempotencyKey === 'string'
            ? request.params.idempotencyKey
            : undefined,
      },
    ),
  );
}

if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
  window.addEventListener('message', (event: MessageEvent<unknown>) => {
  const value = event.data as {
    protocol?: string;
    type?: string;
    sessionId?: string;
    context?: SurfaceContext;
  } | null;
  if (
    !event.isTrusted
    || event.source !== window.parent
    || value?.protocol !== PROTOCOL
    || value.type !== 'aloy.surface.connect'
    || typeof value.sessionId !== 'string'
    || !value.sessionId
    || !value.context
    || event.ports.length !== 1
  ) return;

  port?.close();
  for (const request of pending.values()) clearPendingTimeout(request);
  sessionId = value.sessionId;
  port = event.ports[0];
  port.onmessage = onPortMessage;
  port.start();
  publishContext(value.context);
  publishRuntime({ status: 'healthy' });
  port.postMessage({ protocol: PROTOCOL, type: 'ready', sessionId });
  replayPending();
  });
}

function request<T>(method: string, params: Record<string, unknown>): Promise<T> {
  if (!port || !sessionId) {
    return Promise.reject(
      new SurfaceRequestError('Aloy Surface bridge is not ready', {
        method,
        code: 'disconnected',
        retryable: true,
        idempotencyKey:
          typeof params.idempotencyKey === 'string'
            ? params.idempotencyKey
            : undefined,
      }),
    );
  }
  return new Promise<T>((resolve, reject) => {
    const pendingRequest: PendingRequest = {
      method,
      params,
      attempts: 0,
      requestId: '',
      timeout: null,
      resolve: (value) => resolve(value as T),
      reject,
    };
    send(pendingRequest);
  });
}

function componentId(value?: string) {
  return value?.trim().slice(0, 200) || 'surface';
}

function idempotencyKey(value?: string) {
  return value?.trim().slice(0, 200) || crypto.randomUUID();
}

export function subscribeSurface(listener: () => void) {
  contextSubscribers.add(listener);
  return () => contextSubscribers.delete(listener);
}

export function subscribeSurfaceRuntime(listener: () => void) {
  runtimeSubscribers.add(listener);
  return () => runtimeSubscribers.delete(listener);
}

export function getSurfaceContext(): SurfaceContext | null {
  return context;
}

export function getSurfaceRuntime(): SurfaceRuntimeState {
  return runtime;
}

export function useSurfaceContext(): SurfaceContext | null {
  return useSyncExternalStore(subscribeSurface, getSurfaceContext, () => null);
}

export function useSurfaceRuntime(): SurfaceRuntimeState {
  return useSyncExternalStore(
    subscribeSurfaceRuntime,
    getSurfaceRuntime,
    () => disconnectedRuntime,
  );
}

/** Read the host-owned lifecycle state for one capability-scoped resource. */
export function useSurfaceResourceState(resource: string): SurfaceResourceState {
  const current = useSurfaceContext();
  if (!current) {
    return {
      resource,
      status: 'loading',
      message: 'Aloy is loading this Event data.',
      retryable: true,
      feedbackProps: {
        'data-aloy-resource': resource,
        'data-aloy-resource-state': 'loading',
        'aria-busy': true,
      },
    };
  }
  const declared = current.resource_states?.[resource];
  const value = resource.startsWith('data:')
    ? current.data.surface?.[resource.slice(5)]
    : resource.startsWith('records:')
      ? current.data.records?.[resource.slice(8)]
      : current.data[resource as keyof typeof current.data];
  const empty = value == null
    || (Array.isArray(value) && value.length === 0)
    || (typeof value === 'object' && !Array.isArray(value)
      && Object.keys(value as Record<string, unknown>).length === 0);
  const snapshot = declared ?? {
    status: empty ? 'empty' : 'ready',
    message: empty ? 'No Event data exists here yet.' : 'This Event data is current.',
    retryable: false,
  } satisfies SurfaceResourceSnapshot;
  return {
    resource,
    ...snapshot,
    feedbackProps: {
      'data-aloy-resource': resource,
      'data-aloy-resource-state': snapshot.status,
      'aria-busy': snapshot.status === 'loading' || snapshot.status === 'pending',
    },
  };
}

export function useEvent<T = Record<string, unknown>>(): T | null {
  return (useSurfaceContext()?.data.event as T | undefined) ?? null;
}

export function useSurfaceData<T = Record<string, unknown>>(
  namespace: string,
): Array<SurfaceDataRecord<T>> {
  const records = useSurfaceContext()?.data.surface?.[namespace] ?? [];
  return records as Array<SurfaceDataRecord<T>>;
}

/** Read host-owned, evidence-backed Event records. Generated code cannot mutate them. */
export function useEventRecords<T = Record<string, unknown>>(
  namespace: string,
): Array<EventRecord<T>> {
  const records = useSurfaceContext()?.data.records?.[namespace] ?? [];
  return records as Array<EventRecord<T>>;
}

export function useTasks(): Array<Record<string, unknown>> {
  return useSurfaceContext()?.data.tasks ?? [];
}

/** Read Proposal truth. Approval controls themselves remain host-owned. */
export function useProposals(): SurfaceProposal[] {
  return useSurfaceContext()?.data.proposals ?? [];
}

export function usePendingApprovals(): SurfaceProposal[] {
  const proposals = useProposals();
  return useMemo(
    () => proposals.filter((proposal) => proposal.status === 'pending'),
    [proposals],
  );
}

/**
 * Describe whether the Surface has a protected action waiting on the user.
 * Approval controls remain host-owned; generated UI may only summarize and
 * link to that trusted region.
 */
export function useSurfaceApprovalState(): SurfaceApprovalState {
  const current = useSurfaceContext();
  const proposals = usePendingApprovals();
  const interactions = useMemo(
    () => (current?.data.interactions ?? []).filter(
      (interaction) => interaction.status === 'waiting_approval',
    ),
    [current],
  );
  const status: SurfaceApprovalStatus = !current
    ? 'loading'
    : proposals.length > 0 || interactions.length > 0
      ? 'required'
      : 'clear';
  return {
    status,
    required: status === 'required',
    proposals,
    interactions,
    feedbackProps: {
      'data-aloy-approval-state': status,
      role: 'status',
      'aria-live': 'polite',
      'aria-atomic': true,
    },
  };
}

/** Read receipt-backed external outcomes; absence of a receipt is not success. */
export function useReceipts(): SurfaceReceipt[] {
  return useSurfaceContext()?.data.receipts ?? [];
}

export function useTrail(): SurfaceTrailEntry[] {
  return useSurfaceContext()?.data.trail ?? [];
}

export function useInteractions(): SurfaceInteraction[] {
  return useSurfaceContext()?.data.interactions ?? [];
}

/** Follow one accepted command through Run, approval, execution, and outcome. */
export function useSurfaceInteraction(
  interactionId: string | null | undefined,
): SurfaceInteraction | null {
  const interactions = useInteractions();
  return useMemo(
    () => interactions.find((item) => item.id === interactionId) ?? null,
    [interactionId, interactions],
  );
}

/** Find the newest durable interaction emitted by one generated control. */
export function useLatestSurfaceInteraction(
  name: string,
  componentId?: string,
): SurfaceInteraction | null {
  const interactions = useInteractions();
  return useMemo(
    () => interactions.find(
      (item) => item.name === name
        && (!componentId || item.component_id === componentId),
    ) ?? null,
    [componentId, interactions, name],
  );
}

export function isSurfaceInteractionTerminal(
  status: SurfaceInteractionStatus | null | undefined,
): boolean {
  return status === 'committed'
    || status === 'completed'
    || status === 'rejected'
    || status === 'failed'
    || status === 'cancelled'
    || status === 'indeterminate';
}

export function isSurfaceInteractionActive(
  status: SurfaceInteractionStatus | null | undefined,
): boolean {
  return status === 'pending'
    || status === 'queued'
    || status === 'running'
    || status === 'waiting_approval'
    || status === 'approved'
    || status === 'executing';
}

export function useCommandAttempts(): SurfaceCommandAttempt[] {
  return useSurfaceContext()?.data.command_attempts ?? [];
}

export function dispatch<T = Record<string, unknown>>(
  name: string,
  payload: Record<string, unknown>,
  options: { componentId?: string; idempotencyKey?: string } = {},
): Promise<T> {
  return request<T>('dispatch', {
    name,
    payload,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}

/** Send a manifest-declared command to Aloy's host-owned command runtime. */
export function command<T = Record<string, unknown>>(
  name: string,
  input: Record<string, unknown>,
  options: SurfaceCommandOptions = {},
): Promise<T> {
  return request<T>('command', {
    name,
    payload: input,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}

function commandStatus(result: unknown): SurfaceCommandStatus {
  if (
    result
    && typeof result === 'object'
    && !Array.isArray(result)
    && (result as { status?: unknown }).status === 'committed'
  ) {
    return 'committed';
  }
  return 'accepted';
}

function requestError(cause: unknown, method: string): SurfaceRequestError {
  if (cause instanceof SurfaceRequestError) return cause;
  return new SurfaceRequestError(
    cause instanceof Error ? cause.message : 'Aloy Surface request failed',
    { method },
  );
}

function snapshotCommandInput<TInput extends Record<string, unknown>>(
  input: TInput,
): TInput {
  try {
    return JSON.parse(JSON.stringify(input)) as TInput;
  } catch {
    throw new SurfaceRequestError(
      'Surface command input must be JSON-serializable',
      { method: 'command', code: 'invalid' },
    );
  }
}

/**
 * Host-owned command lifecycle for generated React controls.
 *
 * A new execute call receives a new idempotency key. Retry reuses the exact
 * input and key, while duplicate submits share the in-flight Promise. The
 * command is reported as committed only after the host has acknowledged its
 * durable state result and delivered refreshed canonical context.
 */
export function useSurfaceCommand<
  TInput extends Record<string, unknown> = Record<string, unknown>,
  TResult = SurfaceCommandReceipt,
>(
  name: string,
  options: Pick<SurfaceCommandOptions, 'componentId'> = {},
): SurfaceCommandController<TInput, TResult> {
  const [status, setStatus] = useState<SurfaceCommandStatus>('idle');
  const [result, setResult] = useState<TResult | null>(null);
  const [error, setError] = useState<SurfaceRequestError | null>(null);
  const lastRequest = useRef<{ input: TInput; idempotencyKey: string } | null>(
    null,
  );
  const inFlight = useRef<Promise<TResult> | null>(null);
  const interactions = useInteractions();
  const resultInteractionId = (
    result
    && typeof result === 'object'
    && !Array.isArray(result)
    && typeof (result as { id?: unknown }).id === 'string'
  ) ? (result as unknown as { id: string }).id : null;
  const interaction = useMemo(
    () => interactions.find((item) => item.id === resultInteractionId) ?? null,
    [interactions, resultInteractionId],
  );

  const run = useCallback(
    (input: TInput, requestId: string): Promise<TResult> => {
      if (inFlight.current) return inFlight.current;
      setStatus('pending');
      setError(null);
      const current = command<TResult>(name, input, {
        componentId: options.componentId,
        idempotencyKey: requestId,
      })
        .then((receipt) => {
          setResult(receipt);
          setStatus(commandStatus(receipt));
          return receipt;
        })
        .catch((cause: unknown) => {
          const nextError = requestError(cause, 'command');
          setError(nextError);
          setStatus(nextError.code === 'conflict' ? 'conflict' : 'failed');
          throw nextError;
        })
        .finally(() => {
          if (inFlight.current === current) inFlight.current = null;
        });
      inFlight.current = current;
      return current;
    },
    [name, options.componentId],
  );

  const execute = useCallback(
    (input: TInput) => {
      if (inFlight.current) return inFlight.current;
      let snapshot: TInput;
      try {
        snapshot = snapshotCommandInput(input);
      } catch (cause) {
        const nextError = requestError(cause, 'command');
        setError(nextError);
        setStatus('failed');
        return Promise.reject(nextError);
      }
      const requestId = idempotencyKey();
      lastRequest.current = { input: snapshot, idempotencyKey: requestId };
      return run(snapshot, requestId);
    },
    [run],
  );

  const retry = useCallback(() => {
    if (!lastRequest.current) {
      return Promise.reject(
        new SurfaceRequestError('No Surface command is available to retry', {
          method: 'command',
          code: 'failed',
        }),
      );
    }
    return run(lastRequest.current.input, lastRequest.current.idempotencyKey);
  }, [run]);

  const reset = useCallback(() => {
    if (inFlight.current) return;
    lastRequest.current = null;
    setStatus('idle');
    setResult(null);
    setError(null);
  }, []);

  const feedbackProps = useMemo<SurfaceCommandFeedbackProps>(
    () => ({
      role: status === 'failed' || status === 'conflict' ? 'alert' : 'status',
      'aria-live':
        status === 'failed' || status === 'conflict' ? 'assertive' : 'polite',
      'aria-atomic': true,
      'data-aloy-command-name': name,
      'data-aloy-command-status': status,
      'data-aloy-interaction-status': interaction?.status ?? '',
    }),
    [interaction?.status, name, status],
  );

  return {
    status,
    pending: status === 'pending',
    result,
    error,
    interaction,
    lifecycleStatus: interaction?.status ?? null,
    execute,
    retry,
    reset,
    feedbackProps,
  };
}

export function askAloy<T = Record<string, unknown>>(
  message: string,
  surfaceContext: Record<string, unknown> = {},
  options: { componentId?: string; idempotencyKey?: string } = {},
): Promise<T> {
  return request<T>('askAloy', {
    message,
    context: surfaceContext,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}

export function requestAction<T = Record<string, unknown>>(
  action: SurfaceAction,
  options: { componentId?: string; idempotencyKey?: string } = {},
): Promise<T> {
  return request<T>('requestAction', {
    action,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}
