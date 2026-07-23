import { useState } from 'react';
import { ActionButton, askAloy } from '@aloy/surface';
import { Section } from '../primitives';

type AskStatus = 'idle' | 'sending' | 'sent' | 'error';

export function AskAloy() {
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState<AskStatus>('idle');
  return (
    <Section heading="Ask Aloy">
      <form
        className="baseline-ask"
        onSubmit={async (event) => {
          event.preventDefault();
          const trimmed = message.trim();
          if (!trimmed || status === 'sending') return;
          setStatus('sending');
          try {
            await askAloy(trimmed, {}, { componentId: 'ask-aloy' });
            setStatus('sent');
            setMessage('');
          } catch {
            setStatus('error');
          }
        }}
      >
        <label>
          <span className="baseline-label">Ask Aloy</span>
          <input
            aria-label="Ask Aloy"
            value={message}
            onChange={(event) => {
              setMessage(event.target.value);
              if (status !== 'idle') setStatus('idle');
            }}
            placeholder="Ask about this Event, or ask Aloy to change this Surface"
          />
        </label>
        <ActionButton type="submit" disabled={status === 'sending'}>
          Ask Aloy
        </ActionButton>
      </form>
      <p role="status" aria-live="polite">
        {status === 'sending'
          ? 'Sending to Aloy…'
          : status === 'sent'
            ? 'Sent. Aloy will answer in this Event’s conversation.'
            : status === 'error'
              ? 'That did not reach Aloy. Please try again.'
              : ''}
      </p>
    </Section>
  );
}
