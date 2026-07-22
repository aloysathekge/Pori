import { describe, expect, test } from 'bun:test';

import {
  collapseResolvedSurfaceRequests,
  surfaceConversationPresentation,
} from '../src/components/chat/surfaceConversationPresentation';
import type { MessageResponse } from '../src/types';

function message(overrides: Partial<MessageResponse>): MessageResponse {
  return {
    id: 'message-1',
    role: 'user',
    content: 'Carry out the career.research_requested reasoning command from this Event Surface.',
    metadata: null,
    created_at: '2026-07-22T00:00:00Z',
    ...overrides,
  };
}

describe('Surface conversation presentation', () => {
  test('never displays an internal command identifier as a user message', () => {
    const presentation = surfaceConversationPresentation(message({
      metadata: { kind: 'surface_interaction' },
    }));

    expect(presentation.card).toBe(true);
    expect(presentation.content).toBe('Asked Aloy to continue from the Surface.');
    expect(presentation.content).not.toContain('career.research_requested');
  });

  test('uses a reviewed request label when the host supplies one', () => {
    const presentation = surfaceConversationPresentation(message({
      metadata: {
        kind: 'surface_interaction',
        surface_request_label: 'Research matching roles',
      },
    }));

    expect(presentation.content).toBe('Research matching roles');
    expect(presentation.status).toBe('sent');
  });

  test('replaces worker diagnostics with actionable failure copy', () => {
    const presentation = surfaceConversationPresentation(message({
      role: 'assistant',
      content: 'The worker exhausted its safe retry attempts.',
      metadata: {
        kind: 'surface_reasoning_result',
        status: 'failed',
        surface_request_label: 'Research matching roles',
      },
    }));

    expect(presentation.content).toBe(
      "Aloy couldn't complete Research matching roles right now. Try again from the Surface.",
    );
    expect(presentation.content).not.toContain('worker');
    expect(presentation.status).toBe('failed');
  });

  test('replaces a generated request card with its durable outcome', () => {
    const request = message({
      id: 'request',
      metadata: {
        kind: 'surface_interaction',
        surface_interaction_id: 'interaction-1',
        surface_request_origin: 'surface_control',
      },
    });
    const outcome = message({
      id: 'outcome',
      role: 'assistant',
      content: 'Could not finish.',
      metadata: {
        kind: 'surface_reasoning_result',
        status: 'failed',
        surface_interaction_id: 'interaction-1',
      },
    });

    expect(collapseResolvedSurfaceRequests([request, outcome])).toEqual([outcome]);
    expect(collapseResolvedSurfaceRequests([request])).toEqual([request]);
  });

  test('keeps a direct question visible beside its answer', () => {
    const question = message({
      id: 'question',
      content: 'Compare these hotels.',
      metadata: {
        kind: 'surface_interaction',
        surface_interaction_id: 'interaction-2',
        surface_request_origin: 'user_question',
      },
    });
    const answer = message({
      id: 'answer',
      role: 'assistant',
      content: 'Hotel A is the better fit.',
      metadata: {
        kind: 'surface_reasoning_result',
        status: 'completed',
        surface_interaction_id: 'interaction-2',
      },
    });

    expect(collapseResolvedSurfaceRequests([question, answer])).toEqual([
      question,
      answer,
    ]);
  });
});
