import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ToolOptionsDrawer from './ToolOptionsDrawer';

vi.mock('./useAnnotationServices', () => ({
  default: () => ({
    services: [],
    policyLoading: false,
    policyError: 'routing service unavailable',
    showInstanceWarning: false,
    closeInstanceWarning: vi.fn(),
    confirmInstanceRun: vi.fn(),
  }),
}));

vi.mock('./useRailTools', () => ({
  default: () => ({
    railTool: 'point',
    promptAction: 'nothing',
    changePromptAction: vi.fn(),
  }),
}));

vi.mock('../../../stores/selectors/annotationSelectors', () => ({
  useToggleLeftDrawer: () => vi.fn(),
}));

vi.mock('./CrossImageSuggestionCard', () => ({ default: () => <div>Cross image card</div> }));
vi.mock('./ServiceCard', () => ({ default: () => <div>Service card</div> }));
vi.mock('../modals/InstanceWarningModal', () => ({ default: () => null }));

describe('ToolOptionsDrawer routing status', () => {
  it('shows policy-loading failures instead of silently using stale routing', () => {
    render(<ToolOptionsDrawer />);

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Unable to load model routing policy: routing service unavailable'
    );
  });
});
