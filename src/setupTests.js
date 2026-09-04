import React from 'react';
import '@testing-library/jest-dom';
import { vi } from 'vitest';

globalThis.jest = vi;
vi.mock('canvas', () => ({ default: {} }));
vi.mock('react-konva', () => ({
    Stage: ({ children, ...props }) => React.createElement('div', { 'data-testid': 'mock-stage', ...props }, children),
    Layer: ({ children, ...props }) => React.createElement('div', { 'data-testid': 'mock-layer', ...props }, children),
    Line: (props) => React.createElement('div', { 'data-testid': 'mock-line', ...props }),
    Circle: (props) => React.createElement('div', { 'data-testid': 'mock-circle', ...props }),
    Rect: (props) => React.createElement('div', { 'data-testid': 'mock-rect', ...props }),
    Group: ({ children, ...props }) => React.createElement('div', { 'data-testid': 'mock-group', ...props }, children),
    Image: (props) => React.createElement('div', { 'data-testid': 'mock-image', ...props }),
}));
