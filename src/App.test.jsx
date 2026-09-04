import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders landing page with sign-in form', () => {
  render(<App />);
  expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
});
