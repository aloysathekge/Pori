import { type ButtonHTMLAttributes, forwardRef } from 'react';

const variants = {
  primary:
    'bg-accent-600 text-white shadow-sm shadow-accent-950/50 hover:bg-accent-500 focus-visible:ring-accent-500',
  secondary:
    'bg-zinc-800 text-zinc-100 border border-zinc-700 hover:bg-zinc-700 focus-visible:ring-zinc-500',
  danger:
    'bg-red-600 text-white hover:bg-red-500 focus-visible:ring-red-500',
  ghost:
    'bg-transparent text-zinc-300 hover:bg-zinc-800 focus-visible:ring-zinc-500',
  outline:
    'border border-zinc-600 text-zinc-300 hover:border-accent-600 hover:text-accent-300 focus-visible:ring-zinc-500',
} as const;

const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
  icon: 'p-2',
} as const;

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variants;
  size?: keyof typeof sizes;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'primary', size = 'md', className = '', ...props }, ref) => (
    <button
      ref={ref}
      className={`inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-150 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-900 disabled:opacity-50 disabled:pointer-events-none ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    />
  ),
);

Button.displayName = 'Button';
