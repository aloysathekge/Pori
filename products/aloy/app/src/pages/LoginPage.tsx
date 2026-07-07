import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AloyMark } from '@/components/icons';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

export function LoginPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signIn(email, password);
      navigate('/chat');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sign in failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-950 p-4">
      <div className="w-full max-w-sm space-y-8">
        <div className="text-center">
          <span className="mb-4 inline-flex items-center gap-3">
            <AloyMark size={40} />
            <span className="font-display text-3xl font-semibold tracking-tight text-zinc-100">Aloy</span>
          </span>
          <h1 className="text-2xl font-bold text-zinc-100">Welcome back</h1>
          <p className="mt-1 text-sm text-zinc-400">Sign in to Aloy</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Email"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <Input
            label="Password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {error && (
            <p className="rounded-lg bg-red-900/30 px-3 py-2 text-sm text-red-400">
              {error}
            </p>
          )}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </Button>
        </form>

        <p className="text-center text-sm text-zinc-500">
          Don&apos;t have an account?{' '}
          <Link to="/signup" className="text-accent-400 hover:text-accent-300">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
