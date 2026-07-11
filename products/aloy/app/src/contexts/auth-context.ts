import { createContext } from 'react';
import type { Session, SupabaseClient } from '@supabase/supabase-js';

export interface AuthCtx {
  session: Session | null;
  supabase: SupabaseClient;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

export const AuthContext = createContext<AuthCtx | null>(null);
