import { useState, useEffect } from 'react';
import { Plus, Trash2, Pencil, Bot } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { TextArea } from '@/components/ui/TextArea';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import {
  listAgentConfigs,
  createAgentConfig,
  updateAgentConfig,
  deleteAgentConfig,
  getAvailableModels,
  getAvailableTools,
} from '@/api/agentConfigs';
import type { AgentConfigResponse, AgentConfigCreate, ToolInfo, Provider } from '@/types';

const PROVIDERS: { value: Provider; label: string }[] = [
  { value: 'google', label: 'Google' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'openai', label: 'OpenAI' },
];

export function AgentConfigsPage() {
  const [configs, setConfigs] = useState<AgentConfigResponse[]>([]);
  const [models, setModels] = useState<Record<string, string[]>>({});
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<AgentConfigResponse | null>(null);

  const [form, setForm] = useState<AgentConfigCreate>({
    name: '',
    provider: 'google',
    model: 'gemini-2.5-flash',
    temperature: 0,
    max_steps: 15,
    system_prompt: null,
    tools: null,
    is_default: false,
  });

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const [c, m, t] = await Promise.all([
        listAgentConfigs(),
        getAvailableModels(),
        getAvailableTools(),
      ]);
      setConfigs(c);
      setModels(m);
      setTools(t);
    } catch {
      // handle silently
    } finally {
      setLoading(false);
    }
  }

  function openCreate() {
    setEditing(null);
    setForm({
      name: '',
      provider: 'google',
      model: 'gemini-2.5-flash',
      temperature: 0,
      max_steps: 15,
      system_prompt: null,
      tools: null,
      is_default: false,
    });
    setModalOpen(true);
  }

  function openEdit(config: AgentConfigResponse) {
    setEditing(config);
    setForm({
      name: config.name,
      provider: config.provider as Provider,
      model: config.model,
      temperature: config.temperature,
      max_steps: config.max_steps,
      system_prompt: config.system_prompt,
      tools: config.tools,
      is_default: config.is_default,
    });
    setModalOpen(true);
  }

  async function handleSave() {
    try {
      if (editing) {
        await updateAgentConfig(editing.id, form);
      } else {
        await createAgentConfig(form);
      }
      setModalOpen(false);
      await load();
    } catch {
      // handle silently
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Delete this agent config?')) return;
    try {
      await deleteAgentConfig(id);
      await load();
    } catch {
      // handle silently
    }
  }

  function toggleTool(toolName: string) {
    setForm((prev) => {
      const current = prev.tools || [];
      const next = current.includes(toolName)
        ? current.filter((t) => t !== toolName)
        : [...current, toolName];
      return { ...prev, tools: next.length > 0 ? next : null };
    });
  }

  const providerModels = models[form.provider] || [];

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">Agent Configs</h1>
          <p className="mt-1 text-sm text-zinc-400">
            Customize LLM settings, tools, and system prompts
          </p>
        </div>
        <Button onClick={openCreate}>
          <Plus size={16} /> New Config
        </Button>
      </div>

      {configs.length === 0 ? (
        <Card className="py-12 text-center text-zinc-500">
          <Bot size={32} className="mx-auto mb-3 text-zinc-600" />
          <p>No agent configs yet. Create one to get started.</p>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {configs.map((config) => (
            <Card key={config.id} className="space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold text-zinc-100">{config.name}</h3>
                  <p className="text-xs text-zinc-500">
                    {config.provider}/{config.model}
                  </p>
                </div>
                <div className="flex gap-1">
                  {config.is_default && <Badge color="indigo">Default</Badge>}
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge color="blue">temp: {config.temperature}</Badge>
                <Badge color="gray">steps: {config.max_steps}</Badge>
                {config.tools?.map((t) => (
                  <Badge key={t} color="green">
                    {t}
                  </Badge>
                ))}
              </div>
              {config.system_prompt && (
                <p className="line-clamp-2 text-xs text-zinc-500">
                  {config.system_prompt}
                </p>
              )}
              <div className="flex gap-2 pt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => openEdit(config)}
                >
                  <Pencil size={14} /> Edit
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(config.id)}
                >
                  <Trash2 size={14} /> Delete
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editing ? 'Edit Agent Config' : 'New Agent Config'}
      >
        <div className="space-y-4">
          <Input
            label="Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="My Custom Agent"
          />
          <Select
            label="Provider"
            options={PROVIDERS}
            value={form.provider}
            onChange={(e) =>
              setForm({
                ...form,
                provider: e.target.value as Provider,
                model: models[e.target.value]?.[0] || '',
              })
            }
          />
          <Select
            label="Model"
            options={providerModels.map((m) => ({ value: m, label: m }))}
            value={form.model}
            onChange={(e) => setForm({ ...form, model: e.target.value })}
          />
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Temperature"
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={form.temperature}
              onChange={(e) =>
                setForm({ ...form, temperature: parseFloat(e.target.value) })
              }
            />
            <Input
              label="Max Steps"
              type="number"
              min="1"
              max="50"
              value={form.max_steps}
              onChange={(e) =>
                setForm({ ...form, max_steps: parseInt(e.target.value) })
              }
            />
          </div>
          <TextArea
            label="System Prompt"
            rows={3}
            value={form.system_prompt || ''}
            onChange={(e) =>
              setForm({
                ...form,
                system_prompt: e.target.value || null,
              })
            }
            placeholder="You are a helpful assistant..."
          />
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-300">
              Tools
            </label>
            <div className="flex flex-wrap gap-2">
              {tools.map((tool) => {
                const active = form.tools?.includes(tool.name);
                return (
                  <button
                    key={tool.name}
                    onClick={() => toggleTool(tool.name)}
                    className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                      active
                        ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
                        : 'border-zinc-700 text-zinc-400 hover:border-zinc-600'
                    }`}
                    title={tool.description}
                  >
                    {tool.name}
                  </button>
                );
              })}
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-zinc-300">
            <input
              type="checkbox"
              checked={form.is_default}
              onChange={(e) =>
                setForm({ ...form, is_default: e.target.checked })
              }
              className="rounded border-zinc-600 bg-zinc-800"
            />
            Set as default
          </label>
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="secondary" onClick={() => setModalOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!form.name || !form.model}>
              {editing ? 'Update' : 'Create'}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
