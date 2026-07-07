import { useState, useEffect } from 'react';
import { Plus, Trash2, Pencil, Users, Play, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { TextArea } from '@/components/ui/TextArea';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import {
  listTeams,
  createTeam,
  updateTeam,
  deleteTeam,
  runTeam,
} from '@/api/teams';
import type { TeamConfigResponse, TeamConfigCreate, TeamMember, TeamMode } from '@/types';

const MODES: { value: TeamMode; label: string }[] = [
  { value: 'router', label: 'Router (pick best member)' },
  { value: 'broadcast', label: 'Broadcast (all in parallel)' },
  { value: 'delegate', label: 'Delegate (multi-step plan)' },
];

function emptyMember(): TeamMember {
  return { name: '', description: '', llm_config: null, tools: null };
}

export function TeamsPage() {
  const [teams, setTeams] = useState<TeamConfigResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [runModalOpen, setRunModalOpen] = useState(false);
  const [runTeamId, setRunTeamId] = useState('');
  const [runTask, setRunTask] = useState('');
  const [runResult, setRunResult] = useState<string | null>(null);
  const [runLoading, setRunLoading] = useState(false);
  const [editing, setEditing] = useState<TeamConfigResponse | null>(null);

  const [form, setForm] = useState<TeamConfigCreate>({
    name: '',
    mode: 'router',
    members: [emptyMember()],
    max_delegation_steps: 10,
    max_concurrent_members: 5,
  });

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      setTeams(await listTeams());
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
      mode: 'router',
      members: [emptyMember()],
      max_delegation_steps: 10,
      max_concurrent_members: 5,
    });
    setModalOpen(true);
  }

  function openEdit(team: TeamConfigResponse) {
    setEditing(team);
    setForm({
      name: team.name,
      mode: team.mode,
      members: team.members,
      max_delegation_steps: team.max_delegation_steps,
      max_concurrent_members: team.max_concurrent_members,
    });
    setModalOpen(true);
  }

  async function handleSave() {
    try {
      if (editing) {
        await updateTeam(editing.id, form);
      } else {
        await createTeam(form);
      }
      setModalOpen(false);
      await load();
    } catch {
      // handle silently
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Delete this team?')) return;
    try {
      await deleteTeam(id);
      await load();
    } catch {
      // handle silently
    }
  }

  function updateMember(index: number, updates: Partial<TeamMember>) {
    setForm((prev) => {
      const members = [...prev.members];
      members[index] = { ...members[index], ...updates };
      return { ...prev, members };
    });
  }

  function removeMember(index: number) {
    setForm((prev) => ({
      ...prev,
      members: prev.members.filter((_, i) => i !== index),
    }));
  }

  async function handleRun() {
    if (!runTask.trim()) return;
    setRunLoading(true);
    setRunResult(null);
    try {
      const result = await runTeam(runTeamId, runTask);
      setRunResult(result.final_answer);
    } catch (err) {
      setRunResult(`Error: ${err instanceof Error ? err.message : 'Unknown'}`);
    } finally {
      setRunLoading(false);
    }
  }

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
          <h1 className="text-xl font-bold text-zinc-100">Teams</h1>
          <p className="mt-1 text-sm text-zinc-400">
            Multi-agent team configurations
          </p>
        </div>
        <Button onClick={openCreate}>
          <Plus size={16} /> New Team
        </Button>
      </div>

      {teams.length === 0 ? (
        <Card className="py-12 text-center text-zinc-500">
          <Users size={32} className="mx-auto mb-3 text-zinc-600" />
          <p>No teams yet. Create one to get started.</p>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {teams.map((team) => (
            <Card key={team.id} className="space-y-3">
              <div className="flex items-start justify-between">
                <h3 className="font-semibold text-zinc-100">{team.name}</h3>
                <Badge
                  color={
                    team.mode === 'delegate'
                      ? 'accent'
                      : team.mode === 'broadcast'
                        ? 'green'
                        : 'blue'
                  }
                >
                  {team.mode}
                </Badge>
              </div>
              <div className="space-y-1">
                {team.members.map((m, i) => (
                  <div key={i} className="text-xs text-zinc-400">
                    <span className="font-medium text-zinc-300">{m.name}</span>
                    {' — '}
                    {m.description}
                  </div>
                ))}
              </div>
              <div className="flex gap-2 pt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setRunTeamId(team.id);
                    setRunTask('');
                    setRunResult(null);
                    setRunModalOpen(true);
                  }}
                >
                  <Play size={14} /> Run
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => openEdit(team)}
                >
                  <Pencil size={14} /> Edit
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(team.id)}
                >
                  <Trash2 size={14} />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create/Edit Modal */}
      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editing ? 'Edit Team' : 'New Team'}
      >
        <div className="max-h-96 space-y-4 overflow-y-auto">
          <Input
            label="Team Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
          />
          <Select
            label="Mode"
            options={MODES}
            value={form.mode}
            onChange={(e) =>
              setForm({ ...form, mode: e.target.value as TeamMode })
            }
          />
          <div>
            <div className="mb-2 flex items-center justify-between">
              <label className="text-sm font-medium text-zinc-300">
                Members
              </label>
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  setForm({ ...form, members: [...form.members, emptyMember()] })
                }
              >
                <Plus size={14} /> Add
              </Button>
            </div>
            {form.members.map((member, i) => (
              <div
                key={i}
                className="mb-3 space-y-2 rounded-lg border border-zinc-800 p-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500">Member {i + 1}</span>
                  {form.members.length > 1 && (
                    <button onClick={() => removeMember(i)}>
                      <X size={14} className="text-zinc-500 hover:text-red-600" />
                    </button>
                  )}
                </div>
                <Input
                  placeholder="Name"
                  value={member.name}
                  onChange={(e) => updateMember(i, { name: e.target.value })}
                />
                <Input
                  placeholder="Description"
                  value={member.description}
                  onChange={(e) =>
                    updateMember(i, { description: e.target.value })
                  }
                />
              </div>
            ))}
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="secondary" onClick={() => setModalOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!form.name}>
              {editing ? 'Update' : 'Create'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Run Modal */}
      <Modal
        open={runModalOpen}
        onClose={() => setRunModalOpen(false)}
        title="Run Team Task"
      >
        <div className="space-y-4">
          <TextArea
            label="Task"
            rows={3}
            value={runTask}
            onChange={(e) => setRunTask(e.target.value)}
            placeholder="Compare Rust vs Go for backend services..."
          />
          <Button onClick={handleRun} disabled={runLoading || !runTask.trim()}>
            {runLoading ? (
              <>
                <Spinner className="h-4 w-4" /> Running...
              </>
            ) : (
              'Execute'
            )}
          </Button>
          {runResult && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 text-sm text-zinc-300">
              <p className="whitespace-pre-wrap">{runResult}</p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}
