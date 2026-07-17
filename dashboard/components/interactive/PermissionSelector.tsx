'use client';

import React, { useState } from 'react';
import { PERMISSION_GROUPS } from '@/lib/constants';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface PermissionSelectorProps {
  selected: string[];
  onChange: (selected: string[]) => void;
}

export default function PermissionSelector({ selected, onChange }: PermissionSelectorProps) {
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({
    Location: true,
    Camera: true,
    Microphone: true,
  });

  const toggleGroup = (group: string) =>
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));

  const handlePermissionToggle = (perm: string) => {
    onChange(selected.includes(perm) ? selected.filter((p) => p !== perm) : [...selected, perm]);
  };

  const selectAllInGroup = (group: string, select: boolean) => {
    const permsInGroup = PERMISSION_GROUPS[group];
    if (select) {
      const next = [...selected];
      permsInGroup.forEach((p) => { if (!next.includes(p)) next.push(p); });
      onChange(next);
    } else {
      onChange(selected.filter((p) => !permsInGroup.includes(p)));
    }
  };

  return (
    <div className="glass-panel p-5 flex flex-col gap-2 max-h-[420px] overflow-y-auto">
      <div className="flex items-center justify-between mb-1">
        <h3 className="stat-label">Declared Permissions</h3>
        <span className="chip">{selected.length} selected</span>
      </div>

      {Object.entries(PERMISSION_GROUPS).map(([groupName, perms]) => {
        const isExpanded = !!expandedGroups[groupName];
        const groupSelectedCount = perms.filter((p) => selected.includes(p)).length;
        const allSelected = groupSelectedCount === perms.length;

        return (
          <div key={groupName} className="border-b border-subtle pb-2 last:border-0 last:pb-0">
            <div className="flex justify-between items-center gap-2 py-1">
              <button
                type="button"
                onClick={() => toggleGroup(groupName)}
                className="flex items-center gap-1.5 text-xs font-semibold text-primary hover:text-monitor transition-colors min-w-0"
              >
                {isExpanded ? <ChevronDown size={14} className="text-tertiary flex-shrink-0" /> : <ChevronRight size={14} className="text-tertiary flex-shrink-0" />}
                <span className="truncate">{groupName}</span>
                <span
                  className={`text-[10px] text-mono px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                    groupSelectedCount > 0 ? 'bg-monitor/15 text-monitor' : 'bg-surface text-tertiary'
                  }`}
                >
                  {groupSelectedCount}/{perms.length}
                </span>
              </button>
              <button
                type="button"
                onClick={() => selectAllInGroup(groupName, !allSelected)}
                className="text-[10px] font-semibold text-tertiary hover:text-monitor transition-colors flex-shrink-0"
              >
                {allSelected ? 'Clear' : 'All'}
              </button>
            </div>

            {isExpanded && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5 mt-1.5 ps-5">
                {perms.map((perm) => {
                  const isChecked = selected.includes(perm);
                  return (
                    <label
                      key={perm}
                      className={`flex items-center gap-2 text-[11px] text-mono cursor-pointer rounded-md px-1.5 py-1 transition-colors ${
                        isChecked ? 'text-primary bg-monitor/10' : 'text-secondary hover:bg-surface'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => handlePermissionToggle(perm)}
                        className="h-3.5 w-3.5 rounded flex-shrink-0"
                      />
                      <span className="truncate">{perm}</span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
