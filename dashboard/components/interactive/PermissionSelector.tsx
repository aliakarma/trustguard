'use client';

import React, { useState } from 'react';
import { PERMISSION_GROUPS } from '@/lib/constants';

interface PermissionSelectorProps {
  selected: string[];
  onChange: (selected: string[]) => void;
}

export default function PermissionSelector({
  selected,
  onChange,
}: PermissionSelectorProps) {
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({
    Location: true,
    Camera: true,
    Microphone: true,
  });

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const handlePermissionToggle = (perm: string) => {
    const isSelected = selected.includes(perm);
    if (isSelected) {
      onChange(selected.filter((p) => p !== perm));
    } else {
      onChange([...selected, perm]);
    }
  };

  const selectAllInGroup = (group: string, select: boolean) => {
    const permsInGroup = PERMISSION_GROUPS[group];
    if (select) {
      // Add all perms from group that are not already selected
      const newSelection = [...selected];
      permsInGroup.forEach((p) => {
        if (!newSelection.includes(p)) newSelection.push(p);
      });
      onChange(newSelection);
    } else {
      // Remove all perms from group
      onChange(selected.filter((p) => !permsInGroup.includes(p)));
    }
  };

  return (
    <div className="flex flex-col gap-3 p-4 glass-panel border border-subtle max-h-[400px] overflow-y-auto">
      <h3 className="stat-label mb-2">Declared Permissions</h3>
      {Object.entries(PERMISSION_GROUPS).map(([groupName, perms]) => {
        const isExpanded = !!expandedGroups[groupName];
        const groupSelectedCount = perms.filter((p) => selected.includes(p)).length;
        const allSelected = groupSelectedCount === perms.length;

        return (
          <div key={groupName} className="flex flex-col border-b border-subtle pb-2 mb-2 last:border-0 last:pb-0 last:mb-0">
            <div className="flex justify-between items-center py-1">
              <button
                type="button"
                onClick={() => toggleGroup(groupName)}
                className="text-xs font-bold text-primary flex items-center gap-1 hover:text-secondary"
              >
                <span>{isExpanded ? '▼' : '▶'}</span>
                <span>{groupName}</span>
                <span className="text-[10px] text-tertiary">
                  ({groupSelectedCount}/{perms.length})
                </span>
              </button>
              <button
                type="button"
                onClick={() => selectAllInGroup(groupName, !allSelected)}
                className="text-[10px] text-monitor hover:underline"
              >
                {allSelected ? 'Clear Group' : 'Select Group'}
              </button>
            </div>

            {isExpanded && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2 pl-4">
                {perms.map((perm) => {
                  const isChecked = selected.includes(perm);
                  return (
                    <label
                      key={perm}
                      className="flex items-center gap-2 text-xs text-secondary cursor-pointer hover:text-primary"
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => handlePermissionToggle(perm)}
                        className="rounded border-subtle text-monitor focus:ring-monitor h-3 w-3"
                      />
                      <span>{perm}</span>
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
