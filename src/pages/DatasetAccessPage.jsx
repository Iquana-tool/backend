import React, { useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ClipboardList,
  Link2,
  Loader2,
  ShieldAlert,
  Table2,
  Users2,
} from 'lucide-react';
import { useDataset } from '../contexts/DatasetContext';
import { useDatasetAccess } from '../hooks/useDatasetAccess';
import { usePermissions } from '../hooks/usePermissions';
import { Permission } from '../utils/permissions';
import InvitesPanel from '../components/datasets/access/InvitesPanel';
import MembersPanel from '../components/datasets/access/MembersPanel';
import ReviewPolicyPanel from '../components/datasets/access/ReviewPolicyPanel';
import TaskAssignmentPanel from '../components/datasets/access/TaskAssignmentPanel';
import PermissionMatrix from '../components/datasets/PermissionMatrix';
import RoleBadge from '../components/datasets/RoleBadge';
import DatasetGalleryHeader from '../components/datasets/gallery/DatasetGalleryHeader';

const SECTIONS = {
  MEMBERS: 'members',
  INVITES: 'invites',
  ASSIGNMENTS: 'assignments',
  POLICY: 'policy',
  PERMISSIONS: 'permissions',
};

/**
 * Full-page access management for one dataset.
 *
 * The dataset card's modal covers the quick "add one person" case; this is where
 * there is room for the whole picture — collaborators, invite links, review
 * policy, the permission reference, and (later) who is assigned which images.
 */
const DatasetAccessPage = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const { datasets, fetchDatasets } = useDataset();

  const dataset = useMemo(
    () => datasets?.find((d) => String(d.id) === String(datasetId)) || null,
    [datasets, datasetId]
  );

  const { can } = usePermissions(dataset);
  const access = useDatasetAccess(dataset, { onChange: fetchDatasets });
  const {
    loading,
    error,
    role,
    members,
    canList,
    canInvite,
    canUpdateSettings,
  } = access;

  const [section, setSection] = useState(SECTIONS.MEMBERS);

  // The dataset list is what carries `my_permissions`, so wait for it before
  // deciding anything — otherwise a refresh straight onto this URL would briefly
  // look like a permission failure.
  if (!dataset) {
    return (
      <div className="min-h-screen bg-well flex items-center justify-center">
        {datasets?.length > 0 ? (
          <div className="bg-p1 rounded-xl shadow-sm border border-ln p-8 max-w-md text-center">
            <Users2 className="w-12 h-12 text-t3 mx-auto mb-3" />
            <h1 className="text-lg font-bold text-t1 mb-1">Dataset not found</h1>
            <p className="text-sm text-t2 mb-6">
              It may have been deleted, or your access to it removed.
            </p>
            <button
              onClick={() => navigate('/datasets')}
              className="px-4 py-2 bg-accent text-onAccent rounded-lg hover:brightness-110 transition-colors"
            >
              Back to datasets
            </button>
          </div>
        ) : (
          <div className="flex items-center text-t3">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            Loading dataset…
          </div>
        )}
      </div>
    );
  }

  if (!canList && !can(Permission.MEMBER_GRANT)) {
    return (
      <div className="min-h-screen bg-well flex items-center justify-center p-4">
        <div className="bg-p1 rounded-xl shadow-sm border border-ln p-8 max-w-md text-center">
          <ShieldAlert className="w-12 h-12 text-t3 mx-auto mb-3" />
          <h1 className="text-lg font-bold text-t1 mb-1">Not available</h1>
          <p className="text-sm text-t2 mb-6">
            Your role on this dataset does not include managing who has access.
          </p>
          <button
            onClick={() => navigate(`/dataset/${datasetId}/datamanagement`)}
            className="px-4 py-2 bg-accent text-onAccent rounded-lg hover:brightness-110 transition-colors"
          >
            Back to dataset
          </button>
        </div>
      </div>
    );
  }

  const navItems = [
    { id: SECTIONS.MEMBERS, label: 'Members', icon: Users2, count: members.length },
    ...(canInvite
      ? [{ id: SECTIONS.INVITES, label: 'Invite links', icon: Link2, count: access.invites.length }]
      : []),
    { id: SECTIONS.ASSIGNMENTS, label: 'Task assignment', icon: ClipboardList, soon: true },
    ...(canUpdateSettings
      ? [{ id: SECTIONS.POLICY, label: 'Review policy', icon: ShieldAlert }]
      : []),
    { id: SECTIONS.PERMISSIONS, label: 'What roles can do', icon: Table2 },
  ];

  return (
    <div className="min-h-screen bg-well">
      <DatasetGalleryHeader dataset={dataset} />

      {/* Header */}
      <div className="bg-p1 border-b border-ln">
        <div className="max-w-6xl mx-auto px-4 py-5">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <Users2 className="w-6 h-6 flex-shrink-0 text-ac" />
              {/* Going back is the top bar's job now, and it names the dataset. */}
              <h1 className="text-xl font-semibold tracking-tight text-t1 truncate">Manage access</h1>
            </div>
            {role && <RoleBadge role={role} showDescription />}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6">
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-errBg border border-errLn">
            <p className="text-sm text-err">{error}</p>
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Section nav */}
          <nav className="lg:w-56 flex-shrink-0">
            <ul className="flex lg:flex-col gap-1 overflow-x-auto lg:overflow-visible">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = section === item.id;
                return (
                  <li key={item.id} className="flex-shrink-0">
                    <button
                      onClick={() => setSection(item.id)}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                        isActive
                          ? 'bg-acS text-ac border border-acLn'
                          : 'text-t2 hover:bg-hv border border-transparent'
                      }`}
                    >
                      <Icon className="w-4 h-4 flex-shrink-0" />
                      <span className="flex-1 text-left">{item.label}</span>
                      {item.count !== undefined && (
                        <span className="text-xs text-t3">{item.count}</span>
                      )}
                      {item.soon && (
                        <span className="text-[10px] uppercase tracking-wide text-t3 border border-ln2 rounded px-1">
                          Soon
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          </nav>

          {/* Section body */}
          <div className="flex-1 min-w-0 bg-p1 rounded-xl border border-ln p-5">
            {loading ? (
              <div className="flex items-center justify-center py-16 text-t3">
                <Loader2 className="w-5 h-5 animate-spin mr-2" />
                Loading access settings…
              </div>
            ) : (
              <>
                {section === SECTIONS.MEMBERS && (
                  <MembersPanel access={access} datasetName={dataset.name} />
                )}
                {section === SECTIONS.INVITES && canInvite && (
                  <InvitesPanel access={access} />
                )}
                {section === SECTIONS.ASSIGNMENTS && (
                  <TaskAssignmentPanel members={members} />
                )}
                {section === SECTIONS.POLICY && canUpdateSettings && (
                  <ReviewPolicyPanel access={access} dataset={dataset} />
                )}
                {section === SECTIONS.PERMISSIONS && (
                  <PermissionMatrix highlightRole={role} />
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetAccessPage;
