import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, ShieldCheck, SlidersHorizontal, UserCog, Users } from 'lucide-react';
import { usePermissions } from '../hooks/usePermissions';
import AccountsPanel from '../components/admin/AccountsPanel';
import SettingsPanel from '../components/admin/SettingsPanel';

const TABS = [
  { key: 'accounts', label: 'Accounts', icon: Users },
  { key: 'settings', label: 'Settings', icon: SlidersHorizontal },
];

/**
 * Administering the instance: who is on it, and how it is configured.
 *
 * Two tabs rather than two pages because they are one job — an operator setting
 * up a deployment creates the accounts and sets the tokens in the same sitting —
 * and because everything an admin can do to the instance being in one place is
 * the point of having the page at all.
 */
const AdminPage = () => {
  const navigate = useNavigate();
  const { canManageUsers } = usePermissions();
  const [tab, setTab] = useState('accounts');

  if (!canManageUsers) {
    return (
      <div className="min-h-screen bg-well flex items-center justify-center p-4">
        <div className="bg-p1 rounded-xl shadow-sm border border-ln p-8 max-w-md text-center">
          <ShieldCheck className="w-12 h-12 text-t3 mx-auto mb-3" />
          <h1 className="text-lg font-bold text-t1 mb-1">Admins only</h1>
          <p className="text-sm text-t2 mb-6">
            You need the admin platform role to administer this instance.
          </p>
          <button
            onClick={() => navigate('/datasets')}
            className="px-4 py-2 bg-accent text-onAccent rounded-lg hover:brightness-110 transition-colors"
          >
            Back to datasets
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-well">
      <div className="bg-p1 border-b border-ln">
        <div className="max-w-5xl mx-auto px-4 pt-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <UserCog className="w-6 h-6 text-ac" />
            <h1 className="text-2xl font-semibold tracking-tight text-t1">Administration</h1>
          </div>
          <button
            onClick={() => navigate('/datasets')}
            className="flex items-center gap-2 bg-hv hover:bg-hv2 text-t2 hover:text-t1 py-2 px-4 rounded-lg transition-colors duration-150"
          >
            <ArrowLeft className="w-4 h-4" />
            Datasets
          </button>
        </div>

        <div className="max-w-5xl mx-auto px-4 mt-4 flex gap-1">
          {TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === key
                  ? 'border-acLn text-ac'
                  : 'border-transparent text-t3 hover:text-t1'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-8">
        {tab === 'accounts' ? <AccountsPanel /> : <SettingsPanel />}
      </div>
    </div>
  );
};

export default AdminPage;
