import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle, CheckCircle2, Eye, EyeOff, Loader2, RotateCcw, Save, Send,
} from 'lucide-react';
import * as api from '../../api';
import { useToast } from '../../contexts/ToastContext';

const readableError = (err, fallback) =>
  (err?.message || '').replace(/^API Error:\s*/i, '') || fallback;

/**
 * The deployment's own configuration: credentials, branding, registration policy.
 *
 * Each value resolves as *stored override, then environment, then default*, so a
 * field left alone here keeps behaving exactly as the deployment's `.env` says.
 * Editing one stores an override; "Reset" removes it and falls back again.
 *
 * Secrets are write-only. The backend never sends a key back — only whether one
 * is set and its last four characters — so a secret field always renders empty
 * and an empty secret field means "leave it alone", never "clear it".
 */
const SettingsPanel = () => {
  const { addToast } = useToast();

  const [groups, setGroups] = useState([]);
  const [settings, setSettings] = useState([]);
  const [aiService, setAiService] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pushing, setPushing] = useState(false);
  const [error, setError] = useState(null);
  const [revealed, setRevealed] = useState({});
  // Only the fields actually touched, so saving one tab cannot clobber another
  // admin's edit to a field this page happens to be displaying.
  const [edits, setEdits] = useState({});

  const applyPayload = useCallback((payload) => {
    setGroups(payload.groups || []);
    setSettings(payload.settings || []);
    setAiService(payload.ai_service || null);
    setEdits({});
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      applyPayload(await api.fetchSettings());
    } catch (err) {
      setError(readableError(err, 'Could not load the instance settings.'));
    } finally {
      setLoading(false);
    }
  }, [applyPayload]);

  useEffect(() => {
    load();
  }, [load]);

  const dirty = Object.keys(edits).length > 0;

  const byGroup = useMemo(() => {
    const map = {};
    settings.forEach((setting) => {
      (map[setting.group] ||= []).push(setting);
    });
    return map;
  }, [settings]);

  const valueOf = (setting) =>
    edits[setting.key] ?? (setting.kind === 'secret' ? '' : setting.value ?? '');

  const edit = (key, value) => setEdits((prev) => ({ ...prev, [key]: value }));

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const response = await api.updateSettings(edits);
      applyPayload(response);
      addToast({ message: response.message, type: 'success' });
      // A failed push is not a failed save: the value is stored either way, and
      // the panel below now shows the two sides disagreeing.
      if (response.push && !response.push.pushed) {
        addToast({
          message: `Saved, but the AI service could not be reached: ${response.push.error}`,
          type: 'error',
          duration: 8000,
        });
      }
    } catch (err) {
      setError(readableError(err, 'Could not save the settings.'));
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async (setting) => {
    setSaving(true);
    setError(null);
    try {
      const response = await api.clearSetting(setting.key);
      applyPayload(response);
      addToast({ message: response.message, type: 'success' });
    } catch (err) {
      setError(readableError(err, `Could not reset ${setting.label}.`));
    } finally {
      setSaving(false);
    }
  };

  const handlePush = async () => {
    setPushing(true);
    try {
      const response = await api.pushSettings();
      setAiService(response.ai_service || null);
      addToast({ message: response.message, type: response.success ? 'success' : 'error' });
    } catch (err) {
      setError(readableError(err, 'Could not reach the AI service.'));
    } finally {
      setPushing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-t3">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        Loading settings…
      </div>
    );
  }

  const hfToken = settings.find((setting) => setting.key === 'hf_token');
  // The AI service keeps pushed credentials in memory, so a restart drops them.
  // Saying so beats letting an operator wonder why gated weights stopped loading.
  const outOfSync =
    aiService?.reachable && hfToken && hfToken.is_set && !aiService.hf_token_set;

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm text-t2 max-w-2xl">
          Values set here override this deployment's environment and take effect without a
          restart. A field left untouched keeps using whatever the deployment configures,
          and <span className="font-medium text-t1">Reset</span> puts an overridden one back.
        </p>
        <button
          onClick={handleSave}
          disabled={!dirty || saving}
          className="flex items-center gap-2 shrink-0 bg-accent text-onAccent px-4 py-2 rounded-lg hover:brightness-110 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          {dirty ? 'Save changes' : 'Saved'}
        </button>
      </div>

      {error && (
        <div className="p-4 bg-errBg border border-errLn rounded-lg">
          <p className="text-err text-sm">{error}</p>
        </div>
      )}

      {groups.map((group) => (
        <section key={group.key} className="bg-p1 rounded-xl border border-ln overflow-hidden">
          <header className="px-5 py-4 border-b border-ln bg-well">
            <h3 className="font-semibold text-t1">{group.label}</h3>
            <p className="text-xs text-t3 mt-0.5">{group.description}</p>
          </header>

          <div className="divide-y divide-ln">
            {(byGroup[group.key] || []).map((setting) => (
              <div key={setting.key} className="px-5 py-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <label
                      htmlFor={`setting-${setting.key}`}
                      className="text-sm font-medium text-t1"
                    >
                      {setting.label}
                    </label>
                    <p className="text-xs text-t3 mt-0.5">{setting.description}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {setting.scope === 'ai-service' && (
                      <span
                        className="px-2 py-0.5 rounded-full bg-hv text-[11px] text-t2"
                        title="Stored here and sent to the AI service, which is what actually uses it."
                      >
                        AI service
                      </span>
                    )}
                    {setting.overridden && (
                      <button
                        onClick={() => handleReset(setting)}
                        disabled={saving}
                        title={`Remove the override and fall back to ${setting.env_var}`}
                        className="flex items-center gap-1 text-xs text-t3 hover:text-t1 transition-colors disabled:opacity-50"
                      >
                        <RotateCcw className="w-3 h-3" />
                        Reset
                      </button>
                    )}
                  </div>
                </div>

                <div className="mt-2 flex items-stretch gap-2">
                  {setting.kind === 'bool' ? (
                    <select
                      id={`setting-${setting.key}`}
                      value={valueOf(setting)}
                      onChange={(e) => edit(setting.key, e.target.value)}
                      className="px-3 py-2 border border-ln2 rounded-lg bg-p1 text-t1 focus:ring-2 focus:ring-ac focus:outline-none"
                    >
                      <option value="false">Off</option>
                      <option value="true">On</option>
                    </select>
                  ) : (
                    <>
                      <input
                        id={`setting-${setting.key}`}
                        type={
                          setting.kind === 'secret' && !revealed[setting.key]
                            ? 'password'
                            : 'text'
                        }
                        value={valueOf(setting)}
                        onChange={(e) => edit(setting.key, e.target.value)}
                        autoComplete="off"
                        placeholder={
                          setting.kind === 'secret' && setting.is_set
                            ? `Set (${setting.hint}) — type to replace`
                            : setting.placeholder
                        }
                        className={`flex-1 min-w-0 px-3 py-2 border border-ln2 rounded-lg bg-p1 text-t1 focus:ring-2 focus:ring-ac focus:outline-none ${
                          setting.kind === 'secret' ? 'font-mono text-sm' : ''
                        }`}
                      />
                      {setting.kind === 'secret' && (
                        <button
                          type="button"
                          onClick={() =>
                            setRevealed((prev) => ({
                              ...prev,
                              [setting.key]: !prev[setting.key],
                            }))
                          }
                          title={revealed[setting.key] ? 'Hide' : 'Show what you typed'}
                          className="px-3 bg-hv hover:bg-hv2 text-t2 hover:text-t1 rounded-lg transition-colors"
                        >
                          {revealed[setting.key]
                            ? <EyeOff className="w-4 h-4" />
                            : <Eye className="w-4 h-4" />}
                        </button>
                      )}
                    </>
                  )}
                </div>

                <p className="mt-1 text-[11px] text-t3">
                  {setting.overridden
                    ? `Overriding ${setting.env_var}${
                        setting.updated_by ? ` — set by ${setting.updated_by}` : ''
                      }`
                    : `From ${setting.env_var}`}
                  {setting.kind === 'secret' && ' · leave empty to keep the stored value'}
                </p>
              </div>
            ))}
          </div>
        </section>
      ))}

      {/* AI service state. Shown because the settings above are stored here but
          consumed over there, and the two can disagree. */}
      <section className="bg-p1 rounded-xl border border-ln p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-semibold text-t1 flex items-center gap-2">
              AI service
              {aiService?.reachable ? (
                <CheckCircle2 className="w-4 h-4 text-ok" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-t3" />
              )}
            </h3>
            <p className="text-xs text-t3 mt-1 max-w-xl">
              {!aiService?.reachable
                ? 'Not reachable. Settings are stored here regardless — send them once it is running.'
                : outOfSync
                  ? 'Running without the Hugging Face token stored here — most likely it was restarted. Send it again to restore access to gated model weights.'
                  : aiService.hf_token_set
                    ? `Holding a Hugging Face token (${aiService.hf_token_hint}). Credentials live in memory there, so a restart drops them.`
                    : 'Holding no Hugging Face token. Gated model weights will not download.'}
            </p>
          </div>
          <button
            onClick={handlePush}
            disabled={pushing}
            className="flex items-center gap-2 shrink-0 bg-hv hover:bg-hv2 text-t2 hover:text-t1 px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
          >
            {pushing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            Send to AI service
          </button>
        </div>
      </section>
    </div>
  );
};

export default SettingsPanel;
