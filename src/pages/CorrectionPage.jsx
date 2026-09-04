import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowDownWideNarrow,
  ArrowUpNarrowWide,
  CheckCircle2,
  Loader2,
  Play,
  Wrench,
} from 'lucide-react';
import {
  fetchCorrectionSummary,
  buildCorrectionQueue,
  fetchRejectionReasons,
} from '../api/reviews';
import { useAuth } from '../contexts/AuthContext';
import { useDataset } from '../contexts/DatasetContext';
import { useCorrection } from '../contexts/CorrectionContext';
import { usePermissions } from '../hooks/usePermissions';
import { Permission } from '../utils/permissions';
import RoleBadge from '../components/datasets/RoleBadge';
import DatasetGalleryHeader from '../components/datasets/gallery/DatasetGalleryHeader';

const readableError = (err, fallback) =>
  (err?.message || '').replace(/^API Error:\s*/i, '') || fallback;

/**
 * The correction session's launch pad.
 *
 * Picks an order (and optionally a subset of reasons), builds the queue of open
 * rejections, and hands it to the CorrectionContext — then jumps into the
 * annotation editor on the first item's image, where `CorrectionBar` drives the
 * rest. The queue is memory-only (a snapshot), so this page is where a session
 * always begins; a refresh mid-session lands back here.
 */
const CorrectionPage = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const { datasets } = useDataset();
  const correction = useCorrection();

  const dataset = useMemo(
    () => datasets?.find((d) => String(d.id) === String(datasetId)) || null,
    [datasets, datasetId]
  );
  const { can, role } = usePermissions(dataset);
  const mayCorrect = can(Permission.ANNOTATION_EDIT_OWN);

  const [summary, setSummary] = useState(null);
  const [reasons, setReasons] = useState([]);
  const [order, setOrder] = useState('oldest');
  const [selectedReasons, setSelectedReasons] = useState([]);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!datasetId || !isAuthenticated) return undefined;
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [summaryResponse, reasonResponse] = await Promise.all([
          fetchCorrectionSummary(datasetId),
          fetchRejectionReasons(),
        ]);
        if (cancelled) return;
        if (summaryResponse?.success) setSummary(summaryResponse.summary);
        setReasons(reasonResponse?.reasons || []);
      } catch (err) {
        if (!cancelled) setError(readableError(err, 'Could not load the correction overview.'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId, isAuthenticated]);

  const openCount = summary?.open_rejections ?? null;

  const toggleReason = (value) => {
    setSelectedReasons((current) =>
      current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value]
    );
  };

  const handleStart = useCallback(async () => {
    setBuilding(true);
    setError(null);
    try {
      const response = await buildCorrectionQueue(datasetId, {
        order,
        reasons: selectedReasons.length > 0 ? selectedReasons : null,
      });
      const items = response?.queue?.items || [];
      if (items.length === 0) {
        setError('Nothing matches — there is no correction work with those filters.');
        return;
      }
      correction.start(datasetId, items);
      navigate(`/dataset/${datasetId}/annotate/${items[0].image_id}`);
    } catch (err) {
      setError(readableError(err, 'Could not build the correction queue.'));
    } finally {
      setBuilding(false);
    }
  }, [datasetId, order, selectedReasons, correction, navigate]);

  const canStart = !building && (openCount == null || openCount > 0);

  return (
    <div className="h-screen flex flex-col bg-well">
      <DatasetGalleryHeader dataset={dataset} />

      <div className="bg-p1 border-b border-ln flex-shrink-0">
        <div className="px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <Wrench className="w-5 h-5 flex-shrink-0 text-ac" />
            {/* Going back, and the dataset's name, are the top bar's job now. */}
            <h1 className="text-lg font-semibold tracking-tight text-t1 truncate">
              Correct
            </h1>
            {role && <RoleBadge role={role} showDescription />}
          </div>
        </div>
      </div>

      {error && (
        <div className="px-4 py-2 bg-errBg border-b border-errLn flex-shrink-0">
          <p className="text-sm text-err">{error}</p>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto">
        {!mayCorrect && !loading ? (
          <div className="h-full flex items-center justify-center text-t3 text-sm">
            You do not have permission to correct annotations on this dataset.
          </div>
        ) : loading ? (
          <div className="h-full flex items-center justify-center text-t3">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            Loading…
          </div>
        ) : openCount === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <CheckCircle2 className="w-12 h-12 text-ac mb-4" />
            <h2 className="text-2xl font-bold text-t1 mb-2">Nothing to correct</h2>
            <p className="text-t2 mb-6">
              No annotations have been sent back for correction right now.
            </p>
            <button
              onClick={() => navigate(`/dataset/${datasetId}/datamanagement`)}
              className="px-5 py-2.5 rounded-lg font-semibold bg-accent text-onAccent hover:brightness-110 transition-colors"
            >
              Back to dataset
            </button>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto p-6 sm:p-8">
            <h2 className="text-2xl font-bold text-t1 mb-1">Start a correction session</h2>
            <p className="text-t2 mb-2">
              Work through every instance a reviewer sent back, one at a time, in the
              annotation editor.
            </p>
            {openCount != null && (
              <p className="text-sm font-medium text-ac mb-6">
                {openCount === 1
                  ? '1 instance sent back for correction'
                  : `${openCount} instances sent back for correction`}
                {summary?.affected_images > 0 &&
                  ` across ${summary.affected_images} image${
                    summary.affected_images === 1 ? '' : 's'
                  }.`}
              </p>
            )}

            {/* Order */}
            <div className="bg-p1 rounded-xl border border-ln p-4 mb-6">
              <div className="text-sm font-semibold text-t2 mb-3">Order</div>
              <div className="flex rounded-lg border border-ln2 overflow-hidden w-fit">
                <button
                  onClick={() => setOrder('oldest')}
                  className={`flex items-center gap-1.5 px-3 py-2 text-sm transition-colors ${
                    order === 'oldest'
                      ? 'bg-accent text-onAccent'
                      : 'bg-p1 text-t2 hover:bg-hv'
                  }`}
                >
                  <ArrowUpNarrowWide className="w-4 h-4" />
                  Oldest feedback first
                </button>
                <button
                  onClick={() => setOrder('newest')}
                  className={`flex items-center gap-1.5 px-3 py-2 text-sm transition-colors ${
                    order === 'newest'
                      ? 'bg-accent text-onAccent'
                      : 'bg-p1 text-t2 hover:bg-hv'
                  }`}
                >
                  <ArrowDownWideNarrow className="w-4 h-4" />
                  Newest first
                </button>
              </div>
            </div>

            {/* Reason filter */}
            <div className="bg-p1 rounded-xl border border-ln p-4 mb-6">
              <div className="text-sm font-semibold text-t2 mb-1">
                Limit to reasons ({selectedReasons.length || 'all'})
              </div>
              <p className="text-xs text-t3 mb-3">
                Leave all unchecked to correct every kind of feedback.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
                {reasons.map((reason) => (
                  <label
                    key={reason.value}
                    className="flex items-center gap-2 py-1 px-2 rounded hover:bg-hv cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={selectedReasons.includes(reason.value)}
                      onChange={() => toggleReason(reason.value)}
                      className="rounded border-ln2 text-ac focus:ring-ac"
                    />
                    <span className="text-sm text-t1">{reason.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <button
              onClick={handleStart}
              disabled={!canStart}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold transition-colors ${
                canStart
                  ? 'bg-accent text-onAccent hover:brightness-110'
                  : 'bg-hv2 text-t3 cursor-not-allowed'
              }`}
            >
              {building ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {building ? 'Building queue…' : 'Start correcting'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default CorrectionPage;
