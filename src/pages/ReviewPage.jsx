import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useParams } from 'react-router-dom';
import { ArrowLeft, ClipboardCheck, Loader2 } from 'lucide-react';
import * as api from '../api';
import { fetchReviewSummary, buildReviewQueue } from '../api/reviews';
import { useAuth } from '../contexts/AuthContext';
import { useDataset } from '../contexts/DatasetContext';
import { usePermissions } from '../hooks/usePermissions';
import { Permission } from '../utils/permissions';
import { extractLabelsFromResponse } from '../utils/labelHierarchy';
import RoleBadge from '../components/datasets/RoleBadge';
import DatasetGalleryHeader from '../components/datasets/gallery/DatasetGalleryHeader';
import ReviewSetup from '../components/review/ReviewSetup';
import ReviewSession from '../components/review/ReviewSession';

const readableError = (err, fallback) =>
  (err?.message || '').replace(/^API Error:\s*/i, '') || fallback;

/**
 * The review workflow: pick a granularity, get a queue, work through it.
 *
 * Two phases on one page. "Setup" chooses what to review (whole images,
 * instances ordered by hierarchy, or a custom label selection) and how to order
 * it; "session" plays the resulting queue one item at a time. The queue lives in
 * memory only — a refresh returns to setup, which is the honest thing to do
 * since the backend queue is a snapshot, not a reservation.
 */
const ReviewPage = () => {
  const { datasetId } = useParams();
  const location = useLocation();
  const { isAuthenticated } = useAuth();
  const { datasets } = useDataset();

  const dataset = useMemo(
    () => datasets?.find((d) => String(d.id) === String(datasetId)) || null,
    [datasets, datasetId]
  );
  const { can, role } = usePermissions(dataset);
  const mayReview = can(Permission.REVIEW_APPROVE);

  const [summary, setSummary] = useState(null);
  const [labels, setLabels] = useState([]);
  const [queue, setQueue] = useState(null);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState(null);

  const loadSummary = useCallback(async () => {
    const response = await fetchReviewSummary(datasetId);
    if (response?.success) setSummary(response.summary);
  }, [datasetId]);

  useEffect(() => {
    if (!datasetId || !isAuthenticated) return undefined;
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [summaryResponse, labelResponse] = await Promise.all([
          fetchReviewSummary(datasetId),
          api.fetchLabels(datasetId),
        ]);
        if (cancelled) return;
        if (summaryResponse?.success) setSummary(summaryResponse.summary);
        setLabels(extractLabelsFromResponse(labelResponse));
      } catch (err) {
        if (!cancelled) setError(readableError(err, 'Could not load the review overview.'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId, isAuthenticated]);

  const labelsById = useMemo(
    () => Object.fromEntries(labels.map((label) => [label.id, label])),
    [labels]
  );

  const handleStart = async (options) => {
    setBuilding(true);
    setError(null);
    try {
      const response = await buildReviewQueue(datasetId, options);
      if (response?.success) setQueue(response.queue);
    } catch (err) {
      setError(readableError(err, 'Could not build the review queue.'));
    } finally {
      setBuilding(false);
    }
  };

  // Leaving a session refreshes the counts: they just changed by however many
  // items were approved or sent back.
  const handleExitSession = useCallback(() => {
    setQueue(null);
    loadSummary().catch(() => {});
  }, [loadSummary]);

  return (
    <div className="h-screen flex flex-col bg-well">
      <DatasetGalleryHeader dataset={dataset} />

      <div className="bg-p1 border-b border-ln flex-shrink-0">
        <div className="px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            {/* The "Back" this replaced is the top bar's job now. Ending a
                session is not navigation — it drops the queue and stays — so
                that half of the button stays here, and only while it applies. */}
            {queue && (
              <>
                <button
                  onClick={handleExitSession}
                  className="flex items-center gap-2 text-t2 hover:text-ac transition-colors duration-150 flex-shrink-0"
                >
                  <ArrowLeft className="w-5 h-5" />
                  <span className="hidden sm:inline">End session</span>
                </button>
                <div className="h-6 w-px bg-ln flex-shrink-0" />
              </>
            )}
            <ClipboardCheck className="w-5 h-5 flex-shrink-0 text-ac" />
            {/* The dataset is named in the top bar; repeating it here read as
                a breadcrumb to a page you were already on. */}
            <h1 className="text-lg font-semibold tracking-tight text-t1 truncate">
              Review
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

      <div className="flex-1 min-h-0">
        {!mayReview && !loading ? (
          <div className="h-full flex items-center justify-center text-t3 text-sm">
            You do not have review permissions on this dataset.
          </div>
        ) : loading ? (
          <div className="h-full flex items-center justify-center text-t3">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            Loading…
          </div>
        ) : queue ? (
          <ReviewSession
            queue={queue}
            labels={labels}
            labelsById={labelsById}
            onExit={handleExitSession}
          />
        ) : (
          <ReviewSetup
            summary={summary}
            labels={labels}
            building={building}
            onStart={handleStart}
            defaultOnlySubmitted={location.state?.onlySubmitted !== false}
          />
        )}
      </div>
    </div>
  );
};

export default ReviewPage;
