import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { User } from 'lucide-react';
import { useAuth } from '../../../contexts/AuthContext';
import AuthButtons from '../../auth/AuthButtons';
import DocsLink from '../../ui/DocsLink';
import ReportBugLink from '../../ui/ReportBugLink';
import ThemeToggle from '../../ui/ThemeToggle';
import Wordmark from '../../Wordmark';
import DatasetNav from './DatasetNav';

/**
 * The top bar every dataset page sits under.
 *
 * One row, and every part of it is navigation: the wordmark goes up to the
 * dataset list, the dataset name goes to this dataset's overview, and the menus
 * after it reach every page within it. That is what the left sidebar used to
 * stand in for — badly, since it only repeated the dataset name and the label
 * list while taking a fixed 25rem from every page.
 *
 * One size, too. There used to be a `density` prop, and the pages that asked for
 * "compact" made the bar visibly change height and type size as you navigated
 * between them — chrome that moves is chrome you have to re-read. It bought back
 * a few pixels when this was two rows; at one row it is not worth the flicker.
 *
 * Rendered by `DatasetManagementLayout` and, directly, by the dataset pages that
 * bring their own layout (Review, Correct, Manage Access). The annotation canvas
 * is deliberately not one of them: it owns the whole viewport and has its own
 * chrome, and a second bar over it would take space from the image.
 */
const DatasetGalleryHeader = ({ dataset }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, user } = useAuth();

  const overviewPath = dataset?.id ? `/dataset/${dataset.id}/datamanagement` : null;
  const onOverview = location.pathname === overviewPath;

  return (
    <nav className="bg-p1 text-t1 border-b border-ln sticky top-0 z-50">
      {/* Wraps rather than overflows: Manage Access is reachable below the
          1024px cutoff the management pages set, and this bar shows there too. */}
      <div className="max-w-full mx-auto px-4 py-2.5 flex flex-wrap items-center justify-between gap-y-2 gap-x-4">
        <div className="flex flex-wrap items-center gap-y-2 gap-x-4">
          <button
            onClick={() => navigate("/datasets")}
            title="All datasets"
            className="text-2xl font-bold hover:text-ac transition-colors duration-150"
          >
            <Wordmark />
          </button>

          <div className="h-6 w-px bg-ln"></div>

          {/* The dataset's name is its overview link — the one page the sections
              below all hang off. It carries the active tint when you are on it,
              so the bar still says where you are. */}
          <button
            onClick={() => overviewPath && navigate(overviewPath)}
            aria-current={onOverview ? "page" : undefined}
            title="Dataset overview"
            className={`text-lg px-2.5 py-1 font-medium rounded-lg transition-colors ${
              onOverview ? "bg-acS text-ac" : "text-t2 hover:bg-hv hover:text-t1"
            }`}
          >
            {dataset?.name}
          </button>

          <div className="h-6 w-px bg-ln"></div>

          <DatasetNav dataset={dataset} datasetId={dataset?.id} />
        </div>

        <div className="flex items-center space-x-4">
          {isAuthenticated && user && (
            <div className="flex items-center space-x-2 px-3 py-1.5 text-sm text-t3">
              <User className="w-4 h-4" />
              <span className="font-medium">{user.username}</span>
            </div>
          )}
          <DocsLink className="flex items-center space-x-2 py-2 px-4 rounded-lg bg-hv hover:bg-hv2 text-t2 hover:text-t1 transition-colors" />
          <ThemeToggle />
          <ReportBugLink variant="default" />

          <AuthButtons showLogoutOnly={true} />
        </div>
      </div>
    </nav>
  );
};

export default DatasetGalleryHeader;
