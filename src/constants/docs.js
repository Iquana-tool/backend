/**
 * Where the user-facing documentation lives.
 *
 * The docs are a MkDocs Material site built from the separate
 * `Iquana-tool.github.io` repo, not part of this bundle: they can be corrected
 * and extended without shipping a frontend build, which is why every in-app
 * "Documentation"/"Help" affordance points out here rather than at the old
 * in-app `/docs` page (deprecated — see `pages/DocumentationPage.jsx`).
 */
export const DOCS_URL = 'https://iquana-tool.github.io/docs/';

/**
 * A deep link into a docs page, e.g. `docsUrl(DOCS.annotateWithAi)`.
 *
 * The site is built with MkDocs' default pretty URLs, so a page is addressed by
 * its directory (`guides/annotate-with-ai/`), not by a `.md` file.
 */
export const docsUrl = (path = '') => `${DOCS_URL}${String(path).replace(/^\/+/, '')}`;

/**
 * The pages worth linking to from inside the app, named so a call site says
 * where it is sending the reader rather than spelling out a path that only the
 * docs repo can keep true.
 */
export const DOCS = {
  home: '',
  gettingStarted: 'getting-started/',
  firstRun: 'getting-started/first-run/',
  guides: 'guides/',
  createDataset: 'guides/create-a-dataset/',
  labelHierarchy: 'guides/build-a-label-hierarchy/',
  annotateWithAi: 'guides/annotate-with-ai/',
  drawAndRefine: 'guides/draw-and-refine/',
  calibrate: 'guides/calibrate/',
  batchInference: 'guides/batch-inference/',
  reviewAndCorrect: 'guides/review-and-correct/',
  quantifyAndExport: 'guides/quantify-and-export/',
  trainAModel: 'guides/train-a-model/',
  manageAccess: 'guides/manage-access/',
  roles: 'concepts/roles/',
  shortcuts: 'reference/shortcuts/',
  metrics: 'reference/metrics/',
  troubleshooting: 'operations/troubleshooting/',
};

/**
 * Open the docs in a new tab.
 *
 * For the places that are a button or a menu item rather than a link; anything
 * that can be an anchor should use `<DocsLink>` so middle-click and "copy link"
 * keep working.
 */
export const openDocs = (path = DOCS.home) =>
  window.open(docsUrl(path), '_blank', 'noopener,noreferrer');
