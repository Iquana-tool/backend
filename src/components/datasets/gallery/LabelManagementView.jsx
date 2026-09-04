import React, { useState } from "react";
import { Sparkles, ListTree } from "lucide-react";
import * as api from "../../../api";
import EditableLabels from "./EditableLabels";
import DescribeLabelSpaceModal from "./DescribeLabelSpaceModal";
import { extractLabelsFromResponse } from "../../../utils/labelHierarchy";

const LabelManagementView = ({ dataset, labels, onLabelsUpdated }) => {
  const [showDescribeModal, setShowDescribeModal] = useState(false);

  // Re-fetch labels after the assistant applies a generated hierarchy so the
  // editor below reflects the new labels.
  const refreshLabels = async () => {
    if (!dataset?.id || !onLabelsUpdated) return;
    const labelsData = await api.fetchLabels(dataset.id);
    onLabelsUpdated(extractLabelsFromResponse(labelsData));
  };

  return (
    <div className="h-full flex flex-col bg-p1">
      {/* Header */}
      <div className="p-3 sm:p-4 border-b border-ln bg-p1 sticky top-0 z-10">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-lg sm:text-xl font-bold text-t1">
            Label Management
          </h2>

          <button
            onClick={() => setShowDescribeModal(true)}
            className="inline-flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 text-sm font-medium text-onAccent bg-accent rounded-6 shadow-primary hover:brightness-110 transition-colors"
            title="Describe your label space and build the hierarchy automatically"
          >
            <Sparkles className="w-4 h-4" />
            <span className="hidden sm:inline">Describe your label space</span>
            <span className="sm:hidden">Describe</span>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6">
        <div className="max-w-4xl mx-auto">
          {/* Intro: explain the hierarchical label space */}
          <div className="mb-6 bg-acS border border-acLn rounded-12 p-4 sm:p-5">
            <div className="flex items-start gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-p1 text-ac shadow-sm shrink-0">
                <ListTree className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <h3 className="text-base font-semibold text-t1">
                  Nesting a label means "part of"
                </h3>
                <p className="text-sm text-t2 mt-1 leading-relaxed">
                  A label nested under another one is <span className="font-medium text-t1">a part
                  of</span> it — <span className="font-medium text-t1">Nucleus</span> under{' '}
                  <span className="font-medium text-t1">Cell</span> reads <em>a nucleus is part of a
                  cell</em>. Say the sentence out loud before you nest; if it does not work, the
                  label belongs at the top level.
                </p>
                <p className="text-sm text-t2 mt-2 leading-relaxed">
                  This is what lets you annotate one object inside another:{' '}
                  <span className="font-medium text-t1">an object can only carry a label that is a
                  direct part of the label on the object containing it</span>. Sibling labels should
                  be mutually exclusive — an object is one of them, never two.
                </p>

                {/* The confusion the tree cannot express, said plainly */}
                <p className="mt-3 text-xs text-t3 leading-relaxed">
                  Nesting is not for subtypes. <span className="font-medium text-t2">Acropora</span>{' '}
                  is a <em>kind of</em> coral, not a part of one, so it stays at the top level next
                  to <span className="font-medium text-t2">Coral</span> — "is a kind of" is not
                  something the label space can record yet.
                </p>

                {/* Inline hint toward the assistant */}
                <button
                  onClick={() => setShowDescribeModal(true)}
                  className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-ac hover:text-ac"
                >
                  <Sparkles className="w-4 h-4" />
                  Not sure where to start? Describe your label space and let us draft it for you.
                </button>
              </div>
            </div>
          </div>

          <div className="bg-p1 border border-ln rounded-lg p-4 sm:p-6">
            <EditableLabels
              dataset={dataset}
              labels={labels}
              onLabelsUpdated={onLabelsUpdated}
            />
          </div>
        </div>
      </div>

      {/* Describe-your-label-space assistant */}
      <DescribeLabelSpaceModal
        isOpen={showDescribeModal}
        onClose={() => setShowDescribeModal(false)}
        dataset={dataset}
        onLabelsUpdated={refreshLabels}
      />
    </div>
  );
};

export default LabelManagementView;
