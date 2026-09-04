import React, { useState } from 'react';
import { X, Plus, Clock, Tag } from 'lucide-react';
import * as api from '../../api';
import { getNextAvailableColor } from '../../utils/labelColors';
import { 
  flattenHierarchy 
} from '../../utils/labelHierarchy';
import { useLabelHierarchy } from '../../hooks/useLabelHierarchy';
import LabelHierarchyRenderer from './shared/LabelHierarchyRenderer';

const CreateLabelsModal = ({ isOpen, onClose, dataset, onLabelsCreated }) => {
  const {
    labelHierarchy,
    expandedLabels,
    setLabelHierarchy,
    toggleExpanded,
    expandLabel,
  } = useLabelHierarchy([]);
  
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState(null);
  const [showLabelCreation, setShowLabelCreation] = useState(false);
  const [labelIdCounter, setLabelIdCounter] = useState(1);

  const getNextColor = () => {
    const flatLabels = flattenHierarchy(labelHierarchy);
    const usedColors = flatLabels.map(label => label.color);
    return getNextAvailableColor(usedColors);
  };

  const getNextId = () => {
    const nextId = labelIdCounter;
    setLabelIdCounter(prev => prev + 1);
    return nextId;
  };


  const handleAddLabel = (parentLabel = null) => {
    const newLabel = {
      id: getNextId(),
      name: '',
      color: getNextColor(),
      parent_id: parentLabel ? parentLabel.id : null,
      children: []
    };
    
    if (parentLabel) {
      // Add to parent's children
      const updateLabelHierarchy = (labels) => {
        return labels.map(label => {
          if (label.id === parentLabel.id) {
            return { ...label, children: [...label.children, newLabel] };
          } else if (label.children) {
            return { ...label, children: updateLabelHierarchy(label.children) };
          }
          return label;
        });
      };
      setLabelHierarchy(updateLabelHierarchy(labelHierarchy));
      // Auto-expand the parent
      expandLabel(parentLabel.id);
    } else {
      // Add as root label
      setLabelHierarchy([...labelHierarchy, newLabel]);
    }
  };



  const handleLabelNameChange = (id, name) => {
    const updateLabelName = (labels) => {
      return labels.map(label => {
        if (label.id === id) {
          return { ...label, name };
        } else if (label.children) {
          return { ...label, children: updateLabelName(label.children) };
        }
        return label;
      });
    };
    setLabelHierarchy(updateLabelName(labelHierarchy));
  };

  const handleDeleteLabel = (id) => {
    const deleteLabel = (labels) => {
      return labels
        .filter(label => label.id !== id)
        .map(label => ({
          ...label,
          children: label.children ? deleteLabel(label.children) : []
        }));
    };
    setLabelHierarchy(deleteLabel(labelHierarchy));
  };

  const handleCreateLabels = async () => {
    // Validate that all labels have names
    const flatLabels = flattenHierarchy(labelHierarchy);
    const hasEmptyLabels = flatLabels.some(label => !label.name.trim());

    if (hasEmptyLabels) {
      setError('Please provide names for all labels');
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      // Create labels recursively starting from root
      const createLabelsRecursive = async (labels, parentApiId = null) => {
        for (const label of labels) {
          if (label.name.trim()) {
            const result = await api.createLabel(
              { name: label.name.trim(), parent_id: parentApiId },
              dataset.id
            );
            
            if (result.success) {
              // Recursively create children
              if (label.children && label.children.length > 0) {
                await createLabelsRecursive(label.children, result.class_id);
              }
            }
          }
        }
      };
      
      await createLabelsRecursive(labelHierarchy);
      
      onLabelsCreated();
      onClose();
      
    } catch (err) {
      console.error('Error creating labels:', err);
      setError('Failed to create labels. Please try again.');
    } finally {
      setIsCreating(false);
    }
  };

  const handleCreateNow = () => {
    setShowLabelCreation(true);
  };

  const handleCreateLater = () => {
    onLabelsCreated();
    onClose();
  };

  const handleBackToChoice = () => {
    setShowLabelCreation(false);
    setLabelHierarchy([]);
    setError(null);
  };

  // Get label color
  const getLabelColor = (label) => {
    return label.color || getNextAvailableColor([]);
  };

  // Render label input for creation mode
  const renderLabelInput = (label, depth) => (
    <input
      type="text"
      value={label.name}
      onChange={(e) => handleLabelNameChange(label.id, e.target.value)}
      placeholder={depth > 0 ? "Part name" : "Label name"}
      className="flex-1 px-2 py-1 border-0 bg-transparent focus:outline-none focus:ring-0"
    />
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-scrim flex items-center justify-center z-50 p-4">
      <div className="bg-p1 rounded-xl shadow-xl w-full max-w-md max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-p2 border-b border-ln text-t1 p-6 flex-shrink-0">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold">Labels</h2>
            <button
              onClick={onClose}
              className="text-t3 hover:text-t1 transition-colors duration-150"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {!showLabelCreation ? (
            /* Initial Choice Screen */
            <div className="p-6">
              <p className="text-t2 text-sm mb-6">
                Labels help you identify and categorize objects in your images. You can create them now to start annotating immediately, or add them later when you're ready.
              </p>

              <div className="space-y-4">
                {/* Create Now Option */}
                <div className="border rounded-lg p-4 hover:border-acLn transition-colors cursor-pointer" onClick={handleCreateNow}>
                  <div className="flex items-start">
                    <div className="flex-shrink-0 w-10 h-10 bg-acS rounded-lg flex items-center justify-center mr-4">
                      <Tag className="w-5 h-5 text-ac" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-t1 mb-1">Create labels now</h3>
                      <p className="text-sm text-t2">Set up your labels immediately and start annotating your images right away.</p>
                    </div>
                  </div>
                </div>

                {/* Create Later Option */}
                <div className="border rounded-lg p-4 hover:border-acLn transition-colors cursor-pointer" onClick={handleCreateLater}>
                  <div className="flex items-start">
                    <div className="flex-shrink-0 w-10 h-10 bg-warnBg rounded-lg flex items-center justify-center mr-4">
                      <Clock className="w-5 h-5 text-warn" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-t1 mb-1">Create labels later</h3>
                      <p className="text-sm text-t2">Skip label creation for now. You can add labels anytime from the dataset overview.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Label Creation Screen */
            <div className="p-6">
              <div className="flex items-center mb-4">
                <button
                  onClick={handleBackToChoice}
                  className="text-ac hover:text-ac text-sm mr-2"
                >
                  ← Back
                </button>
              </div>

              <p className="text-t2 text-sm mb-6">
                Please provide the labels for this dataset. The labels represent the objects you want to identify on the images. Labels can be nested indicating they only occur inside another structure.
              </p>

              {/* Labels List */}
              <div className="space-y-3">
                {labelHierarchy.length === 0 ? (
                  <div className="text-center py-8 text-t3">
                    <p className="text-sm">No labels created yet</p>
                    <p className="text-xs mt-1">Click "Add new label" to get started</p>
                  </div>
                ) : (
                  <LabelHierarchyRenderer
                    labels={labelHierarchy}
                    expandedLabels={expandedLabels}
                    onToggleExpanded={toggleExpanded}
                    onAddLabel={handleAddLabel}
                    onDeleteLabel={(label) => handleDeleteLabel(label.id)}
                    mode="creation"
                    getLabelColor={getLabelColor}
                    renderLabelInput={renderLabelInput}
                  />
                )}
              </div>

              {/* Add New Label Button */}
              <button
                onClick={() => handleAddLabel()}
                className="w-full mt-4 p-3 border-2 border-dashed border-ln2 rounded-lg text-t3 hover:border-ln2 hover:text-t1 transition-colors flex items-center justify-center"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add new label
              </button>

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-3 bg-errBg border border-errLn rounded-lg">
                  <p className="text-sm text-err">{error}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        {showLabelCreation && (
          <div className="flex-shrink-0 p-6 bg-well border-t border-ln">
            <div className="flex space-x-3">
              <button
                onClick={onClose}
                disabled={isCreating}
                className="flex-1 bg-ln2 text-t2 py-3 px-6 rounded-lg hover:bg-t3 disabled:opacity-50 transition-colors font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateLabels}
                disabled={isCreating || flattenHierarchy(labelHierarchy).every(label => !label.name.trim())}
                className="flex-1 bg-accent text-onAccent py-3 px-6 rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {isCreating ? 'Creating...' : 'Create Labels'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CreateLabelsModal; 