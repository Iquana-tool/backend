import React from 'react';
import { ChevronRight, ChevronDown, Plus, Trash2, Edit2, X, CornerUpRight } from 'lucide-react';
import { hasChildren } from '../../../utils/labelHierarchy';

// Vertical centre of a collapsed row (py-1.5 + a 22px control), which is where
// the connector elbows meet the spine. Kept as a constant so the two spans that
// draw a connector cannot drift apart.
const ROW_MID_Y = '17px';

/**
 * Shared component for rendering label hierarchy
 * Used by both EditableLabels and CreateLabelsModal to ensure consistency
 */
const LabelHierarchyRenderer = ({
  labels,
  expandedLabels,
  onToggleExpanded,
  onAddLabel,
  onEditLabel,
  onMoveLabel,
  onDeleteLabel,
  mode = 'editable', // 'editable' | 'creation'
  depth = 0,
  getLabelColor,
  renderLabelInput,
  renderLabelActions,
  // Optional per-row extras, used by the editable tree for drag-to-reparent:
  // { className, badge, ...domProps } spread onto the row.
  getRowProps
}) => {
  if (!labels || labels.length === 0) {
    return null;
  }

  return labels.map((label) => {
    const hasChildLabels = hasChildren(label);
    const isExpanded = expandedLabels.has(label.id);
    const marginLeft = depth * 20;
    const isCreation = mode === 'creation';

    // Default color function if not provided
    const getColor = getLabelColor || ((label) => {
      if (label.color) return label.color;
      return `hsl(${(label.id * 137.508) % 360}, 70%, 50%)`;
    });

    // Default label input rendering
    const renderInput = renderLabelInput || ((label, depth) => (
      <span className="font-medium text-t1">{label.name}</span>
    ));

    // Default actions rendering. Muted until hovered so a deep tree does not read
    // as a wall of coloured icons, but never hidden — see the note on the row.
    const renderActions = renderLabelActions || ((label) => (
      <div className="flex items-center space-x-1">
        {onAddLabel && (
          <button
            onClick={() => onAddLabel(label)}
            className="p-1 text-t3 hover:text-ac hover:bg-acS rounded transition-colors"
            title={`Add a part of "${label.name}"`}
          >
            <Plus size={14} />
          </button>
        )}
        {onEditLabel && (
          <button
            onClick={() => onEditLabel(label)}
            className="p-1 text-t3 hover:text-ac hover:bg-acS rounded transition-colors"
            title={`Rename "${label.name}"`}
          >
            <Edit2 size={14} />
          </button>
        )}
        {onMoveLabel && (
          <button
            onClick={() => onMoveLabel(label)}
            className="p-1 text-t3 hover:text-ac hover:bg-acS rounded transition-colors"
            title={`Move "${label.name}" — make it a part of something else`}
          >
            <CornerUpRight size={14} />
          </button>
        )}
        {onDeleteLabel && (
          <button
            onClick={() => onDeleteLabel(label)}
            className="p-1 text-t3 hover:text-err hover:bg-errBg rounded transition-colors"
            title={`Delete "${label.name}"`}
          >
            {isCreation ? <X size={14} /> : <Trash2 size={14} />}
          </button>
        )}
      </div>
    ));

    // Editable mode: an indented tree drawn with real elbow connectors, so the
    // shape of the hierarchy is visible rather than merely implied by indentation.
    if (!isCreation) {
      const childCount = hasChildLabels ? label.children.length : 0;
      const { className: rowClassName = '', badge: rowBadge = null, ...rowDomProps } =
        (getRowProps && getRowProps(label)) || {};

      return (
        <div key={label.id}>
          <div
            className={`group flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-hv ${rowClassName}`}
            {...rowDomProps}
          >
            {/* Expand/Collapse Button (or aligning spacer) */}
            {hasChildLabels && onToggleExpanded ? (
              <button
                onClick={() => onToggleExpanded(label.id)}
                className="p-0.5 text-t3 hover:text-t1 rounded shrink-0"
                aria-expanded={isExpanded}
                title={isExpanded ? 'Collapse' : 'Expand'}
              >
                {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
            ) : (
              <span className="w-[22px] shrink-0" aria-hidden="true" />
            )}

            {/* Color Dot — the shared palette, so a label looks the same here as
                it does on the canvas and in review. */}
            <div
              className="w-3.5 h-3.5 rounded-full shrink-0 ring-2 ring-p1 shadow-sm"
              style={{ backgroundColor: getColor(label) }}
            />

            {/* Label Name */}
            <div className="min-w-0 flex items-center">
              {renderInput(label, depth)}
            </div>

            {/* Part count — plain text; a pill on every branch row is noise */}
            {hasChildLabels && (
              <span className="shrink-0 text-xs text-t3">
                {childCount} part{childCount !== 1 ? 's' : ''}
              </span>
            )}

            <div className="flex-1" />

            {/* What this row would cost as a drop target, while a drag is in flight */}
            {rowBadge}

            {/* Actions stay in the layout rather than appearing on hover: users
                who never hover a row never discover that nesting exists. */}
            {(onAddLabel || onEditLabel || onMoveLabel || onDeleteLabel) && (
              <div className="shrink-0">
                {renderActions(label)}
              </div>
            )}
          </div>

          {/* Children — each hung off the parent with a vertical spine and an
              elbow, the spine stopping at the last child. */}
          {hasChildLabels && isExpanded && label.children && (
            <div className="ml-[18px] pl-5">
              {label.children.map((child, index) => {
                const isLastChild = index === label.children.length - 1;
                return (
                  <div key={child.id} className="relative">
                    <span
                      aria-hidden="true"
                      className="absolute left-0 top-0 w-px bg-ln"
                      style={{ height: isLastChild ? ROW_MID_Y : '100%' }}
                    />
                    <span
                      aria-hidden="true"
                      className="absolute left-0 h-px w-4 bg-ln"
                      style={{ top: ROW_MID_Y }}
                    />
                    <LabelHierarchyRenderer
                      labels={[child]}
                      expandedLabels={expandedLabels}
                      onToggleExpanded={onToggleExpanded}
                      onAddLabel={onAddLabel}
                      onEditLabel={onEditLabel}
                      onMoveLabel={onMoveLabel}
                      onDeleteLabel={onDeleteLabel}
                      mode={mode}
                      depth={depth + 1}
                      getLabelColor={getLabelColor}
                      renderLabelInput={renderLabelInput}
                      renderLabelActions={renderLabelActions}
                      getRowProps={getRowProps}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      );
    }

    return (
      <div key={label.id} className={mode === 'creation' ? 'mb-3' : 'mb-2'}>
        {/* Label Container */}
        <div
          className={
            mode === 'creation'
              ? 'border rounded-lg'
              : 'border border-ln rounded-lg p-3 bg-p1'
          }
          style={mode === 'creation' ? {} : { marginLeft: `${marginLeft}px` }}
        >
          <div className={mode === 'creation' ? 'flex items-center p-3' : 'flex items-center justify-between'}>
            <div className="flex items-center flex-1">
              {/* Expand/Collapse Button */}
              {hasChildLabels && onToggleExpanded && (
                <button
                  onClick={() => onToggleExpanded(label.id)}
                  className={`p-1 mr-2 ${mode === 'creation' ? 'text-t3 hover:text-t1' : 'text-t2 hover:bg-hv rounded'}`}
                  title={isExpanded ? 'Collapse' : 'Expand'}
                >
                  {isExpanded ? (
                    <ChevronDown size={mode === 'creation' ? 16 : 16} />
                  ) : (
                    <ChevronRight size={mode === 'creation' ? 16 : 16} />
                  )}
                </button>
              )}

              {/* Color Dot */}
              <div
                className={`${mode === 'creation' ? 'w-6 h-6' : 'w-4 h-4'} rounded-full mr-3 flex-shrink-0`}
                style={{
                  backgroundColor: getColor(label),
                  ...(mode === 'creation' && { marginLeft: `${marginLeft}px` })
                }}
              />

              {/* Label Name/Input */}
              {renderInput(label, depth)}

              {/* Part Count Badge */}
              {hasChildLabels && (
                <span className={`ml-2 text-xs text-ac bg-acS px-2 py-1 rounded-full ${mode === 'creation' ? 'mr-2' : ''}`}>
                  {label.children.length} part{label.children.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>

            {/* Actions */}
            {(onAddLabel || onEditLabel || onMoveLabel || onDeleteLabel) && (
              <div className={mode === 'creation' ? '' : 'flex items-center space-x-1'}>
                {renderActions(label)}
              </div>
            )}
          </div>

          {/* Children Labels */}
          {hasChildLabels && isExpanded && label.children && (
            <div className={mode === 'creation' ? 'border-t bg-well p-3' : 'mt-2'}>
              <LabelHierarchyRenderer
                labels={label.children}
                expandedLabels={expandedLabels}
                onToggleExpanded={onToggleExpanded}
                onAddLabel={onAddLabel}
                onEditLabel={onEditLabel}
                onMoveLabel={onMoveLabel}
                onDeleteLabel={onDeleteLabel}
                mode={mode}
                depth={depth + 1}
                getLabelColor={getLabelColor}
                renderLabelInput={renderLabelInput}
                renderLabelActions={renderLabelActions}
              />
            </div>
          )}
        </div>
      </div>
    );
  });
};

export default LabelHierarchyRenderer;

