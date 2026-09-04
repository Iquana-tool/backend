
/**
 * Get the single saved model routing policy for a dataset.
 * Returns null when 204 No Content.
 */
export const getInferenceRoutingPolicy = async (datasetId) => {
    const params = new URLSearchParams({ dataset_id: String(datasetId) });
    const response = await fetch(`${API_BASE_URL}/inference/config?${params.toString()}`, {
        headers: getAuthHeaders(),
    });
    if (response.status === 204) {
        return null;
    }
    return handleApiError(response);
};

/**
 * Save or replace the single model routing policy for a dataset.
 * Expects { dataset_id: number, bindings: Array<ModelRoutingBinding> }
 */
export const updateInferenceRoutingPolicy = async (body) => {
    const response = await fetch(`${API_BASE_URL}/inference/config`, {
        method: "PUT",
        headers: jsonHeaders(),
        body: JSON.stringify({
            dataset_id: Number(body.dataset_id),
            bindings: body.bindings || [],
        }),
    });
    return handleApiError(response);
};

/**
 * Delete the model routing policy for a dataset.
 */
export const deleteInferenceRoutingPolicy = async (datasetId) => {
    const params = new URLSearchParams({ dataset_id: String(datasetId) });
    const response = await fetch(`${API_BASE_URL}/inference/config?${params.toString()}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
    });
    return handleApiError(response);
};

/**
 * Run one routed model step for a single image with patch semantics.
 */
export const suggestModelRoutingStep = async ({
    datasetId,
    imageId,
    maskId = null,
    labelId,
    task = "cross-image-suggestion",
}) => {
    const response = await fetch(`${API_BASE_URL}/inference/config/suggest`, {
        method: "POST",
        headers: jsonHeaders(),
        body: JSON.stringify({
            dataset_id: Number(datasetId),
            image_id: Number(imageId),
            mask_id: maskId != null ? Number(maskId) : null,
            label_id: Number(labelId),
            task: task || "cross-image-suggestion",
        }),
    });
    return handleApiError(response);
};
