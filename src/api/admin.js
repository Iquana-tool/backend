
/**
 * Create an account outright, without an invite or self-registration.
 *
 * An instance is closed by default and an invite only ever grants access to one
 * dataset, so this is how somebody is handed an account at all. The password is
 * chosen here and passed on out of band — iquana sends no mail, so there is
 * nowhere to deliver an activation link to.
 *
 * @param {{username: string, password: string, global_role?: string, is_active?: boolean}} account
 */
export const createUser = async (account) => {
    const response = await fetch(`${API_BASE_URL}/admin/users`, {
        method: "POST",
        headers: jsonHeaders(),
        body: JSON.stringify(account),
    });
    return handleApiError(response);
};

/**
 * Describe every editable instance setting, plus the AI service's live state.
 *
 * Secrets come back as `{is_set, hint}` and never as a value — the field renders
 * blank and is only ever written to.
 *
 * @returns {Promise<{success: boolean, groups: Array, settings: Array, ai_service: Object}>}
 */
export const fetchSettings = async () => {
    const response = await fetch(`${API_BASE_URL}/admin/settings`, {
        headers: getAuthHeaders(),
    });
    return handleApiError(response);
};

/**
 * Store overrides for the given settings.
 *
 * Sparse on purpose: send only what was edited, so two admins on different tabs
 * cannot clobber each other. An empty string for a secret means "leave it
 * alone", because the current value is never sent to the browser to begin with.
 *
 * @param {Object<string, string|null>} values - `{settingKey: newValue}`
 */
export const updateSettings = async (values) => {
    const response = await fetch(`${API_BASE_URL}/admin/settings`, {
        method: "PATCH",
        headers: jsonHeaders(),
        body: JSON.stringify({ values }),
    });
    return handleApiError(response);
};

/**
 * Drop one override, falling back to the value configured for the deployment.
 * @param {string} key
 */
export const clearSetting = async (key) => {
    const response = await fetch(
        `${API_BASE_URL}/admin/settings/${encodeURIComponent(key)}`,
        { method: "DELETE", headers: getAuthHeaders() }
    );
    return handleApiError(response);
};

/**
 * Re-send the settings the AI service consumes.
 *
 * That service holds them in memory, so restarting it drops whatever was pushed;
 * this is how an operator puts them back without editing a second `.env`.
 */
export const pushSettings = async () => {
    const response = await fetch(`${API_BASE_URL}/admin/settings/push`, {
        method: "POST",
        headers: getAuthHeaders(),
    });
    return handleApiError(response);
};
