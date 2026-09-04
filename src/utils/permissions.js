/**
 * Permission and role vocabulary, mirroring `app/schemas/permissions.py`.
 *
 * These are string constants only — the authoritative role -> permission matrix
 * lives on the backend and is fetched from `GET /datasets/roles/catalog`. Nothing
 * here decides what a role grants; the UI asks what *this* caller may do via the
 * `my_permissions` list the backend attaches to each dataset.
 *
 * Gating the UI is a convenience, not a security boundary: every action is
 * re-checked server-side.
 */

/** Permission strings, as returned in `dataset.my_permissions`. */
export const Permission = {
    // Dataset lifecycle
    DATASET_READ: 'dataset.read',
    DATASET_UPDATE: 'dataset.update',
    DATASET_DELETE: 'dataset.delete',
    DATASET_TRANSFER_OWNERSHIP: 'dataset.transfer_ownership',

    // Membership
    MEMBER_LIST: 'member.list',
    MEMBER_GRANT: 'member.grant',
    MEMBER_REVOKE: 'member.revoke',
    INVITE_CREATE: 'invite.create',
    INVITE_REVOKE: 'invite.revoke',

    // Image data
    IMAGE_READ: 'image.read',
    IMAGE_UPLOAD: 'image.upload',
    IMAGE_DELETE: 'image.delete',
    IMAGE_METADATA_WRITE: 'image.metadata_write',
    PIXEL_SCALE_SET: 'pixel_scale.set',
    CALIBRATION_SET: 'calibration.set',

    // Label space
    LABEL_READ: 'label.read',
    LABEL_MANAGE: 'label.manage',

    // Annotation
    ANNOTATION_READ: 'annotation.read',
    ANNOTATION_CREATE: 'annotation.create',
    ANNOTATION_EDIT_OWN: 'annotation.edit_own',
    ANNOTATION_DELETE_OWN: 'annotation.delete_own',
    ANNOTATION_EDIT_ANY: 'annotation.edit_any',
    MASK_SUBMIT: 'mask.submit',
    MASK_REOPEN: 'mask.reopen',
    MASK_DELETE: 'mask.delete',

    // Review
    REVIEW_APPROVE: 'review.approve',
    REVIEW_REJECT: 'review.reject',
    REVIEW_REVOKE: 'review.revoke',
    REVIEW_PURGE_UNREVIEWED: 'review.purge_unreviewed',

    // AI assistance
    AI_INTERACTIVE: 'ai.interactive',
    AI_BATCH_INFER: 'ai.batch_infer',
    AI_TRAIN: 'ai.train',

    // Export
    EXPORT_ANNOTATIONS: 'export.annotations',
    EXPORT_IMAGES: 'export.images',
    EXPORT_QUANTIFICATION: 'export.quantification',

    // Global (answered by the account's global role, not by dataset membership)
    DATASET_CREATE: 'dataset.create',
    USER_MANAGE: 'user.manage',
    USER_SET_GLOBAL_ROLE: 'user.set_global_role',
    SYSTEM_MANAGE_MODELS: 'system.manage_models',
    SYSTEM_MANAGE_SETTINGS: 'system.manage_settings',
};

/** Per-dataset roles, least to most privileged. */
export const DatasetRole = {
    VIEWER: 'viewer',
    ANNOTATOR: 'annotator',
    REVIEWER: 'reviewer',
    CURATOR: 'curator',
    OWNER: 'owner',
};

/** Platform-level roles. */
export const GlobalRole = {
    ADMIN: 'admin',
    MEMBER: 'member',
    GUEST: 'guest',
};

/** Privilege ranking, for comparisons and for ordering member lists. */
export const DATASET_ROLE_ORDER = {
    [DatasetRole.VIEWER]: 0,
    [DatasetRole.ANNOTATOR]: 1,
    [DatasetRole.REVIEWER]: 2,
    [DatasetRole.CURATOR]: 3,
    [DatasetRole.OWNER]: 4,
};

/** Short, user-facing description of what each role is for. */
export const DATASET_ROLE_LABELS = {
    [DatasetRole.VIEWER]: {
        label: 'Viewer',
        description: 'Can look at images and annotations. No downloads, no edits.',
    },
    [DatasetRole.ANNOTATOR]: {
        label: 'Annotator',
        description: 'Can create and edit their own annotations and submit them for review.',
    },
    [DatasetRole.REVIEWER]: {
        label: 'Reviewer',
        description: 'Can review anyone\'s annotations, approve or reject them, and export annotations.',
    },
    [DatasetRole.CURATOR]: {
        label: 'Curator',
        description: 'Runs the dataset: uploads images, manages labels, trains models, exports everything.',
    },
    [DatasetRole.OWNER]: {
        label: 'Owner',
        description: 'Full control, including deleting the dataset and managing who has access.',
    },
};

export const GLOBAL_ROLE_LABELS = {
    [GlobalRole.ADMIN]: {
        label: 'Admin',
        description: 'Manages accounts and can access every dataset.',
    },
    [GlobalRole.MEMBER]: {
        label: 'Member',
        description: 'Can create their own datasets and be invited to others.',
    },
    [GlobalRole.GUEST]: {
        label: 'Guest',
        description: 'Cannot create datasets. Works only in datasets they were invited to.',
    },
};

/** Roles that can be assigned to a collaborator. Ownership needs an explicit transfer. */
export const ASSIGNABLE_DATASET_ROLES = [
    DatasetRole.VIEWER,
    DatasetRole.ANNOTATOR,
    DatasetRole.REVIEWER,
    DatasetRole.CURATOR,
];

/** Tailwind classes for the role badge, escalating in visual weight. */
export const DATASET_ROLE_BADGE_CLASSES = {
    [DatasetRole.VIEWER]: 'bg-well text-t2 border-ln',
    [DatasetRole.ANNOTATOR]: 'bg-acS text-ac border-acLn',
    [DatasetRole.REVIEWER]: 'bg-acS text-ac border-acLn',
    [DatasetRole.CURATOR]: 'bg-warnBg text-warn border-warnLn',
    [DatasetRole.OWNER]: 'bg-acS text-ac border-acLn',
};

/**
 * Plain-language name for each permission, grouped for the reference table.
 *
 * Only presentation — the role catalog itself comes from the backend, so a
 * permission missing from this map still shows up in the table (falling back to
 * its raw key) rather than disappearing from it.
 */
export const PERMISSION_GROUPS = [
    {
        id: 'dataset',
        title: 'Dataset',
        permissions: [
            [Permission.DATASET_READ, 'See the dataset and its progress'],
            [Permission.DATASET_UPDATE, 'Edit the name, description and review policy'],
            [Permission.DATASET_DELETE, 'Delete the dataset and everything in it'],
            [Permission.DATASET_TRANSFER_OWNERSHIP, 'Hand ownership to someone else'],
        ],
    },
    {
        id: 'members',
        title: 'People',
        permissions: [
            [Permission.MEMBER_LIST, 'See who has access'],
            [Permission.MEMBER_GRANT, 'Add collaborators and change their roles'],
            [Permission.MEMBER_REVOKE, 'Remove someone’s access'],
            [Permission.INVITE_CREATE, 'Create invite links'],
            [Permission.INVITE_REVOKE, 'Revoke invite links'],
        ],
    },
    {
        id: 'images',
        title: 'Images',
        permissions: [
            [Permission.IMAGE_READ, 'View images and thumbnails'],
            [Permission.IMAGE_UPLOAD, 'Upload new images'],
            [Permission.IMAGE_DELETE, 'Delete images'],
            [Permission.IMAGE_METADATA_WRITE, 'Edit image metadata (the subgroups the dataset is compared across)'],
            [Permission.PIXEL_SCALE_SET, 'Set the pixel scale (rescales every measurement)'],
            [Permission.CALIBRATION_SET, 'Set colour and intensity calibration'],
        ],
    },
    {
        id: 'labels',
        title: 'Labels',
        permissions: [
            [Permission.LABEL_READ, 'See the label hierarchy'],
            [Permission.LABEL_MANAGE, 'Create, edit and delete labels (deleting removes them from annotations)'],
        ],
    },
    {
        id: 'annotation',
        title: 'Annotation',
        permissions: [
            [Permission.ANNOTATION_READ, 'View annotations'],
            [Permission.ANNOTATION_CREATE, 'Draw new annotations'],
            [Permission.ANNOTATION_EDIT_OWN, 'Edit your own annotations'],
            [Permission.ANNOTATION_DELETE_OWN, 'Delete your own annotations'],
            [Permission.ANNOTATION_EDIT_ANY, 'Edit and delete anyone’s annotations'],
            [Permission.MASK_SUBMIT, 'Mark an image as fully annotated (submit for review)'],
            [Permission.MASK_REOPEN, 'Reopen a submitted image for editing'],
            [Permission.MASK_DELETE, 'Delete an image’s whole annotation set'],
        ],
    },
    {
        id: 'review',
        title: 'Review',
        permissions: [
            [Permission.REVIEW_APPROVE, 'Approve annotations'],
            [Permission.REVIEW_REJECT, 'Send work back with a reason'],
            [Permission.REVIEW_REVOKE, 'Withdraw approvals given by others'],
            [Permission.REVIEW_PURGE_UNREVIEWED, 'Bulk-delete every unapproved annotation on an image'],
        ],
    },
    {
        id: 'ai',
        title: 'AI assistance',
        permissions: [
            [Permission.AI_INTERACTIVE, 'Use click-to-segment while annotating'],
            [Permission.AI_BATCH_INFER, 'Run a model across the dataset'],
            [Permission.AI_TRAIN, 'Train a model on the dataset'],
        ],
    },
    {
        id: 'export',
        title: 'Download',
        permissions: [
            [Permission.EXPORT_ANNOTATIONS, 'Download annotations (COCO JSON)'],
            [Permission.EXPORT_IMAGES, 'Download the raw images'],
            [Permission.EXPORT_QUANTIFICATION, 'Download measurements and statistics'],
        ],
    },
];

/** Flat permission -> description lookup, derived from the groups above. */
export const PERMISSION_LABELS = Object.fromEntries(
    PERMISSION_GROUPS.flatMap((group) => group.permissions)
);

/**
 * Whether a role ranks at or above a minimum.
 * @param {string} role
 * @param {string} minimum
 * @returns {boolean}
 */
export const isAtLeast = (role, minimum) =>
    (DATASET_ROLE_ORDER[role] ?? -1) >= (DATASET_ROLE_ORDER[minimum] ?? Infinity);

/**
 * Whether a permission list contains a permission.
 * @param {string[]|undefined} permissions - Typically `dataset.my_permissions`.
 * @param {string} permission
 * @returns {boolean}
 */
export const hasPermission = (permissions, permission) =>
    Array.isArray(permissions) && permissions.includes(permission);
