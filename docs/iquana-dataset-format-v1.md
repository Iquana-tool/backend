# IQUANA Dataset Archive Format Specification (v1)

> **Normative Specification for Issue #94**  
> **Status:** Draft / Phase 1 technical implementation complete; maintainer contract gate pending  
> **Target Format:** Ordinary `.zip` archive containing native images, a COCO-oriented `annotations.json`, and an optional `config.json`.

---

## 1. Overview and Objectives

The IQUANA dataset archive format enables lossless, self-contained migration and backup of image datasets between IQUANA instances. Unlike standard COCO export projections, this format preserves rich multi-level contour hierarchies, masks, completion statuses, rejection histories, typed dataset metadata schemas and image metadata values, image calibrations, and (optionally) portable dataset configuration including model routing and quantification profile definitions.

### Key Principles

1. **Standard ZIP Packaging:** Uses standard ZIP files (DEFLATE / stored) with standard Python `zipfile` compatibility. No custom file extensions (not `.iquana`), no YAML, no proprietary archive containers.
2. **COCO-Oriented Foundation:** `annotations.json` retains standard COCO top-level arrays (`info`, `licenses`, `images`, `annotations`, `categories`) allowing third-party tools to inspect basic annotations while encapsulating IQUANA-native features within namespaced `iquana` objects.
3. **Lossless Images:** Image files are stored byte-for-byte with SHA-256 integrity verification.
4. **Canonical Coordinate System:** Normalized coordinates in `[0.0, 1.0]` (within `[-1.5, 1.5]` tolerance, `abs(value) <= 1.5`) are preserved as geometric ground truth in `iquana.geometry`. COCO pixel `segmentation`, `bbox`, and `area` are derived from the polygon and native image dimensions.
5. **No Serialized Quantifications:** Stored quantification tables (`ContourMetrics`) and legacy contour metric columns (`area_mm2`, `perimeter`, `circularity`, `diameter`) are **strictly excluded**. Geometry metrics are recomputed locally upon import using the native dual-write path, while dedicated scientific measurement exports continue to serve quantification data.
6. **Decoupled Configuration:** Dataset settings (review policy, calibration defaults, profile definitions, and model routing) live in an optional `config.json`. Label selectors in configuration resolve via dataset-unique label names rather than database primary keys.
7. **Safe Actor Provenance:** Source usernames and approvals are exported for provenance only. On import, the importing user becomes the local dataset owner, all account-linked actor foreign keys are cleared, and approval associations are dropped with an aggregated warning.

---

## 2. Archive Layout

```text
<dataset-name>.zip
├── annotations.json                          # Required: COCO core + IQUANA extensions
├── config.json                               # Optional: Portable dataset settings (default: excluded)
└── images/
    └── <archive-image-id>/<sanitized-name>   # Raw images isolated by archive ID
```

- Every member path uses forward slashes `/`.
- Member paths starting with `/`, drive letters, containing `..`, or pointing to symlinks or device nodes are strictly invalid and rejected.
- Each image is nested inside its ZIP-local positive integer image ID folder (`images/<archive-image-id>/<sanitized-basename>`) to prevent collisions between images with identical filenames.

---

## 3. `annotations.json` Specification

Top-level structure:

```json
{
  "format": "iquana",
  "format_version": 1,
  "info": {
    "description": "Coral Reef Survey 2026",
    "version": "1.0",
    "year": 2026,
    "date_created": "2026-09-14T20:00:00Z",
    "contributor": null,
    "url": null
  },
  "licenses": [],
  "images": [ ... ],
  "annotations": [ ... ],
  "categories": [ ... ],
  "iquana": {
    "dataset": { ... },
    "actors": [ ... ],
    "metadata_keys": [ ... ],
    "masks": [ ... ],
    "rejections": [ ... ],
    "counts": { ... },
    "files": [ ... ]
  }
}
```

### 3.1. Document Header & Info

| Field | Type | Description |
|---|---|---|
| `format` | `string` | Must be `"iquana"`. |
| `format_version` | `integer` | Must be `1`. |
| `info.description` | `string` | Dataset description or generated export title. |
| `info.version` | `string` | Document version (default `"1.0"`). |
| `info.year` | `integer` | Creation calendar year. |
| `info.date_created` | `string` | ISO-8601 UTC timestamp. |

### 3.2. `images[]`

Standard COCO image fields with nested IQUANA extension:

```json
{
  "id": 1,
  "file_name": "site_a_001.jpg",
  "width": 3840,
  "height": 2160,
  "iquana": {
    "archive_path": "images/1/site_a_001.jpg",
    "color_mode": "RGB",
    "scale_x": 0.05,
    "scale_y": 0.05,
    "unit": "mm",
    "description": "North transect quadrant 1",
    "metadata": {
      "site": "Reef Alpha",
      "water_depth": "12.5"
    },
    "calibrations": [
      {
        "kind": "scale",
        "source": "manual",
        "params": { "scale_x": 0.05, "scale_y": 0.05, "unit": "mm" },
        "created_by": "alice",
        "created_at": "2026-09-14T10:00:00Z",
        "updated_at": "2026-09-14T10:00:00Z"
      }
    ],
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "size_bytes": 2456102
  }
}
```

- `width` and `height` must reflect authoritative native full-resolution dimensions read from the image header.
- `metadata` stores raw string key-value pairs matching keys declared in `iquana.metadata_keys`.
- `calibrations` records parameter payloads, calibration kind, and provenance.

### 3.3. `categories[]`

Categories represent label classes and label hierarchies:

```json
{
  "id": 1,
  "name": "Hard Coral",
  "supercategory": "Substrate",
  "iquana": {
    "value": 2,
    "parent_id": null
  }
}
```

- `name`: Must be unique across the entire dataset (case-sensitive). Duplicate category names are rejected at export and import.
- `iquana.value`: Integer label mask value.
- `iquana.parent_id`: ID of parent category in `categories[]` (null for root).

### 3.4. `annotations[]`

Annotations represent non-temporary contour geometries:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "segmentation": [[192.0, 108.0, 384.0, 216.0, 199.68, 298.08]],
  "area": 17832.96,
  "bbox": [192.0, 108.0, 192.0, 190.08],
  "iscrowd": 0,
  "iquana": {
    "mask_id": 1,
    "parent_id": null,
    "geometry": {
      "x": [0.05, 0.10, 0.052],
      "y": [0.05, 0.10, 0.138]
    },
    "added_by": "SAM2",
    "confidence_score": 0.96,
    "created_at": "2026-09-14T11:00:00Z",
    "author_username": "bob",
    "reviewed_by": ["alice"]
  }
}
```

- `category_id`: May be `null` for unlabelled contours accepted by the user.
- `iquana.geometry`: Ground-truth normalized coordinates. Must satisfy `len(x) == len(y) >= 3` and `abs(val) <= 1.5`.
- `segmentation`, `area`, and `bbox`: Derived from polygon geometry and native image pixel dimensions. Import rejects values that do not match that canonical projection.
- `temporary=True` contours and their descendant subtrees are **omitted from export**, with count recorded in `counts.temporary_contours_omitted`.

### 3.5. `iquana` Top-Level Extension

- `dataset`:
  - `name`: Target display name (1–50 characters).
  - `description`: Optional description.
  - `dataset_type`: Must be `"image"`.
  - `created_by`: Source creator username (export provenance only).
- `actors`: List of `{ username, roles }` recording participants from the source instance.
- `metadata_keys`: List of `{ key, value_type, unit, options, description }`.
- `masks`: List of `{ id, image_id, fully_annotated }`.
- `rejections`: List of `{ id, mask_id, annotation_id, reason, note, created_by, created_at, resolved_at, resolved_by, resolution }`.
- `counts`: Summary counts of all entities.
- `files`: Manifest list of `{ image_id, path, sha256, size_bytes, width, height, color_mode }`.

---

## 4. `config.json` Specification

Optional standalone document exported when the user selects "Include internal configuration".

```json
{
  "format": "iquana",
  "format_version": 1,
  "dataset": {
    "require_independent_review": true
  },
  "calibration_defaults": [
    {
      "kind": "response",
      "defaults": { "strategy": "gray_wedge", "card": "kodak_q13", "fit_model": "linear" }
    }
  ],
  "quantification_profiles": [
    {
      "name": "Coral Ecology Profile",
      "is_default": true,
      "entries": [
        {
          "metric_key": "area",
          "params": {},
          "label_names": ["Hard Coral", "Soft Coral"]
        }
      ]
    }
  ],
  "model_routing": {
    "bindings": [
      {
        "task": "prompted-segmentation",
        "label_name": "Hard Coral",
        "model_registry_key": "sam2",
        "inputs": null
      }
    ]
  },
  "omitted_fields": [
    {
      "section": "model_routing.bindings",
      "field": "inputs.conditioning.query_contour_id",
      "task": "prompted-segmentation",
      "label_name": "Hard Coral",
      "reason": "query_contour_id is a local database ID and not portable"
    }
  ]
}
```

`quantification_profiles[].entries[].metric_key` must be one of the v1-supported
keys: `area`, `perimeter`, `circularity`, `max_diameter`, `mean_color_rgb`,
`mean_color_lab`, `mean_intensity`, `nn_distance`, `mean_knn_distance`, or
`n_children`. This fixed set matches the current runtime metric registry. A
syntactically valid but unsupported or future key makes the configuration invalid:
the importer rejects the archive before writes with a validation error (`422`).
It is never silently dropped or retained as an unavailable metric warning. Model
routing keys have a separate policy: unavailable model keys may be retained with
a curator warning.

### Decoupling Rules for Configuration

1. **No Database Primary Keys:** Label references in `quantification_profiles` and `model_routing` use dataset-unique label names (`label_name` or `label_names`), not category IDs or database primary keys.
2. **Local ID Stripping:** Runtime contour IDs (specifically `inputs.conditioning.query_contour_id`) are stripped before export and audited in `omitted_fields`.
3. **No Measurement Results:** Profile entries define metric keys, parameters, and scoped label names. No measured numeric results are stored.
4. **Resilience to Missing Models:** When an imported model registry key is not available locally, the binding is retained but flagged with an import warning to allow curator repair.
5. **Metric Compatibility:** Profile entries are limited to the fixed v1-supported metric-key set listed above. An unavailable, unknown, or future metric key is a validation error for the complete configuration/archive before writes; no profile entry is dropped or retained as an unavailable warning.

---

## 5. Portability Matrix

| State Category | Included in Archive? | Import Behavior |
|---|---|---|
| **Original image bytes** | Yes (`images/` directory) | Byte-for-byte SHA-256 match; preserved intact. |
| **Image metadata & calibrations** | Yes (`annotations.json`) | Re-linked to newly allocated image rows. |
| **Dataset metadata keys** | Yes (`annotations.json`) | Inserted before values; coerced with type validation. |
| **Label hierarchy** | Yes (`categories[]`) | New IDs allocated; parent relationships remapped. |
| **Contours & hierarchies** | Yes (`annotations[]`) | New IDs allocated; parents remapped; normalized coords preserved. |
| **Derived geometry metrics** | Excluded | Locally recomputed on import (`dual_write_geometry_metrics`). |
| **Stored `ContourMetrics`** | Excluded | Never exported; recomputed fresh on demand. |
| **Quantification profiles** | Yes (`config.json`) | Definitions with v1-supported metric keys are preserved; an unavailable, unknown, or future metric key rejects the complete config/archive before writes with a validation error. No profile entry is silently dropped. |
| **Approvals (`reviewed_by`)** | Exported as provenance | Dropped on import with an aggregated warning. |
| **Actor usernames** | Exported as provenance | Cleared on imported rows; importing caller owns dataset. |
| **Temporary contours** | Excluded | Dropped with count recorded in `counts`. |
| **Thumbnails & embeddings** | Excluded | Rebuilt fresh locally in staging before commit. |
| **Batch jobs & undo queues** | Excluded | Operational state discarded. |

---

## 6. Provisional Safety Limits and Constraints

> [!IMPORTANT]
> **Provisional Limits & Open Approval Gate:**
> All limits below are **provisional defaults** configured in `backend/config.py` and documented in `backend/env.example`. The approval gate remains open pending empirical evidence from representative user-study dataset distributions (e.g. gigapixel microscopy, high-depth image series).
>
> **Reverse Proxy / ASGI Protection Requirement:**
> `DATASET_ARCHIVE_MAX_COMPRESSED_BYTES` is a secondary application-level check on the staged archive size. Because HTTP multipart parsers (Starlette / `python-multipart`) spool incoming request bodies to temporary storage *before* route execution, application-level limit checks cannot protect server resources from upload exhaustion during transmission. Frontline request body limits **must** be enforced upstream at the reverse proxy (e.g., Nginx `client_max_body_size`) or ASGI server layer.

| Setting | Provisional Default | Purpose & Caveats |
|---|---|---|
| `DATASET_ARCHIVE_MAX_COMPRESSED_BYTES` | 10 GiB | Secondary application ceiling on staged archive size. Upstream reverse proxy must enforce body limits to prevent upload spool exhaustion. |
| `DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES` | 20 GiB | Provisional ceiling protecting against decompression bombs across all archive members. |
| `DATASET_ARCHIVE_MAX_MEMBERS` | 50,000 | Provisional ceiling preventing zip-bomb directory table floods. |
| `DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES` | 64 MiB | Provisional uncompressed ceiling for each control document (`annotations.json` and `config.json`); enforce from the central directory before reading JSON into memory. |
| `DATASET_ARCHIVE_MAX_MEMBER_BYTES` | 2 GiB | Provisional limit on maximum uncompressed size for any individual image or file member. |

### Validation Staging Pipeline

1. Direct seekable read of `UploadFile.file` via `ZipFile`.
2. Central directory validation: reject encrypted files, links, duplicate member paths, absolute paths, paths with `..`, and either control JSON document above its dedicated size limit before reading it.
3. Strict Pydantic schema validation of `annotations.json` and optional `config.json`, including the fixed v1 metric-key set.
4. Streaming hash validation of all image files against manifest `sha256` and header dimensions.
5. Thumbnail generation and semantic mask rasterization in staging prior to opening database transactions.
6. Single database transaction commit: insert dataset, images, labels, contours, and metadata.
7. Atomic filesystem move into `DATASETS_DIR` and `THUMBNAILS_DIR` immediately prior to transaction commit.
8. Error cleanup: handled errors immediately roll back DB transaction and purge staged directories.

---

## 7. Golden Fixture Hashes (v1)

Deterministic test fixtures generated with fixed `ZipInfo` timestamps (`2026-09-14 00:00:00`) and standard permissions:

| Fixture Variant | Members | Deterministic SHA-256 Digest |
|---|---|---|
| **With `config.json`** | `annotations.json`, `config.json`, `images/1/coral_survey.png`, `images/2/coral_survey.png` | `f29176398fd1d85c9e122f2fe6ed87726864e1664e91f89014b9edc52c9c0f3f` |
| **Without `config.json`** | `annotations.json`, `images/1/coral_survey.png`, `images/2/coral_survey.png` | `7f649288cc3790f27b7a67445498fa93c4424801a550bfee731ef7a88aa1f89f` |
