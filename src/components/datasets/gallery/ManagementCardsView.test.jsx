import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect } from "vitest";
import ManagementCardsView from "./ManagementCardsView";
import { Permission } from "../../../utils/permissions";

const mockUsePermissions = vi.fn();
vi.mock("../../../hooks/usePermissions", () => ({
    usePermissions: (dataset) => mockUsePermissions(dataset),
}));

vi.mock("../../../stores/selectors", () => ({
    useGalleryStats: () => ({
        total: 10,
        overall: { finished: 5, in_progress: 2, not_started: 3 },
    }),
}));

vi.mock("../../../api/reviews", () => ({
    fetchReviewSummary: vi.fn().mockResolvedValue({ success: true, summary: { pending_instances: 0, reviewed_instances: 0 } }),
    fetchCorrectionSummary: vi.fn().mockResolvedValue({ success: true, summary: { open_rejections: 0 } }),
}));

describe("ManagementCardsView - Model Orchestration card", () => {
    const dataset = { id: 101, name: "Test Dataset" };

    it("renders Model Orchestration card when user has AI_BATCH_INFER permission and triggers click", () => {
        mockUsePermissions.mockReturnValue({
            can: (perm) => perm === Permission.AI_BATCH_INFER,
            canAny: (perms) => perms.includes(Permission.AI_BATCH_INFER),
            role: "owner",
        });

        const onModelOrchestrationClick = vi.fn();

        render(
            <ManagementCardsView
                dataset={dataset}
                onModelOrchestrationClick={onModelOrchestrationClick}
            />
        );

        const cardTitle = screen.getByText("Model Orchestration");
        expect(cardTitle).toBeInTheDocument();

        fireEvent.click(cardTitle);
        expect(onModelOrchestrationClick).toHaveBeenCalledTimes(1);
    });

    it("renders Model Orchestration card when user has AI_INTERACTIVE permission", () => {
        mockUsePermissions.mockReturnValue({
            can: (perm) => perm === Permission.AI_INTERACTIVE,
            canAny: (perms) => perms.includes(Permission.AI_INTERACTIVE),
            role: "annotator",
        });

        render(
            <ManagementCardsView
                dataset={dataset}
                onModelOrchestrationClick={vi.fn()}
            />
        );

        expect(screen.getByText("Model Orchestration")).toBeInTheDocument();
    });

    it("does not render Model Orchestration card when user lacks AI permissions", () => {
        mockUsePermissions.mockReturnValue({
            can: () => false,
            canAny: () => false,
            role: "viewer",
        });

        render(
            <ManagementCardsView
                dataset={dataset}
                onModelOrchestrationClick={vi.fn()}
            />
        );

        expect(screen.queryByText("Model Orchestration")).not.toBeInTheDocument();
    });
});
