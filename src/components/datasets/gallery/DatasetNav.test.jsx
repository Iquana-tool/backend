import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi, describe, it, expect, beforeEach } from "vitest";
import DatasetNav from "./DatasetNav";
import { Permission } from "../../../utils/permissions";

const mockNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
    const actual = await vi.importActual("react-router-dom");
    return { ...actual, useNavigate: () => mockNavigate };
});

const mockUsePermissions = vi.fn();
vi.mock("../../../hooks/usePermissions", () => ({
    usePermissions: (dataset) => mockUsePermissions(dataset),
}));

const dataset = { id: 7, name: "Reef survey" };

/** Grant everything, so a test can focus on the permission it cares about. */
const allowAll = () => ({
    can: () => true,
    canAny: () => true,
    canAll: () => true,
});

const renderNav = (initialPath = "/dataset/7/datamanagement") =>
    render(
        <MemoryRouter initialEntries={[initialPath]}>
            <DatasetNav dataset={dataset} datasetId={dataset.id} />
        </MemoryRouter>
    );

describe("DatasetNav", () => {
    beforeEach(() => {
        mockNavigate.mockClear();
        mockUsePermissions.mockReturnValue(allowAll());
    });

    it("navigates to a section picked from a group menu", () => {
        renderNav();

        fireEvent.click(screen.getByText("Data & Analysis"));
        fireEvent.click(screen.getByText("Label Management"));

        expect(mockNavigate).toHaveBeenCalledWith(
            "/dataset/7/datamanagement/labels",
            undefined
        );
    });

    it("hands the dataset to the model zoo in router state", () => {
        renderNav();

        fireEvent.click(screen.getByText("Artificial Intelligence"));
        fireEvent.click(screen.getByText("Model Zoo"));

        expect(mockNavigate).toHaveBeenCalledWith("/models", {
            state: { datasetId: 7 },
        });
    });

    it("marks the group holding the open page as current", () => {
        renderNav("/dataset/7/training");

        expect(screen.getByText("Artificial Intelligence").closest("button"))
            .toHaveClass("text-ac");
        expect(screen.getByText("Data & Analysis").closest("button"))
            .not.toHaveClass("text-ac");
    });

    it("marks the open item within a menu as current", () => {
        renderNav("/dataset/7/training");

        fireEvent.click(screen.getByText("Artificial Intelligence"));
        expect(screen.getByText("Model Training").closest("button")).toHaveAttribute(
            "aria-current",
            "page"
        );
        expect(screen.getByText("Batch Inference").closest("button"))
            .not.toHaveAttribute("aria-current");
    });

    it("leaves out destinations the role cannot reach", () => {
        mockUsePermissions.mockReturnValue({
            can: (permission) => permission === Permission.IMAGE_READ,
            canAny: (permissions) => permissions.includes(Permission.IMAGE_READ),
            canAll: () => false,
        });

        renderNav();

        // Data & Analysis survives on IMAGE_READ alone...
        fireEvent.click(screen.getByText("Data & Analysis"));
        expect(screen.getByText("Data Management")).toBeInTheDocument();
        expect(screen.queryByText("Label Management")).not.toBeInTheDocument();

        // ...while the groups with nothing left in them are not rendered at all.
        expect(screen.queryByText("Artificial Intelligence")).not.toBeInTheDocument();
        expect(screen.queryByText("Configuration")).not.toBeInTheDocument();
    });

    it("closes an open menu on Escape", () => {
        renderNav();

        fireEvent.click(screen.getByText("Data & Analysis"));
        expect(screen.getByText("Quantifications")).toBeInTheDocument();

        fireEvent.keyDown(document, { key: "Escape" });
        expect(screen.queryByText("Quantifications")).not.toBeInTheDocument();
    });
});
