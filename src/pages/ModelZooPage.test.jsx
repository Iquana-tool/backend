import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import ModelZooPage from "./ModelZooPage";
import {
    getAllModels,
    startSemanticTraining,
    getSemanticTrainingStatus,
    cancelSemanticTraining,
} from "../api/training";
import {
    getModelFavorites,
    setModelFavorite,
    clearModelFavorite,
} from "../api/models";

const addToast = vi.hoisted(() => vi.fn());

vi.mock("react-router-dom", () => ({
    useNavigate: () => vi.fn(),
    useLocation: () => ({ state: {} }),
    useParams: () => ({}),
}));

vi.mock("../contexts/AuthContext", () => ({
    useAuth: () => ({ isAuthenticated: true, user: { username: "tester" } }),
}));

vi.mock("../contexts/ToastContext", () => ({
    useToast: () => ({ addToast }),
}));

vi.mock("../api/training", () => ({
    getAllModels: vi.fn(),
    startSemanticTraining: vi.fn(),
    getSemanticTrainingStatus: vi.fn(),
    cancelSemanticTraining: vi.fn(),
}));

vi.mock("../api/models", () => ({
    getModelFavorites: vi.fn(),
    setModelFavorite: vi.fn(),
    clearModelFavorite: vi.fn(),
}));

describe("ModelZooPage task-specific favorites", () => {
    const mockModels = [
        {
            identifier: "model-dual",
            name: "Dual Task Model",
            tasks: ["instance-segmentation", "prompted-segmentation"],
            description: "Dual purpose model",
            parameters: {},
        },
        {
            identifier: "model-inst-only",
            name: "Instance Specialist",
            tasks: ["instance-segmentation"],
            description: "Only instance",
            parameters: {},
        },
    ];

    beforeEach(() => {
        vi.clearAllMocks();
        getAllModels.mockResolvedValue({
            success: true,
            models: mockModels,
        });
        // Model is favorited ONLY for prompted-segmentation, NOT for instance-segmentation
        getModelFavorites.mockResolvedValue({
            success: true,
            favorites: {
                "prompted-segmentation": "model-dual",
            },
        });
        setModelFavorite.mockResolvedValue({ success: true });
        clearModelFavorite.mockResolvedValue({ success: true });
    });

    it("displays favorite when viewing all tasks, but NOT when viewing task where it is not favorited", async () => {
        render(<ModelZooPage />);

        await waitFor(() => {
            expect(screen.getByText("Dual Task Model")).toBeInTheDocument();
        });

        // Under 'All tasks', favorites facet button exists because favoriteCount > 0
        expect(screen.getByRole("button", { name: "Favorites" })).toBeInTheDocument();

        // Switch facet to 'Full-Image' (where model-dual is NOT favorited)
        const instFacetBtn = screen.getByRole("button", { name: "Full-Image" });
        fireEvent.click(instFacetBtn);

        // Under Full-Image, favorites facet button should NOT be rendered because favoriteCount is 0
        await waitFor(() => {
            expect(screen.queryByRole("button", { name: "Favorites" })).not.toBeInTheDocument();
        });
    });

    it("sets favorite for the active task when starred in a task-filtered view", async () => {
        render(<ModelZooPage />);

        await waitFor(() => {
            expect(screen.getByText("Dual Task Model")).toBeInTheDocument();
        });

        // Switch to 'Full-Image'
        const instFacetBtn = screen.getByRole("button", { name: "Full-Image" });
        fireEvent.click(instFacetBtn);

        // Find star button for Dual Task Model (in chip)
        const starBtn = screen.getAllByRole("button", { name: "Add to favorites" })[0];
        fireEvent.click(starBtn);

        expect(setModelFavorite).toHaveBeenCalledWith("instance-segmentation", "model-dual");
    });

    it("opens task chooser in All view for multi-task model and stars specific task", async () => {
        render(<ModelZooPage />);

        await waitFor(() => {
            expect(screen.getByText("Dual Task Model")).toBeInTheDocument();
        });

        // In 'All' view, click star on Dual Task Model chip
        const starBtn = screen.getAllByRole("button", { name: "Remove from favorites" })[0];
        fireEvent.click(starBtn);

        // Task chooser popover should appear with options
        expect(screen.getByText("Star default for:")).toBeInTheDocument();
        const instanceOpt = screen.getAllByRole("button", { name: /toggle favorite for instance segmentation/i })[0];
        fireEvent.click(instanceOpt);

        expect(setModelFavorite).toHaveBeenCalledWith("instance-segmentation", "model-dual");
    });

    it("loads models when favorites loading rejects", async () => {
        getModelFavorites.mockRejectedValueOnce(new Error("favorites unavailable"));

        render(<ModelZooPage />);

        await waitFor(() => {
            expect(screen.getByText("Dual Task Model")).toBeInTheDocument();
        });
        expect(screen.queryByText("Failed to load models")).not.toBeInTheDocument();
    });

    it("resyncs and toasts when setting a favorite resolves unsuccessfully", async () => {
        getModelFavorites
            .mockResolvedValueOnce({
                success: true,
                favorites: { "prompted-segmentation": "model-dual" },
            })
            .mockResolvedValueOnce({ success: true, favorites: {} });
        setModelFavorite.mockResolvedValueOnce({ success: false });

        render(<ModelZooPage />);

        await waitFor(() => {
            expect(screen.getByText("Dual Task Model")).toBeInTheDocument();
        });
        fireEvent.click(screen.getByRole("button", { name: "Full-Image" }));
        fireEvent.click(screen.getAllByRole("button", { name: "Add to favorites" })[0]);

        await waitFor(() => {
            expect(addToast).toHaveBeenCalledWith({
                message: "Couldn't update favorites. Try again.",
                type: "error",
            });
        });
        expect(setModelFavorite).toHaveBeenCalledWith("instance-segmentation", "model-dual");
        expect(getModelFavorites).toHaveBeenCalledTimes(2);
    });

    it("toasts safely when clearing fails and the favorites resync rejects", async () => {
        getModelFavorites
            .mockResolvedValueOnce({
                success: true,
                favorites: { "prompted-segmentation": "model-dual" },
            })
            .mockRejectedValueOnce(new Error("favorites unavailable"));
        clearModelFavorite.mockResolvedValueOnce({ success: false });

        render(<ModelZooPage />);

        await waitFor(() => {
            expect(screen.getByText("Dual Task Model")).toBeInTheDocument();
        });
        fireEvent.click(screen.getByRole("button", { name: "Prompted" }));
        fireEvent.click(screen.getAllByRole("button", { name: "Remove from favorites" })[0]);

        await waitFor(() => {
            expect(addToast).toHaveBeenCalledWith({
                message: "Couldn't update favorites. Try again.",
                type: "error",
            });
        });
        expect(clearModelFavorite).toHaveBeenCalledWith("prompted-segmentation");
        expect(getModelFavorites).toHaveBeenCalledTimes(2);
        expect(screen.getAllByRole("button", { name: "Remove from favorites" }).length).toBeGreaterThan(0);
    });
});
