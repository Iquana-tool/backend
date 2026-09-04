import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import CrossImageSuggestionCard from "./CrossImageSuggestionCard";
import * as useInferenceConfigSuggestionModule from "./useInferenceConfigSuggestion";
import * as annotationSelectors from "../../../stores/selectors/annotationSelectors";

describe("CrossImageSuggestionCard", () => {
    const setActiveLabelId = vi.fn();
    const suggestLabel = vi.fn();

    const mockLabels = [
        { id: 1, name: "Coral", color: "#FF0000" },
        { id: 2, name: "Bleached", color: "#00FF00" },
    ];

    beforeEach(() => {
        vi.clearAllMocks();

        vi.spyOn(annotationSelectors, "useDatasetLabels").mockReturnValue(mockLabels);
        vi.spyOn(annotationSelectors, "useActiveLabelId").mockReturnValue(1);
        vi.spyOn(annotationSelectors, "useSetActiveLabelId").mockReturnValue(setActiveLabelId);
        vi.spyOn(annotationSelectors, "useLabelColorOverrides").mockReturnValue({});
    });

    it("renders closed by default and expands on toggle", () => {
        vi.spyOn(useInferenceConfigSuggestionModule, "useInferenceConfigSuggestion").mockReturnValue({
            isLoadingConfig: false,
            configError: null,
            isConfigured: vi.fn().mockReturnValue(true),
            getResolvedBinding: vi.fn().mockReturnValue({
                model: { name: "SAM 3 - Suggestion" },
                binding: { model_registry_key: "sam3" },
            }),
            isRunning: vi.fn().mockReturnValue(false),
            isAnyRunning: false,
            suggestLabel,
        });

        render(<CrossImageSuggestionCard />);

        const headerBtn = screen.getByRole("button", { name: /cross-image suggestion/i });
        expect(headerBtn).toBeInTheDocument();
        expect(screen.queryByText("SAM 3 - Suggestion")).not.toBeInTheDocument();

        // Expand card
        fireEvent.click(headerBtn);
        expect(screen.getByText("SAM 3 - Suggestion")).toBeInTheDocument();
    });

    it("renders configured model and executes suggestLabel on click", () => {
        vi.spyOn(useInferenceConfigSuggestionModule, "useInferenceConfigSuggestion").mockReturnValue({
            isLoadingConfig: false,
            configError: null,
            isConfigured: vi.fn().mockReturnValue(true),
            getResolvedBinding: vi.fn().mockReturnValue({
                model: { name: "SAM 3 - Suggestion" },
                binding: { model_registry_key: "sam3" },
            }),
            isRunning: vi.fn().mockReturnValue(false),
            isAnyRunning: false,
            suggestLabel,
        });

        render(<CrossImageSuggestionCard />);

        // Open card
        fireEvent.click(screen.getByRole("button", { name: /cross-image suggestion/i }));

        const suggestBtn = screen.getByRole("button", { name: "Suggest Coral" });
        expect(suggestBtn).toBeInTheDocument();
        expect(suggestBtn).not.toBeDisabled();

        fireEvent.click(suggestBtn);
        expect(suggestLabel).toHaveBeenCalledWith(1);
    });

    it("disables suggest button when model is not configured for label", () => {
        vi.spyOn(useInferenceConfigSuggestionModule, "useInferenceConfigSuggestion").mockReturnValue({
            isLoadingConfig: false,
            configError: null,
            isConfigured: vi.fn().mockReturnValue(false),
            getResolvedBinding: vi.fn().mockReturnValue(null),
            isRunning: vi.fn().mockReturnValue(false),
            isAnyRunning: false,
            suggestLabel,
        });

        render(<CrossImageSuggestionCard />);

        // Open card
        fireEvent.click(screen.getByRole("button", { name: /cross-image suggestion/i }));

        const suggestBtn = screen.getByRole("button", { name: "Suggest Coral" });
        expect(suggestBtn).toBeDisabled();
    });

    it("shows loading state when suggestion is running", () => {
        vi.spyOn(useInferenceConfigSuggestionModule, "useInferenceConfigSuggestion").mockReturnValue({
            isLoadingConfig: false,
            configError: null,
            isConfigured: vi.fn().mockReturnValue(true),
            getResolvedBinding: vi.fn().mockReturnValue({
                model: { name: "SAM 3" },
            }),
            isRunning: vi.fn().mockReturnValue(true),
            isAnyRunning: true,
            suggestLabel,
        });

        render(<CrossImageSuggestionCard />);

        // Open card
        fireEvent.click(screen.getByRole("button", { name: /cross-image suggestion/i }));

        expect(screen.getByText("Suggesting…")).toBeInTheDocument();
    });

    it("changing target label dropdown switches active label", () => {
        vi.spyOn(useInferenceConfigSuggestionModule, "useInferenceConfigSuggestion").mockReturnValue({
            isLoadingConfig: false,
            configError: null,
            isConfigured: vi.fn().mockReturnValue(true),
            getResolvedBinding: vi.fn().mockReturnValue({
                model: { name: "SAM 3" },
            }),
            isRunning: vi.fn().mockReturnValue(false),
            isAnyRunning: false,
            suggestLabel,
        });

        render(<CrossImageSuggestionCard />);

        // Open card
        fireEvent.click(screen.getByRole("button", { name: /cross-image suggestion/i }));

        const select = screen.getByLabelText("Target label for suggestion");
        fireEvent.change(select, { target: { value: "2" } });
        expect(setActiveLabelId).toHaveBeenCalledWith(2);
    });
});
