import { describe, it, expect, vi, beforeEach } from "vitest";
import useAnnotationStore from "../useAnnotationStore";
import * as modelsApi from "../../api/models";

vi.mock("../../api/models", () => ({
    getModelFavorites: vi.fn(),
    setModelFavorite: vi.fn(),
    clearModelFavorite: vi.fn(),
    getPromptedModels: vi.fn(),
    getSuggestionModels: vi.fn(),
}));

describe("modelsSlice favorites", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        modelsApi.getModelFavorites.mockResolvedValue({
            success: true,
            favorites: { "prompted-segmentation": "sam2-cell" },
        });
        modelsApi.setModelFavorite.mockResolvedValue({ success: true });
        modelsApi.clearModelFavorite.mockResolvedValue({ success: true });

        // Reset store models state
        useAnnotationStore.setState((state) => {
            state.models.favorites = {};
            state.models.favoritesLoaded = false;
            state.models.availablePromptedModels = [];
            state.models.isLoadingModels = false;
            state.models.promptedModel = null;
        });
    });

    it("fetches favorites on ensureFavoritesLoaded when favoritesLoaded is false, and caches it", async () => {
        const store = useAnnotationStore.getState();
        expect(store.models.favoritesLoaded).toBe(false);

        await store.ensureFavoritesLoaded();

        expect(modelsApi.getModelFavorites).toHaveBeenCalledTimes(1);
        expect(useAnnotationStore.getState().models.favorites).toEqual({
            "prompted-segmentation": "sam2-cell",
        });
        expect(useAnnotationStore.getState().models.favoritesLoaded).toBe(true);

        // Second call should not refetch
        await useAnnotationStore.getState().ensureFavoritesLoaded();
        expect(modelsApi.getModelFavorites).toHaveBeenCalledTimes(1);
    });

    it("degrades a success:false favorites response and still loads prompted models", async () => {
        modelsApi.getModelFavorites.mockResolvedValue({ success: false });
        modelsApi.getPromptedModels.mockResolvedValue({
            success: true,
            models: [{ id: "prompted-1", name: "Prompted model" }],
        });

        await useAnnotationStore.getState().fetchAvailablePromptedModels();

        expect(useAnnotationStore.getState().models.favorites).toEqual({});
        expect(useAnnotationStore.getState().models.favoritesLoaded).toBe(true);
        expect(modelsApi.getPromptedModels).toHaveBeenCalledTimes(1);
        expect(useAnnotationStore.getState().models.availablePromptedModels).toHaveLength(1);
        expect(useAnnotationStore.getState().models.isLoadingModels).toBe(false);
    });

    it("degrades a rejected favorites request and still loads prompted models", async () => {
        modelsApi.getModelFavorites.mockRejectedValue(new Error("favorites unavailable"));
        modelsApi.getPromptedModels.mockResolvedValue({
            success: true,
            models: [{ id: "prompted-1", name: "Prompted model" }],
        });

        await useAnnotationStore.getState().fetchAvailablePromptedModels();

        expect(useAnnotationStore.getState().models.favorites).toEqual({});
        expect(useAnnotationStore.getState().models.favoritesLoaded).toBe(true);
        expect(modelsApi.getPromptedModels).toHaveBeenCalledTimes(1);
        expect(useAnnotationStore.getState().models.availablePromptedModels).toHaveLength(1);
        expect(useAnnotationStore.getState().models.isLoadingModels).toBe(false);
    });

    it("sets favorite model optimistically and calls setModelFavorite", async () => {
        const store = useAnnotationStore.getState();

        await store.setFavoriteModel("instance-segmentation", "m2f-base");

        expect(useAnnotationStore.getState().models.favorites["instance-segmentation"]).toBe("m2f-base");
        expect(modelsApi.setModelFavorite).toHaveBeenCalledWith("instance-segmentation", "m2f-base");
    });

    it("clears favorite model optimistically and calls clearModelFavorite", async () => {
        useAnnotationStore.setState((state) => {
            state.models.favorites = { "prompted-segmentation": "sam2-cell" };
        });

        const store = useAnnotationStore.getState();
        await store.clearFavoriteModel("prompted-segmentation");

        expect(useAnnotationStore.getState().models.favorites["prompted-segmentation"]).toBeUndefined();
        expect(modelsApi.clearModelFavorite).toHaveBeenCalledWith("prompted-segmentation");
    });

    it("resyncs favorites when an optimistic update rejects", async () => {
        modelsApi.setModelFavorite.mockRejectedValueOnce(new Error("update failed"));
        modelsApi.getModelFavorites.mockResolvedValueOnce({
            success: true,
            favorites: { "prompted-segmentation": "server-default" },
        });

        await useAnnotationStore
            .getState()
            .setFavoriteModel("prompted-segmentation", "local-choice");

        expect(useAnnotationStore.getState().models.favorites).toEqual({
            "prompted-segmentation": "server-default",
        });
    });

    it("restores known favorites when mutation and resync both fail", async () => {
        useAnnotationStore.setState((state) => {
            state.models.favorites = { "prompted-segmentation": "sam2-cell" };
            state.models.favoritesLoaded = true;
        });
        modelsApi.clearModelFavorite.mockRejectedValueOnce(new Error("update failed"));
        modelsApi.getModelFavorites.mockRejectedValueOnce(new Error("refresh failed"));

        await useAnnotationStore.getState().clearFavoriteModel("prompted-segmentation");

        expect(useAnnotationStore.getState().models.favorites).toEqual({
            "prompted-segmentation": "sam2-cell",
        });
    });

    it("toggles favorite between set and clear", async () => {
        const store = useAnnotationStore.getState();

        // Initially empty -> sets favorite
        await store.toggleFavorite("prompted-segmentation", "sam2-cell");
        expect(useAnnotationStore.getState().models.favorites["prompted-segmentation"]).toBe("sam2-cell");
        expect(modelsApi.setModelFavorite).toHaveBeenCalledWith("prompted-segmentation", "sam2-cell");

        // Already favorited with same model -> clears favorite
        await store.toggleFavorite("prompted-segmentation", "sam2-cell");
        expect(useAnnotationStore.getState().models.favorites["prompted-segmentation"]).toBeUndefined();
        expect(modelsApi.clearModelFavorite).toHaveBeenCalledWith("prompted-segmentation");
    });

    it("deduplicates concurrent in-flight fetchModelFavorites calls", async () => {
        const store = useAnnotationStore.getState();

        // Trigger multiple concurrent fetches before first finishes
        const [res1, res2, res3] = await Promise.all([
            store.fetchModelFavorites(),
            store.fetchModelFavorites(),
            store.fetchModelFavorites(),
        ]);

        expect(modelsApi.getModelFavorites).toHaveBeenCalledTimes(1);
        expect(res1).toEqual({ "prompted-segmentation": "sam2-cell" });
        expect(res2).toEqual({ "prompted-segmentation": "sam2-cell" });
        expect(res3).toEqual({ "prompted-segmentation": "sam2-cell" });
    });

    it("deduplicates a failed fetch and permits a later retry", async () => {
        modelsApi.getModelFavorites.mockRejectedValueOnce(new Error("favorites unavailable"));
        const store = useAnnotationStore.getState();

        const results = await Promise.all([
            store.fetchModelFavorites(),
            store.fetchModelFavorites(),
            store.fetchModelFavorites(),
        ]);

        expect(modelsApi.getModelFavorites).toHaveBeenCalledTimes(1);
        expect(results).toEqual([{}, {}, {}]);
        expect(useAnnotationStore.getState().models.favoritesLoaded).toBe(true);

        useAnnotationStore.setState((state) => {
            state.models.favoritesLoaded = false;
        });
        await useAnnotationStore.getState().fetchModelFavorites();

        expect(modelsApi.getModelFavorites).toHaveBeenCalledTimes(2);
    });
});
