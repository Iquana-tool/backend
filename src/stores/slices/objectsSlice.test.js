import { describe, it, expect, beforeEach } from "vitest";
import useAnnotationStore from "../useAnnotationStore";

describe("objectsSlice setObjectsFromHierarchy", () => {
    beforeEach(() => {
        useAnnotationStore.setState((state) => {
            state.objects = {
                list: [],
                colors: {},
                selectedIds: [],
                visible: {},
                locked: {},
            };
        });
    });

    it("populates objects correctly from raw root_contours shape", () => {
        const labelsMap = new Map([
            [1, "Coral"],
            ["1", "Coral"],
        ]);

        const rawHierarchy = {
            root_contours: [
                {
                    id: 101,
                    label_id: 1,
                    points: [{ x: 0.1, y: 0.2 }],
                    children: [],
                },
            ],
        };

        const store = useAnnotationStore.getState();
        store.setObjectsFromHierarchy(rawHierarchy, labelsMap);

        const updated = useAnnotationStore.getState().objects;
        expect(updated.list.length).toBe(1);
        expect(updated.list[0].id).toBe(101);
        expect(updated.list[0].label).toBe("Coral");
    });

    it("populates objects correctly from backend API envelope { contours: { root_contours: [...] } }", () => {
        const labelsMap = new Map([
            [2, "Bleached"],
            ["2", "Bleached"],
        ]);

        const backendEnvelope = {
            success: true,
            message: "Contours hierarchy retrieved.",
            contours: {
                root_contours: [
                    {
                        id: 202,
                        label_id: 2,
                        points: [{ x: 0.3, y: 0.4 }],
                        children: [
                            {
                                id: 203,
                                label_id: 2,
                                points: [{ x: 0.35, y: 0.45 }],
                                children: [],
                            },
                        ],
                    },
                ],
            },
        };

        const store = useAnnotationStore.getState();
        store.setObjectsFromHierarchy(backendEnvelope, labelsMap);

        const updated = useAnnotationStore.getState().objects;
        expect(updated.list.length).toBe(2);
        expect(updated.list[0].id).toBe(202);
        expect(updated.list[0].label).toBe("Bleached");
        expect(updated.list[1].id).toBe(203);
        expect(updated.list[1].parent_id).toBe(202);
    });
});
